"""LabelBuffer keeps (obs, label) pairs together, FIFO, sampled uniformly."""

import numpy as np
import pytest
import torch

from pufferlib.distill.bicycle import STEERING, ego_state_from_obs
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.teacher import Teacher

from test_teacher import rollout_obs
from conftest import needs_teacher

OBS_DIM = 8
CAPACITY = 1000
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
UNIFORMITY_SAMPLE_COUNT = 200_000
# Chi-square with k-1 degrees of freedom: mean k-1, sd sqrt(2(k-1))
CHI_SQUARE_SIGMAS = 6
FP16_MAX = 65504.0
# 4 x 1024 teacher: 0.9977 (4 x 512: above 0.999); labels come from the fp32
# obs, so this bounds only how much the stored inputs lose
MIN_ARGMAX_AGREEMENT = 0.995


def rows_with_ids(
        first_id: int,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Obs and labels that carry their row id (exact in fp16 up to 2048).

    Args:
        first_id (int): id of the first row.
        row_count (int): number of rows.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: obs (n,OBS_DIM) with the id in
            every column, labels (n,2) with the id in column 0.
    """
    ids = torch.arange(first_id, first_id + row_count, dtype=torch.float32)
    obs = ids[:, None].expand(row_count, OBS_DIM).clone()
    labels = torch.stack([ids, -ids], -1)
    return obs, labels


@pytest.mark.parametrize("device", DEVICES)
def test_obs_and_labels_stay_paired(
        device: str,
    ) -> None:
    """Every sampled row's obs id equals its label id."""
    buffer = LabelBuffer(CAPACITY, OBS_DIM, device)
    obs, labels = rows_with_ids(0, 700)
    buffer.add(obs, labels)
    sampled_obs, sampled_labels = buffer.sample(4096)
    assert torch.equal(sampled_obs[:, 0], sampled_labels[:, 0])
    assert torch.equal(sampled_obs[:, -1], -sampled_labels[:, 1])


@pytest.mark.parametrize("device", DEVICES)
def test_fifo_keeps_exactly_the_newest_capacity_rows(
        device: str,
    ) -> None:
    """After 2.5x capacity in uneven chunks, the newest rows remain."""
    buffer = LabelBuffer(CAPACITY, OBS_DIM, device)
    added_count = 0
    for chunk_size in [300, 450, 999, 1, 250, 500]:
        obs, labels = rows_with_ids(added_count, chunk_size)
        buffer.add(obs, labels)
        added_count += chunk_size
    stored_ids = set(buffer.labels[:, 0].long().cpu().tolist())
    newest_ids = set(range(added_count - CAPACITY, added_count))
    assert buffer.filled_count == CAPACITY
    assert stored_ids == newest_ids


@pytest.mark.parametrize("device", DEVICES)
def test_sample_covers_filled_part_uniformly(
        device: str,
    ) -> None:
    """Only filled rows are sampled, with uniform frequency."""
    buffer = LabelBuffer(CAPACITY, OBS_DIM, device)
    filled_count = CAPACITY // 2
    buffer.add(*rows_with_ids(0, filled_count))
    _, labels = buffer.sample(UNIFORMITY_SAMPLE_COUNT)
    ids = labels[:, 0].long().cpu().numpy()
    assert ids.max() < filled_count
    counts = np.bincount(ids, minlength=filled_count)
    expected = UNIFORMITY_SAMPLE_COUNT / filled_count
    chi_square = ((counts - expected) ** 2 / expected).sum()
    degrees = filled_count - 1
    assert chi_square < degrees + CHI_SQUARE_SIGMAS * np.sqrt(2 * degrees)


@pytest.mark.parametrize("device", DEVICES)
def test_snapshot_is_evenly_spaced_oldest_first_and_rng_free(
        device: str,
    ) -> None:
    """Over a wrapped buffer the snapshot spans the newest rows in age
    order, whole or windowed, and does not advance sample()'s RNG."""
    buffer = LabelBuffer(CAPACITY, OBS_DIM, device)
    obs, labels = rows_with_ids(0, 700)
    buffer.add(obs, labels)
    obs, labels = rows_with_ids(700, 600)
    buffer.add(obs, labels)
    before = buffer.generator.get_state()

    snap_obs, snap_labels = buffer.snapshot(11)
    ids = snap_labels[:, 0].long().cpu()
    assert torch.equal(snap_obs[:, 0], snap_labels[:, 0])
    # Newest 1000 rows are ids 300..1299
    assert ids[0] == 300 and ids[-1] == 1299
    assert bool((ids[1:] > ids[:-1]).all())

    windowed = buffer.snapshot(5, window=101)[1][:, 0].long().cpu()
    assert windowed.tolist() == [1199, 1224, 1249, 1274, 1299]
    assert buffer.snapshot(50, window=10)[0].shape[0] == 10
    assert torch.equal(buffer.generator.get_state(), before)


def test_invalid_use_fails() -> None:
    """Empty sample, oversized add, wrong width and non-finite values."""
    buffer = LabelBuffer(CAPACITY, OBS_DIM, "cpu")
    with pytest.raises(ValueError):
        buffer.sample(1)
    with pytest.raises(ValueError):
        buffer.add(*rows_with_ids(0, CAPACITY + 1))
    with pytest.raises(ValueError):
        buffer.add(torch.zeros(3, OBS_DIM + 1), torch.zeros(3, 2))
    obs, labels = rows_with_ids(0, 3)
    obs[1, 2] = float("nan")
    with pytest.raises(ValueError):
        buffer.add(obs, labels)
    obs, labels = rows_with_ids(0, 3)
    labels[0, 1] = float("inf")
    with pytest.raises(ValueError):
        buffer.add(obs, labels)
    with pytest.raises(ValueError):
        LabelBuffer(0, OBS_DIM, "cpu")


@needs_teacher
def test_fp16_obs_keep_precision(
        teacher: Teacher,
    ) -> None:
    """Real obs fit fp16, and fp16 storage doesn't change the teacher."""
    obs = rollout_obs(teacher, 40, seed=31)
    assert float(obs.abs().max()) < FP16_MAX
    roundtrip = obs.half().float()
    agreement = (
        teacher.action_indices(obs) == teacher.action_indices(roundtrip)
    ).float().mean()
    print(f"\n  teacher argmax agreement after fp16: {float(agreement):.5f}")
    assert float(agreement) >= MIN_ARGMAX_AGREEMENT
    # Labels also depend on the steering read from obs
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    steering_roundtrip = ego_state_from_obs(roundtrip)[0][:, STEERING]
    assert float((steering - steering_roundtrip).abs().max()) < 1e-3
