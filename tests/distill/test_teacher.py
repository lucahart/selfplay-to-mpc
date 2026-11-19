"""Teacher labels are the PPO policy's argmax, decoded as drive.h does."""

import pickle
import types

import numpy as np
import pytest
import torch

import pufferlib.pytorch
from pufferlib import pufferl
from pufferlib.distill.bicycle import (
    ACCELERATION_VALUES,
    DT_SECONDS,
    MAX_STEERING_RATE_RADPS,
    STEERING,
    STEERING_VALUES,
    U_HI,
    U_LO,
    ego_state_from_obs,
    env_action_from_control,
)
from pufferlib.distill.env import TEACHER_CHECKPOINT, make_env
from pufferlib.distill.teacher import Teacher
from pufferlib.ocean import torch as ocean_torch

from conftest import make_small_env, needs_teacher

ROLLOUT_STEPS = 40
TEACHER_SEED = 21


def rollout_obs(
        teacher: Teacher,
        step_count: int,
        seed: int,
    ) -> torch.Tensor:
    """
    Observations of a teacher-driven rollout, not only the reset obs.

    Args:
        teacher (Teacher): drives the env.
        step_count (int): env steps.
        seed (int): env seed.

    Returns:
        torch.Tensor: (step_count*B,obs_dim) observations on the
            teacher's device.
    """
    env = make_small_env(seed)
    obs, _ = env.reset(seed=seed)
    collected = []
    for _ in range(step_count):
        obs_tensor = torch.from_numpy(obs).to(teacher.device)
        collected.append(obs_tensor)
        steering = ego_state_from_obs(obs_tensor)[0][:, STEERING]
        control = teacher.label(obs_tensor, steering)
        action = env_action_from_control(control, steering)
        obs, _, _, _, _ = env.step(action.cpu().numpy())
    env.close()
    return torch.cat(collected)


def drive_h_discrete_rule(
        action_idx: torch.Tensor,
        steering_rad: torch.Tensor,
    ) -> torch.Tensor:
    """
    drive.h's discrete step written out independently: table accel, and
    the steering change towards the table target (rate clip), as a rate.

    Args:
        action_idx (torch.Tensor): (B,) discrete action index.
        steering_rad (torch.Tensor): (B,) current steering [rad].

    Returns:
        torch.Tensor: (B,2) control (accel, steering_rate).
    """
    dtype = steering_rad.dtype
    accel_table = torch.tensor(ACCELERATION_VALUES, dtype=dtype)
    steering_table = torch.tensor(STEERING_VALUES, dtype=dtype)
    accel = accel_table[action_idx.cpu() // 9]
    target = steering_table[action_idx.cpu() % 9]
    max_change = MAX_STEERING_RATE_RADPS * DT_SECONDS
    change = (target - steering_rad.cpu()).clamp(-max_change, max_change)
    return torch.stack([accel, change / DT_SECONDS], -1)


@needs_teacher
def test_labels_equal_ppo_argmax_decoded(
        teacher: Teacher,
        distill_args: dict,
        device: str,
    ) -> None:
    """Labels match pufferl.load_policy + sample_logits + drive.h rule."""
    discrete_env, _ = make_env("discrete", 0, 64)
    args = dict(
        distill_args,
        load_id=None,
        load_model_path=TEACHER_CHECKPOINT,
        train=dict(distill_args["train"], device=device),
    )
    vecenv = types.SimpleNamespace(driver_env=discrete_env)
    policy = pufferl.load_policy(args, vecenv, "puffer_drive").eval()
    discrete_env.close()

    obs = rollout_obs(teacher, ROLLOUT_STEPS, TEACHER_SEED)
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs)
        action_idx, _, _ = pufferlib.pytorch.sample_logits(
            logits, deterministic=True
        )
    expected = drive_h_discrete_rule(action_idx.view(-1), steering)
    labels = teacher.label(obs, steering).cpu()
    assert torch.allclose(labels, expected.to(labels.dtype), atol=1e-6)
    # Several distinct actions, so the comparison is not trivial
    assert action_idx.unique().numel() > 5


@needs_teacher
def test_teacher_is_frozen_and_deterministic(
        teacher: Teacher,
    ) -> None:
    """No trainable parameters, eval mode, same labels twice."""
    assert not any(p.requires_grad for p in teacher.policy.parameters())
    assert not teacher.policy.training
    obs = rollout_obs(teacher, 5, TEACHER_SEED)
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    assert torch.equal(
        teacher.label(obs, steering), teacher.label(obs, steering)
    )


@needs_teacher
def test_labels_inside_control_box(
        teacher: Teacher,
    ) -> None:
    """Labels are (B,2), on the obs device, within the control box."""
    obs = rollout_obs(teacher, ROLLOUT_STEPS, TEACHER_SEED)
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    labels = teacher.label(obs, steering)
    assert labels.shape == (obs.shape[0], 2)
    assert labels.device == obs.device
    low, high = U_LO.to(labels), U_HI.to(labels)
    assert bool(((labels >= low) & (labels <= high)).all())


@needs_teacher
def test_bad_checkpoint_is_rejected(
        tmp_path,
        distill_args: dict,
        device: str,
    ) -> None:
    """Missing, truncated and continuous-head checkpoints fail at once."""
    with pytest.raises(FileNotFoundError):
        Teacher(str(tmp_path / "missing.pt"), distill_args, device)

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(open(TEACHER_CHECKPOINT, "rb").read()[:1000])
    with pytest.raises((RuntimeError, EOFError, pickle.UnpicklingError)):
        Teacher(str(truncated), distill_args, device)

    continuous_env, _ = make_env("continuous", 0, 64)
    continuous_policy = getattr(ocean_torch, distill_args["policy_name"])(
        continuous_env, **distill_args["policy"]
    )
    continuous_env.close()
    continuous_path = tmp_path / "continuous.pt"
    torch.save(continuous_policy.state_dict(), continuous_path)
    with pytest.raises(RuntimeError):
        Teacher(str(continuous_path), distill_args, device)


def test_drive_h_rule_matches_bicycle_decoding() -> None:
    """The test's own rule and bicycle.control_from_discrete agree on all
    63 actions over a grid of steering angles."""
    from pufferlib.distill.bicycle import control_from_discrete

    action_idx = torch.arange(63).repeat_interleave(11)
    steering = torch.linspace(-0.667, 0.667, 11).repeat(63).double()
    expected = drive_h_discrete_rule(action_idx, steering)
    decoded = control_from_discrete(action_idx, steering)
    assert np.allclose(decoded.numpy(), expected.numpy(), atol=1e-12)
