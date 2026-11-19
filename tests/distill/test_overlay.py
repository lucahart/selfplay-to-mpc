"""Video overlay rows of the MPC student (overlay.py) and their hand-off
to the env (Drive.set_overlay)."""

import numpy as np
import pytest
import torch

from conftest import make_small_env, needs_teacher
from pufferlib.distill.bicycle import Bicycle
from pufferlib.distill.overlay import (
    NODE_ELLIPSES_START,
    NODE_PLANES_START,
    NODE_TARGET_Y_IDX,
    NODE_WEIGHT_Y_IDX,
    OVERLAY_ELLIPSE_COUNT_IDX,
    OVERLAY_FLOATS,
    OVERLAY_HEADER,
    OVERLAY_MAX_ELLIPSES,
    OVERLAY_MAX_NODES,
    OVERLAY_NODE_FLOATS,
    pack_overlay,
    planned_positions,
)
from pufferlib.distill.policy import ContinuousPolicy
from test_students import OBS_SEED, fresh_mpc_student
from test_teacher import rollout_obs


def node_block(
        rows: torch.Tensor,
    ) -> torch.Tensor:
    """
    Args:
        rows (torch.Tensor): (B,OVERLAY_FLOATS) overlay rows.

    Returns:
        torch.Tensor: (B,OVERLAY_MAX_NODES,OVERLAY_NODE_FLOATS) view.
    """
    return rows[:, OVERLAY_HEADER:].view(
        rows.shape[0], OVERLAY_MAX_NODES, OVERLAY_NODE_FLOATS
    )


def test_pack_overlay_places_every_block():
    horizon_steps, row_count, plane_count = 4, 3, 2
    positions = torch.randn(row_count, horizon_steps, 2)
    target_y = torch.randn(horizon_steps, row_count)
    weight_y = torch.rand(horizon_steps, row_count)
    planes = torch.randn(horizon_steps, row_count, plane_count, 3)
    rows = pack_overlay(positions, target_y, weight_y, planes)
    assert rows.shape == (row_count, OVERLAY_FLOATS)
    assert rows.dtype == torch.float32
    torch.testing.assert_close(
        rows[:, :OVERLAY_HEADER],
        torch.tensor([1.0, horizon_steps, plane_count, 0.0]).expand(
            row_count, OVERLAY_HEADER
        ),
    )
    nodes = node_block(rows)
    torch.testing.assert_close(nodes[:, :horizon_steps, :2], positions)
    torch.testing.assert_close(
        nodes[:, :horizon_steps, NODE_TARGET_Y_IDX], target_y.T
    )
    torch.testing.assert_close(
        nodes[:, :horizon_steps, NODE_WEIGHT_Y_IDX], weight_y.T
    )
    torch.testing.assert_close(
        nodes[:, :horizon_steps, NODE_PLANES_START:NODE_PLANES_START + 6],
        planes.transpose(0, 1).flatten(2),
    )
    assert bool((nodes[:, horizon_steps:] == 0).all())
    assert bool((nodes[:, :, NODE_PLANES_START + 6:] == 0).all())


def test_pack_overlay_places_partner_ellipses_after_the_planes():
    horizon_steps, row_count, ellipse_count = 3, 2, 4
    ellipses = torch.randn(horizon_steps, row_count, ellipse_count, 6)
    rows = pack_overlay(
        torch.zeros(row_count, horizon_steps, 2),
        torch.zeros(horizon_steps, row_count),
        torch.zeros(horizon_steps, row_count),
        torch.zeros(horizon_steps, row_count, 2, 3),
        ellipses,
    )
    assert bool((rows[:, OVERLAY_ELLIPSE_COUNT_IDX] == ellipse_count).all())
    nodes = node_block(rows)
    ellipses_end = NODE_ELLIPSES_START + 6 * ellipse_count
    torch.testing.assert_close(
        nodes[:, :horizon_steps, NODE_ELLIPSES_START:ellipses_end],
        ellipses.transpose(0, 1).flatten(2),
    )
    assert bool((nodes[:, :, ellipses_end:] == 0).all())
    with pytest.raises(ValueError):
        pack_overlay(
            torch.zeros(1, 1, 2), torch.zeros(1, 1), torch.zeros(1, 1),
            None, torch.zeros(1, 1, OVERLAY_MAX_ELLIPSES + 1, 6),
        )


def test_pack_overlay_rejects_long_horizons():
    with pytest.raises(ValueError):
        pack_overlay(
            torch.zeros(1, OVERLAY_MAX_NODES + 1, 2),
            torch.zeros(OVERLAY_MAX_NODES + 1, 1),
            torch.zeros(OVERLAY_MAX_NODES + 1, 1),
            None,
        )


def test_planned_positions_follow_the_bicycle():
    x0 = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.1]])
    wheelbase = torch.tensor([2.7])
    controls = torch.tensor([[[1.0, 0.2]], [[0.0, -0.1]], [[0.5, 0.0]]])
    positions = planned_positions(x0, wheelbase, controls)
    dynamics = Bicycle(wheelbase)
    state = x0
    for step_idx in range(3):
        torch.testing.assert_close(positions[:, step_idx], state[:, :2])
        state = dynamics(state, controls[step_idx])


def test_set_overlay_checks_its_input():
    env = make_small_env(seed=3)
    try:
        env.reset(seed=3)
        agent_count = env.observations.shape[0]
        rows = np.zeros((agent_count, OVERLAY_FLOATS), dtype=np.float32)
        env.set_overlay(rows)
        with pytest.raises(ValueError):
            env.set_overlay(rows[:, :-1])
        bad = rows.copy()
        bad[:, 0] = 1.0
        bad[:, OVERLAY_HEADER] = np.nan
        with pytest.raises(ValueError):
            env.set_overlay(bad)
        bad = rows.copy()
        bad[:, 0] = 1.0
        bad[:, OVERLAY_ELLIPSE_COUNT_IDX] = OVERLAY_MAX_ELLIPSES + 1
        with pytest.raises(ValueError):
            env.set_overlay(bad)
        # A valid row with partner ellipses is accepted
        good = rows.copy()
        good[:, 0] = 1.0
        good[:, 1] = 2.0
        good[:, OVERLAY_ELLIPSE_COUNT_IDX] = 1.0
        node_one = OVERLAY_HEADER + OVERLAY_NODE_FLOATS + NODE_ELLIPSES_START
        good[:, node_one:node_one + 6] = [5.0, 0.0, 1.0, 0.0, 4.0, 2.0]
        env.set_overlay(good)
    finally:
        env.close()


@needs_teacher
def test_policy_hands_out_the_mpc_students_overlay(teacher, device):
    student = fresh_mpc_student(teacher, device)
    policy = ContinuousPolicy(student)
    obs = rollout_obs(teacher, 3, OBS_SEED)[:20].clone()
    obs[::5] = 0.0
    policy.forward_eval(obs)
    # The first call switches recording on
    assert policy.render_overlay() is None
    policy.forward_eval(obs)
    rows = policy.render_overlay()
    assert rows.shape == (20, OVERLAY_FLOATS)
    assert bool((rows[::5] == 0).all())
    assert bool((rows[1::5, 0] == 1).all())
    assert bool(np.isfinite(rows).all())
    assert ContinuousPolicy(teacher).render_overlay() is None
