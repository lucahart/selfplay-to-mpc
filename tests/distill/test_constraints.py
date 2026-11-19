"""Constraint types, plane prior and head mapping (constraints.py)."""

import math

import pytest
import torch

from pufferlib.distill.bicycle import DT_SECONDS
from pufferlib.distill.constraints import (
    CONSTRAINT_ELLIPSE,
    CONSTRAINT_NONE,
    CONSTRAINT_PLANE,
    CONSTRAINT_TUBE,
    FRONT_PLANE_MARGIN_M,
    LANE_HALF_WIDTH_M,
    PLANE_DUMMY_OFFSET_M,
    PLANE_HEAD_OUTPUTS,
    PLANE_OFFSET_SCALE_M,
    PLANE_PARAM_COUNT,
    dummy_constraints,
    plane_prior,
    planes_from_head,
    validate_constraints,
)
from pufferlib.distill.obstacles import (
    EGO_WIDTH_IDX,
    OBS_NORM_VEH_WIDTH_M,
)
from pufferlib.distill.students import MPCStudentConfig

HORIZON_STEPS = 10
EGO_WIDTH_M = 2.0
SPEED_MPS = 8.0
OBS_DIM = 16


def ego_obs(
        row_count: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    """
    Observations whose only set feature is the ego width.

    Args:
        row_count (int): B.
        dtype (torch.dtype): float dtype.

    Returns:
        torch.Tensor: (B,OBS_DIM) observations.
    """
    obs = torch.zeros(row_count, OBS_DIM, dtype=dtype)
    obs[:, EGO_WIDTH_IDX] = EGO_WIDTH_M / OBS_NORM_VEH_WIDTH_M
    return obs


def test_prior_is_lane_bounds_and_a_front_plane():
    speed = torch.full((2,), SPEED_MPS)
    width = torch.full((2,), EGO_WIDTH_M)
    angles, offsets = plane_prior(3, speed, width, HORIZON_STEPS)
    torch.testing.assert_close(
        angles, torch.tensor([0.0, math.pi, -0.5 * math.pi])
    )
    side = LANE_HALF_WIDTH_M - 0.5 * EGO_WIDTH_M
    torch.testing.assert_close(offsets[..., 0], torch.full((10, 2), side))
    torch.testing.assert_close(offsets[..., 1], torch.full((10, 2), side))
    coasting = SPEED_MPS * DT_SECONDS * torch.arange(HORIZON_STEPS)
    torch.testing.assert_close(
        offsets[..., 2], (coasting + FRONT_PLANE_MARGIN_M)[:, None].expand(
            HORIZON_STEPS, 2
        ),
    )


def test_zero_head_output_gives_the_prior():
    obs = ego_obs(2)
    plane_raw = torch.zeros(HORIZON_STEPS, 2, 3, PLANE_HEAD_OUTPUTS)
    speed = torch.full((2,), SPEED_MPS)
    planes = planes_from_head(plane_raw, speed, obs)
    assert planes.shape == (HORIZON_STEPS, 2, 3, PLANE_PARAM_COUNT)
    expected_normals = torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        planes[..., :2], expected_normals.expand_as(planes[..., :2]),
        atol=1e-6, rtol=0.0,
    )
    _, offsets = plane_prior(
        3, speed, torch.full((2,), EGO_WIDTH_M), HORIZON_STEPS
    )
    torch.testing.assert_close(planes[..., 2], offsets)


def test_head_output_turns_and_shifts_planes():
    obs = ego_obs(1)
    plane_raw = torch.zeros(HORIZON_STEPS, 1, 2, PLANE_HEAD_OUTPUTS)
    plane_raw[..., 0, 0] = 0.3
    plane_raw[..., 1, 1] = 0.5
    planes = planes_from_head(plane_raw, torch.zeros(1), obs)
    normal_norm = planes[..., :2].norm(dim=-1)
    torch.testing.assert_close(normal_norm, torch.ones_like(normal_norm))
    torch.testing.assert_close(
        planes[0, 0, 0, :2], torch.tensor([-math.sin(0.3), math.cos(0.3)])
    )
    side = LANE_HALF_WIDTH_M - 0.5 * EGO_WIDTH_M
    torch.testing.assert_close(
        planes[0, 0, 1, 2], torch.tensor(side + 0.5 * PLANE_OFFSET_SCALE_M)
    )


def test_angle_gradient_matches_finite_differences():
    # The unit normal is (-sin, cos) of the angle: autograd through it is
    # the gradient the student's head receives
    generator = torch.Generator().manual_seed(0)
    obs = ego_obs(4, torch.float64)
    speed = torch.full((4,), SPEED_MPS, dtype=torch.float64)
    plane_raw = 0.1 * torch.randn(
        HORIZON_STEPS, 4, 3, PLANE_HEAD_OUTPUTS, generator=generator,
        dtype=torch.float64,
    )
    weights = torch.randn(
        HORIZON_STEPS, 4, 3, PLANE_PARAM_COUNT, generator=generator,
        dtype=torch.float64,
    )
    leaf = plane_raw.clone().requires_grad_(True)
    (planes_from_head(leaf, speed, obs) * weights).sum().backward()
    step = 1e-6
    shift = torch.zeros_like(plane_raw)
    shift[..., 0] = step
    with torch.no_grad():
        numeric = (
            (planes_from_head(plane_raw + shift, speed, obs) * weights).sum()
            - (planes_from_head(plane_raw - shift, speed, obs)
               * weights).sum()
        ) / (2.0 * step)
    torch.testing.assert_close(
        leaf.grad[..., 0].sum(), numeric, rtol=1e-6, atol=1e-8
    )


def test_dummy_planes_never_bind_and_keep_unit_normals():
    dummies = dummy_constraints(
        CONSTRAINT_PLANE, HORIZON_STEPS, 3, 2, torch.float32,
        torch.device("cpu"),
    )
    assert dummies.shape == (HORIZON_STEPS, 3, 2, PLANE_PARAM_COUNT)
    torch.testing.assert_close(
        dummies[..., :2].norm(dim=-1), torch.ones(HORIZON_STEPS, 3, 2)
    )
    assert bool((dummies[..., 2] == PLANE_DUMMY_OFFSET_M).all())


@pytest.mark.parametrize("constraint_type, constraint_count", [
    ("box", 2),
    (CONSTRAINT_NONE, 2),
    (CONSTRAINT_PLANE, 0),
    (CONSTRAINT_ELLIPSE, -1),
])
def test_invalid_constraints_are_rejected(constraint_type, constraint_count):
    with pytest.raises(ValueError):
        validate_constraints(constraint_type, constraint_count)


@pytest.mark.parametrize("constraint_type", [CONSTRAINT_PLANE, CONSTRAINT_ELLIPSE])
def test_student_takes_only_the_tube(constraint_type):
    with pytest.raises(ValueError, match="constraint_type"):
        MPCStudentConfig(constraint_type=constraint_type, constraint_count=2)


def test_constraints_need_warpmpc():
    with pytest.raises(ValueError):
        MPCStudentConfig(constraint_type=CONSTRAINT_TUBE, constraint_count=2)
