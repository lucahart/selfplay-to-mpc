"""Constraint rows of the WarpMPC student: types, parameter layout,
padding dummies and the head-to-plane mapping.

Each constraint row has one slack s >= 0 priced in the solver's cost and
acts on nodes k >= 1 (node 0 is pinned to x0). Parameters arrive as
(T,B,count,P) tensors; the solver ignores step 0.

Planes are half-planes
on the ego-frame position, a . p <= b with a unit normal a = (-sin phi,
cos phi). The head outputs a residual on phi and b around a fixed prior:
straight lane bounds left and right of the ego that keep its body inside a
lane, and for a third plane a stop line just ahead of the coasting rollout.

The tube mixes row types on one node:
a left and a right plane fixed on the observed road edge (tube.py), then
constraint_count partner ellipses. Its parameters arrive flat, (T,B,V)
with V = constraint_value_count.

Torch only, so the student can import it without the WarpMPC venv.
"""

import math

import torch

from pufferlib.distill.bicycle import DT_SECONDS
from pufferlib.distill.obstacles import (
    EGO_WIDTH_IDX,
    OBS_NORM_VEH_WIDTH_M,
    OBSTACLE_PARAM_COUNT,
    dummy_obstacles,
)

# Constraint types (one per compiled solver)
CONSTRAINT_NONE = "none"
CONSTRAINT_PLANE = "plane"
CONSTRAINT_ELLIPSE = "ellipse"
CONSTRAINT_TUBE = "tube"
CONSTRAINT_TYPES = (
    CONSTRAINT_NONE, CONSTRAINT_PLANE, CONSTRAINT_ELLIPSE, CONSTRAINT_TUBE,
)

# Plane parameters per node: (a_px, a_py, b)
PLANE_PARAM_COUNT = 3
# Head outputs per plane and step: angle residual, offset residual
PLANE_HEAD_OUTPUTS = 2
# Prior normal angles in plane order: left (a = (0, 1)), right
# (a = (0, -1)), front (a = (1, 0))
PLANE_INIT_ANGLES_RAD = (0.0, math.pi, -0.5 * math.pi)
SIDE_PLANE_COUNT = 2
MIN_PLANE_COUNT = SIDE_PLANE_COUNT
MAX_PLANE_COUNT = len(PLANE_INIT_ANGLES_RAD)
# Half of a 3.7 m lane
LANE_HALF_WIDTH_M = 1.85
# The front prior sits this far ahead of the coasting position of node k
FRONT_PLANE_MARGIN_M = 0.5
# Offset change per unit of the head's offset output
PLANE_OFFSET_SCALE_M = 2.0
# Padding / empty slot: p_y <= this, never active; unit normal keeps its
# row gradient at 1 (large row gradients disturb the ADMM, see
# ELLIPSE_DISTANCE_EPS in warpmpc_solver.py)
PLANE_DUMMY_OFFSET_M = 500.0
# Tube: a left and a right plane on every node, before the ellipses
TUBE_PLANE_COUNT = 2
TUBE_PLANE_VALUE_COUNT = TUBE_PLANE_COUNT * PLANE_PARAM_COUNT


def constraint_param_count(
        constraint_type: str,
    ) -> int:
    """
    Solver parameters per constraint row and node, for the types with one
    kind of row; the tube mixes kinds (constraint_value_count).

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES but the tube.

    Returns:
        int: 0, PLANE_PARAM_COUNT or OBSTACLE_PARAM_COUNT.
    """
    if constraint_type == CONSTRAINT_TUBE:
        raise ValueError("the tube has no single row size")
    if constraint_type == CONSTRAINT_PLANE:
        return PLANE_PARAM_COUNT
    if constraint_type == CONSTRAINT_ELLIPSE:
        return OBSTACLE_PARAM_COUNT
    return 0


def constraint_value_count(
        constraint_type: str,
        constraint_count: int,
    ) -> int:
    """
    Solver parameters per node k >= 1.

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES.
        constraint_count (int): rows per node; ellipses for the tube.

    Returns:
        int: count * P, or TUBE_PLANE_VALUE_COUNT + count * 6 for the tube.
    """
    if constraint_type == CONSTRAINT_TUBE:
        return TUBE_PLANE_VALUE_COUNT + (
            constraint_count * OBSTACLE_PARAM_COUNT
        )
    return constraint_count * constraint_param_count(constraint_type)


def constraint_slack_count(
        constraint_type: str,
        constraint_count: int,
    ) -> int:
    """
    Constraint rows, each with one slack, per node k >= 1.

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES.
        constraint_count (int): rows per node; ellipses for the tube.

    Returns:
        int: count, plus TUBE_PLANE_COUNT for the tube.
    """
    if constraint_type == CONSTRAINT_TUBE:
        return TUBE_PLANE_COUNT + constraint_count
    return constraint_count


def constraint_count_from_params(
        constraint_type: str,
        constraints: torch.Tensor,
    ) -> int:
    """
    constraint_count of a parameter tensor.

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES but none.
        constraints (torch.Tensor): (T,B,C,P), or (T,B,V) for the tube.

    Returns:
        int: C, or the tube's ellipse count.
    """
    if constraint_type != CONSTRAINT_TUBE:
        return constraints.shape[2]
    ellipse_values = constraints.shape[2] - TUBE_PLANE_VALUE_COUNT
    if ellipse_values < 0 or ellipse_values % OBSTACLE_PARAM_COUNT != 0:
        raise ValueError("tube parameters do not fit planes and ellipses")
    return ellipse_values // OBSTACLE_PARAM_COUNT


def constraint_head_outputs(
        constraint_type: str,
    ) -> int:
    """
    Head outputs per constraint row and step; ellipses come from the
    observation and add none.

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES.

    Returns:
        int: PLANE_HEAD_OUTPUTS for planes, else 0.
    """
    if constraint_type == CONSTRAINT_PLANE:
        return PLANE_HEAD_OUTPUTS
    return 0


def validate_constraints(
        constraint_type: str,
        constraint_count: int,
    ) -> None:
    """
    Reject an unknown type and a count that does not fit it. The student's
    plane prior further needs MIN_PLANE_COUNT..MAX_PLANE_COUNT planes
    (MPCStudentConfig); the solver takes any count.

    Args:
        constraint_type (str): one of CONSTRAINT_TYPES.
        constraint_count (int): rows per node; ellipses for the tube.
    """
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(f"constraint_type must be one of {CONSTRAINT_TYPES}")
    if constraint_type == CONSTRAINT_NONE:
        if constraint_count != 0:
            raise ValueError("constraint_type=none needs constraint_count=0")
        return
    # The tube's planes are always there; its ellipses may be none
    if constraint_type == CONSTRAINT_TUBE:
        if constraint_count < 0:
            raise ValueError("constraint_count must be >= 0")
        return
    if constraint_count <= 0:
        raise ValueError("constraint_count must be positive")


def dummy_constraints(
        constraint_type: str,
        horizon_steps: int,
        row_count: int,
        constraint_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
    """
    Constraint rows that never bind, for padding rows and absent edges.

    Args:
        constraint_type (str): CONSTRAINT_PLANE, CONSTRAINT_ELLIPSE or
            CONSTRAINT_TUBE.
        horizon_steps (int): T.
        row_count (int): B.
        constraint_count (int): rows per node; ellipses for the tube.
        dtype (torch.dtype): float dtype.
        device (torch.device): device.

    Returns:
        torch.Tensor: (T,B,count,P) parameters, (T,B,V) for the tube.
    """
    if constraint_type == CONSTRAINT_ELLIPSE:
        return dummy_obstacles(
            horizon_steps, row_count, constraint_count, dtype, device
        )
    dummy = torch.tensor(
        [0.0, 1.0, PLANE_DUMMY_OFFSET_M], dtype=dtype, device=device
    )
    if constraint_type == CONSTRAINT_PLANE:
        return dummy.expand(
            horizon_steps, row_count, constraint_count, PLANE_PARAM_COUNT
        ).clone()
    planes = dummy.repeat(TUBE_PLANE_COUNT).expand(
        horizon_steps, row_count, TUBE_PLANE_VALUE_COUNT
    )
    ellipses = dummy_obstacles(
        horizon_steps, row_count, constraint_count, dtype, device
    )
    return torch.cat([planes, ellipses.flatten(2)], dim=-1)


def plane_prior(
        plane_count: int,
        speed_mps: torch.Tensor,
        ego_width_m: torch.Tensor,
        horizon_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Planes at zero head output: side planes at the lane half-width minus
    the ego's half-width (the body stays in a straight lane through the
    ego), a front plane FRONT_PLANE_MARGIN_M ahead of coasting at node k.

    Args:
        plane_count (int): H, MIN_PLANE_COUNT..MAX_PLANE_COUNT.
        speed_mps (torch.Tensor): (B,) current signed speed [m/s].
        ego_width_m (torch.Tensor): (B,) ego width [m].
        horizon_steps (int): T.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: normal angles (H,) [rad] and
            offsets (T,B,H) [m].
    """
    angles = torch.tensor(
        PLANE_INIT_ANGLES_RAD[:plane_count],
        dtype=speed_mps.dtype,
        device=speed_mps.device,
    )
    row_count = speed_mps.shape[0]
    side_offset = LANE_HALF_WIDTH_M - 0.5 * ego_width_m
    offsets = [
        side_offset.expand(horizon_steps, row_count)
        for _ in range(min(plane_count, SIDE_PLANE_COUNT))
    ]
    if plane_count > SIDE_PLANE_COUNT:
        times_s = DT_SECONDS * torch.arange(
            horizon_steps, dtype=speed_mps.dtype, device=speed_mps.device
        )
        offsets.append(
            speed_mps * times_s[:, None] + FRONT_PLANE_MARGIN_M
        )
    return angles, torch.stack(offsets, dim=-1)


def planes_from_head(
        plane_raw: torch.Tensor,
        speed_mps: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Plane parameters from the head's plane outputs: residuals on the
    prior's normal angle and offset (plane_prior).

    Args:
        plane_raw (torch.Tensor): (T,B,H,PLANE_HEAD_OUTPUTS) head outputs
            (angle, offset).
        speed_mps (torch.Tensor): (B,) current signed speed [m/s].
        obs (torch.Tensor): (B,obs_dim) observations (ego width).

    Returns:
        torch.Tensor: (T,B,H,PLANE_PARAM_COUNT) (a_px, a_py, b), a unit
            and differentiable with respect to plane_raw.
    """
    horizon_steps, _, plane_count = plane_raw.shape[:3]
    ego_width_m = obs[:, EGO_WIDTH_IDX].to(plane_raw.dtype) * (
        OBS_NORM_VEH_WIDTH_M
    )
    prior_angles, prior_offsets = plane_prior(
        plane_count, speed_mps, ego_width_m, horizon_steps
    )
    angles = prior_angles + plane_raw[..., 0]
    offsets = prior_offsets + PLANE_OFFSET_SCALE_M * plane_raw[..., 1]
    return torch.stack(
        [-torch.sin(angles), torch.cos(angles), offsets], dim=-1
    )
