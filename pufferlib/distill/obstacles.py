"""Ellipses around the nearest partners, predicted over the MPC horizon.

Read from the partner block of the observation (drive.h write_partner_obs,
PARTNER_FEATURES per slot, nearest first): ego-frame position, length,
width, heading and signed speed. Each partner moves on at constant speed
along its heading. Its box is covered by an ellipse in its own frame,
widened by the ego's half-size, so the ego centre must stay outside:
((d . h) / a)^2 + ((d x h) / b)^2 >= 1 with d = p_ego - p_partner and h
the partner's heading. Everything is in the ego frame at the current step,
the frame of the MPC state x0 = (0, 0, 0, speed, steering).

Per obstacle and step the parameters are (px, py, cos, sin, a, b):
OBSTACLE_PARAM_COUNT values, as warpmpc_solver.py reads them.
"""

import math

import torch

from pufferlib.distill.bicycle import DT_SECONDS, MAX_SPEED_MPS

# Partner slot layout (drive.h write_partner_obs)
PARTNER_FEATURES = 9
PARTNER_X_IDX = 0
PARTNER_Y_IDX = 1
PARTNER_LENGTH_IDX = 3
PARTNER_WIDTH_IDX = 4
PARTNER_HEADING_X_IDX = 5
PARTNER_HEADING_Y_IDX = 6
PARTNER_SPEED_IDX = 7
# Ego observation layout (drive.h write_ego_obs)
EGO_WIDTH_IDX = 1
EGO_LENGTH_IDX = 2
# Normalizers of puffer_drive.yaml
OBS_NORM_XY_OFFSET_M = 120.0
OBS_NORM_VEH_LENGTH_M = 15.0
OBS_NORM_VEH_WIDTH_M = 10.0

OBSTACLE_PARAM_COUNT = 6
# The ellipse with the box's aspect ratio through its corners
ELLIPSE_PER_HALF_SIZE = math.sqrt(2.0)
# Empty slots: a unit circle this far ahead; its row is never active
# (value ~ 2.5e5, still exact in fp32)
DUMMY_DISTANCE_M = 500.0


def dummy_obstacles(
        horizon_steps: int,
        row_count: int,
        obstacle_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
    """
    Obstacles that never constrain the plan, for empty slots and padding.

    Args:
        horizon_steps (int): T.
        row_count (int): B.
        obstacle_count (int): K.
        dtype (torch.dtype): float dtype.
        device (torch.device): device.

    Returns:
        torch.Tensor: (T,B,K,6) unit circles at (DUMMY_DISTANCE_M, 0).
    """
    dummy = torch.tensor(
        [DUMMY_DISTANCE_M, 0.0, 1.0, 0.0, 1.0, 1.0],
        dtype=dtype,
        device=device,
    )
    return dummy.expand(
        horizon_steps, row_count, obstacle_count, OBSTACLE_PARAM_COUNT
    ).clone()


def partner_obstacles(
        obs: torch.Tensor,
        partner_start_idx: int,
        obstacle_count: int,
        horizon_steps: int,
    ) -> torch.Tensor:
    """
    Constant-velocity ellipses of the obstacle_count nearest partners at
    each of the T plan nodes (node k at k * DT_SECONDS).

    Args:
        obs (torch.Tensor): (B,obs_dim) observations.
        partner_start_idx (int): first index of the partner block: the
            ego block plus the reward and goal context (the backbone's
            ego_dim + context_dim).
        obstacle_count (int): K, nearest partners kept (<= slots).
        horizon_steps (int): T.

    Returns:
        torch.Tensor: (T,B,K,6) obstacle parameters; empty slots are
            dummy_obstacles.
    """
    row_count = obs.shape[0]
    partner_end = partner_start_idx + obstacle_count * PARTNER_FEATURES
    slots = obs[:, partner_start_idx:partner_end].reshape(
        row_count, obstacle_count, PARTNER_FEATURES
    )
    length_m = slots[..., PARTNER_LENGTH_IDX] * OBS_NORM_VEH_LENGTH_M
    width_m = slots[..., PARTNER_WIDTH_IDX] * OBS_NORM_VEH_WIDTH_M
    ego_length_m = obs[:, EGO_LENGTH_IDX, None] * OBS_NORM_VEH_LENGTH_M
    ego_width_m = obs[:, EGO_WIDTH_IDX, None] * OBS_NORM_VEH_WIDTH_M
    semi_axis_long = ELLIPSE_PER_HALF_SIZE * 0.5 * (length_m + ego_length_m)
    semi_axis_lat = ELLIPSE_PER_HALF_SIZE * 0.5 * (width_m + ego_width_m)
    cos_heading = slots[..., PARTNER_HEADING_X_IDX]
    sin_heading = slots[..., PARTNER_HEADING_Y_IDX]
    speed_mps = slots[..., PARTNER_SPEED_IDX] * MAX_SPEED_MPS

    times_s = DT_SECONDS * torch.arange(
        horizon_steps, dtype=obs.dtype, device=obs.device
    )[:, None, None]
    position_x = (
        slots[..., PARTNER_X_IDX] * OBS_NORM_XY_OFFSET_M
        + speed_mps * cos_heading * times_s
    )
    position_y = (
        slots[..., PARTNER_Y_IDX] * OBS_NORM_XY_OFFSET_M
        + speed_mps * sin_heading * times_s
    )
    per_partner = (cos_heading, sin_heading, semi_axis_long, semi_axis_lat)
    obstacles = torch.stack(
        [position_x, position_y]
        + [value.expand_as(position_x) for value in per_partner],
        dim=-1,
    )
    # An empty slot is all zeros (length 0)
    present = (length_m > 0)[None, :, :, None]
    dummy = dummy_obstacles(
        horizon_steps, row_count, obstacle_count, obs.dtype, obs.device
    )
    return torch.where(present, obstacles, dummy)
