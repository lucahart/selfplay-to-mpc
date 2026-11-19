"""Partner ellipses from the observation (obstacles.py)."""

import math

import torch

from pufferlib.distill.bicycle import DT_SECONDS, MAX_SPEED_MPS
from pufferlib.distill.obstacles import (
    DUMMY_DISTANCE_M,
    EGO_LENGTH_IDX,
    EGO_WIDTH_IDX,
    OBS_NORM_VEH_LENGTH_M,
    OBS_NORM_VEH_WIDTH_M,
    OBS_NORM_XY_OFFSET_M,
    PARTNER_FEATURES,
    partner_obstacles,
)

EGO_DIM = 10
SLOT_COUNT = 3


def observation_with_partner(
        x_m: float,
        y_m: float,
        heading_rad: float,
        speed_mps: float,
    ) -> torch.Tensor:
    """
    One observation: a 4.5 x 2 m ego and one 5 x 2 m partner in slot 0;
    the other slots are empty.

    Args:
        x_m (float): partner x in the ego frame [m].
        y_m (float): partner y in the ego frame [m].
        heading_rad (float): partner heading in the ego frame [rad].
        speed_mps (float): partner signed speed [m/s].

    Returns:
        torch.Tensor: (1,EGO_DIM + SLOT_COUNT * PARTNER_FEATURES).
    """
    obs = torch.zeros(1, EGO_DIM + SLOT_COUNT * PARTNER_FEATURES)
    obs[0, EGO_LENGTH_IDX] = 4.5 / OBS_NORM_VEH_LENGTH_M
    obs[0, EGO_WIDTH_IDX] = 2.0 / OBS_NORM_VEH_WIDTH_M
    obs[0, EGO_DIM:EGO_DIM + PARTNER_FEATURES] = torch.tensor([
        x_m / OBS_NORM_XY_OFFSET_M,
        y_m / OBS_NORM_XY_OFFSET_M,
        0.0,
        5.0 / OBS_NORM_VEH_LENGTH_M,
        2.0 / OBS_NORM_VEH_WIDTH_M,
        math.cos(heading_rad),
        math.sin(heading_rad),
        speed_mps / MAX_SPEED_MPS,
        0.0,
    ])
    return obs


def test_partner_moves_at_constant_velocity():
    obs = observation_with_partner(10.0, 3.0, math.pi / 2, 5.0)
    obstacles = partner_obstacles(obs, EGO_DIM, SLOT_COUNT, 4)
    assert obstacles.shape == (4, 1, SLOT_COUNT, 6)
    times_s = DT_SECONDS * torch.arange(4.0)
    torch.testing.assert_close(obstacles[:, 0, 0, 0], torch.full((4,), 10.0))
    torch.testing.assert_close(obstacles[:, 0, 0, 1], 3.0 + 5.0 * times_s)
    torch.testing.assert_close(
        obstacles[0, 0, 0, 2:4], torch.tensor([0.0, 1.0]),
        atol=1e-6, rtol=0.0,
    )


def test_semi_axes_cover_both_cars():
    obs = observation_with_partner(10.0, 0.0, 0.0, 0.0)
    semi_axes = partner_obstacles(obs, EGO_DIM, SLOT_COUNT, 2)[0, 0, 0, 4:]
    expected = math.sqrt(2.0) * torch.tensor([0.5 * 9.5, 0.5 * 4.0])
    torch.testing.assert_close(semi_axes, expected)


def test_empty_slots_are_far_away_dummies():
    obs = observation_with_partner(10.0, 0.0, 0.0, 0.0)
    obstacles = partner_obstacles(obs, EGO_DIM, SLOT_COUNT, 2)
    torch.testing.assert_close(
        obstacles[:, 0, 1:, 0], torch.full((2, SLOT_COUNT - 1),
                                           DUMMY_DISTANCE_M)
    )
