"""Fixed tube road bounds: edge segments, references (rollout, previous
plan shifted, straight), projection, sides, sector and offsets, route
anchors and route planes (tube.py), and the tube's config checks."""

import math
import types

import numpy as np
import pytest
import torch

from pufferlib.distill.bicycle import (
    DT_SECONDS,
    HEADING,
    PX,
    SPEED,
    Bicycle,
)
from pufferlib.distill.constraints import (
    CONSTRAINT_TUBE,
    PLANE_DUMMY_OFFSET_M,
    TUBE_PLANE_VALUE_COUNT,
    constraint_count_from_params,
    constraint_slack_count,
    constraint_value_count,
    dummy_constraints,
    validate_constraints,
)
from pufferlib.distill.obstacles import OBS_NORM_XY_OFFSET_M
from pufferlib.distill.students import MPCStudentConfig
from pufferlib.distill.tube import (
    OBS_NORM_ROAD_SEG_LENGTH_M,
    ROUTE_ANCHOR_MAX_DISTANCE_M,
    ROUTE_EDGE_PARALLEL_MAX_RAD,
    ROUTE_PLANE_MAX_OFFSET_M,
    ROUTE_WINDOW_SPACING_M,
    TUBE_NORMAL_HEADING,
    TUBE_NORMAL_PROJECTION,
    TUBE_SECTOR_FULL,
    TUBE_SECTOR_NARROW,
    planned_states,
    poses_from_global_state,
    projection_planes,
    reference_positions,
    road_edge_segments,
    route_anchors,
    route_planes,
    shifted_reference,
    straight_reference,
    tube_planes,
)

EGO_WIDTH_M = 2.0
MARGIN_M = 0.2
BODY_OFFSET_M = 0.5 * EGO_WIDTH_M + MARGIN_M
DUMMY_ROW = [0.0, 1.0, PLANE_DUMMY_OFFSET_M]


EGO_LENGTH_M = 4.6


def tube_rows(
        segments: list[tuple[float, float, float, float]],
        position: tuple[float, float] = (0.0, 0.0),
        heading_rad: float = 0.0,
        valid: list[bool] | None = None,
        ego_width_m: float = EGO_WIDTH_M,
        margin_m: float = MARGIN_M,
        normal_mode: str = TUBE_NORMAL_HEADING,
        sector_mode: str = TUBE_SECTOR_NARROW,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tube planes of one node against the given segments; the car is at the
    origin.

    Args:
        segments (list): per segment (centre x, centre y [m], direction
            angle [rad], half-length [m]).
        position (tuple): reference position p_hat [m].
        heading_rad (float): reference heading.
        valid (list[bool] | None): per segment validity; None is all.
        ego_width_m (float): ego width [m].
        margin_m (float): tube margin [m].
        normal_mode (str): one of TUBE_NORMALS.
        sector_mode (str): one of TUBE_SECTORS.

    Returns:
        tuple: (2,3) left and right plane, (2,) guarded flags.
    """
    values = torch.tensor(segments, dtype=torch.float64)
    centres = values[None, :, :2]
    directions = torch.stack(
        [torch.cos(values[:, 2]), torch.sin(values[:, 2])], -1
    )[None]
    half_lengths = values[None, :, 3]
    if valid is None:
        valid = [True] * len(segments)
    planes, guarded = tube_planes(
        torch.tensor([[position]], dtype=torch.float64),
        torch.full((1, 1), heading_rad, dtype=torch.float64),
        centres,
        directions,
        half_lengths,
        torch.tensor(valid)[None],
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), ego_width_m, dtype=torch.float64),
        margin_m,
        normal_mode,
        sector_mode,
    )
    return planes[0, 0], guarded[0, 0]


def planes_at_origin(
        segments: list[tuple[float, float, float, float]],
        heading_rad: float = 0.0,
        valid: list[bool] | None = None,
        ego_width_m: float = EGO_WIDTH_M,
        margin_m: float = MARGIN_M,
    ) -> torch.Tensor:
    """
    Heading-normal tube planes of one node at the origin (tube_rows).

    Args:
        segments (list): see tube_rows.
        heading_rad (float): reference heading.
        valid (list[bool] | None): per segment validity; None is all.
        ego_width_m (float): ego width [m].
        margin_m (float): tube margin [m].

    Returns:
        torch.Tensor: (2,3) left and right plane.
    """
    planes, guarded = tube_rows(
        segments, heading_rad=heading_rad, valid=valid,
        ego_width_m=ego_width_m, margin_m=margin_m,
    )
    assert not bool(guarded.any())
    return planes


def assert_row(
        plane: torch.Tensor,
        expected: list[float],
    ) -> None:
    """
    Args:
        plane (torch.Tensor): (3,) plane (a_x, a_y, b).
        expected (list[float]): expected (a_x, a_y, b).
    """
    torch.testing.assert_close(
        plane, torch.tensor(expected, dtype=plane.dtype),
        atol=1e-9, rtol=0.0,
    )


def test_reference_integrates_target_speed_along_target_heading():
    horizon_steps, row_count = 5, 2
    target = torch.zeros(horizon_steps, row_count, 7)
    target[..., SPEED] = torch.tensor([10.0, 4.0])
    target[..., HEADING] = torch.linspace(0.0, 0.4, horizon_steps)[:, None]
    target.requires_grad_(True)
    positions, headings = reference_positions(target)
    assert not positions.requires_grad and not headings.requires_grad
    torch.testing.assert_close(positions[0], torch.zeros(row_count, 2))
    expected = torch.zeros(row_count, 2)
    for step_idx in range(1, horizon_steps):
        heading = target[step_idx - 1, :, HEADING].detach()
        speed = target[step_idx - 1, :, SPEED].detach()
        expected = expected + DT_SECONDS * speed[:, None] * torch.stack(
            [torch.cos(heading), torch.sin(heading)], -1
        )
        torch.testing.assert_close(positions[step_idx], expected)
    torch.testing.assert_close(headings, target[..., HEADING].detach())


def test_projection_inside_a_slanted_segment():
    # Origin onto the 45 degree line through (0, 4): foot (-2, 2), 135
    # degrees to the left
    planes = planes_at_origin([(0.0, 4.0, 0.25 * math.pi, 5.0)])
    assert_row(planes[0], [0.0, 1.0, 2.0 - BODY_OFFSET_M])
    assert_row(planes[1], DUMMY_ROW)


def test_projection_clipped_to_the_segment_end():
    # Same line, half-length 1: the foot is the end at 1 m from the centre
    end = (-math.sqrt(0.5), 4.0 - math.sqrt(0.5))
    planes = planes_at_origin([(0.0, 4.0, 0.25 * math.pi, 1.0)])
    assert_row(planes[0], [0.0, 1.0, end[1] - BODY_OFFSET_M])


def test_each_side_takes_its_nearest_edge_point():
    planes = planes_at_origin([
        (0.0, 5.0, 0.0, 0.0),
        (0.0, 3.0, 0.0, 0.0),
        (0.0, -2.0, 0.0, 0.0),
        (0.0, -4.0, 0.0, 0.0),
    ])
    assert_row(planes[0], [0.0, 1.0, 3.0 - BODY_OFFSET_M])
    # -p_y <= 2 - (w/2 + mu)
    assert_row(planes[1], [0.0, -1.0, 2.0 - BODY_OFFSET_M])


@pytest.mark.parametrize("ahead_or_behind_x_m", [5.0, -5.0])
def test_points_outside_the_sector_are_skipped(ahead_or_behind_x_m):
    # (5, 2) lies 22 degrees left of the heading, (-5, 2) 158 degrees:
    # both nearer than (0, 6) but neither counts as the left edge
    planes = planes_at_origin([
        (ahead_or_behind_x_m, 2.0, 0.0, 0.0),
        (0.0, 6.0, 0.0, 0.0),
    ])
    assert_row(planes[0], [0.0, 1.0, 6.0 - BODY_OFFSET_M])
    alone = planes_at_origin([(ahead_or_behind_x_m, 2.0, 0.0, 0.0)])
    assert_row(alone[0], DUMMY_ROW)


def test_sector_follows_the_reference_heading():
    # Heading +90 degrees: (-3, 0) is on the left, the normal is (-1, 0)
    planes = planes_at_origin(
        [(-3.0, 0.0, 0.5 * math.pi, 0.0)], heading_rad=0.5 * math.pi
    )
    assert_row(planes[0], [-1.0, 0.0, 3.0 - BODY_OFFSET_M])
    assert_row(planes[1], DUMMY_ROW)


@pytest.mark.parametrize("ego_width_m, margin_m", [(2.0, 0.2), (1.6, 0.5)])
def test_offset_keeps_half_the_width_and_the_margin(ego_width_m, margin_m):
    planes = planes_at_origin(
        [(1.0, 2.5, 0.0, 4.0), (1.0, -3.5, 0.0, 4.0)],
        ego_width_m=ego_width_m, margin_m=margin_m,
    )
    body_offset_m = 0.5 * ego_width_m + margin_m
    assert_row(planes[0], [0.0, 1.0, 2.5 - body_offset_m])
    assert_row(planes[1], [0.0, -1.0, 3.5 - body_offset_m])


def test_no_observed_edge_gives_dummies():
    planes = planes_at_origin(
        [(0.0, 3.0, 0.0, 2.0), (0.0, -3.0, 0.0, 2.0)],
        valid=[False, False],
    )
    assert_row(planes[0], DUMMY_ROW)
    assert_row(planes[1], DUMMY_ROW)


def test_edge_segments_are_read_from_the_boundary_block():
    backbone = types.SimpleNamespace(
        road_features_count=7,
        obs_slots_partners_n=2,
        partner_features_count=9,
        obs_slots_lane_kept=1,
        obs_slots_boundary_kept=3,
    )
    partner_start_idx = 4
    edge_start = partner_start_idx + 2 * 9 + 7
    obs = torch.zeros(2, edge_start + 3 * 7 + 5)
    # Row 0, slot 1: centre (12, -6) m, half-length 2 m, direction (0, 1)
    obs[0, edge_start + 7:edge_start + 14] = torch.tensor([
        12.0 / OBS_NORM_XY_OFFSET_M, -6.0 / OBS_NORM_XY_OFFSET_M, 0.0,
        2.0 / OBS_NORM_ROAD_SEG_LENGTH_M, 0.7, 0.0, 1.0,
    ])
    centres, directions, half_lengths, valid = road_edge_segments(
        obs, partner_start_idx, backbone
    )
    assert centres.shape == (2, 3, 2)
    torch.testing.assert_close(centres[0, 1], torch.tensor([12.0, -6.0]))
    torch.testing.assert_close(directions[0, 1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(half_lengths[0, 1], torch.tensor(2.0))
    assert valid.tolist() == [[False, True, False], [False, False, False]]


def test_tube_layout_counts_planes_then_ellipses():
    validate_constraints(CONSTRAINT_TUBE, 0)
    assert constraint_slack_count(CONSTRAINT_TUBE, 0) == 2
    assert constraint_slack_count(CONSTRAINT_TUBE, 3) == 5
    assert constraint_value_count(CONSTRAINT_TUBE, 3) == 6 + 3 * 6
    dummies = dummy_constraints(
        CONSTRAINT_TUBE, 4, 3, 1, torch.float32, torch.device("cpu")
    )
    assert dummies.shape == (4, 3, TUBE_PLANE_VALUE_COUNT + 6)
    assert constraint_count_from_params(CONSTRAINT_TUBE, dummies) == 1
    torch.testing.assert_close(
        dummies[..., :TUBE_PLANE_VALUE_COUNT],
        torch.tensor(DUMMY_ROW * 2).expand(4, 3, -1),
    )
    with pytest.raises(ValueError):
        validate_constraints(CONSTRAINT_TUBE, -1)


def test_tube_config_rejects_bad_margins():
    # Ellipses are allowed (constraint_count = K); still need WarpMPC
    with pytest.raises(ValueError, match="need solver=warpmpc"):
        MPCStudentConfig(constraint_type=CONSTRAINT_TUBE, constraint_count=4)
    for margin_m in (-0.1, math.nan):
        with pytest.raises(ValueError, match="tube_margin_m"):
            MPCStudentConfig(
                constraint_type=CONSTRAINT_TUBE, tube_margin_m=margin_m
            )
    with pytest.raises(ValueError, match="need solver=warpmpc"):
        MPCStudentConfig(constraint_type=CONSTRAINT_TUBE)


def straight_plan(
        horizon_steps: int,
        speed_mps: float,
    ) -> torch.Tensor:
    """
    A plan along +x at constant speed in its own frame.

    Args:
        horizon_steps (int): T.
        speed_mps (float): speed [m/s].

    Returns:
        torch.Tensor: (T,1,5) states, node k at k * DT * speed.
    """
    states = torch.zeros(horizon_steps, 1, 5, dtype=torch.float64)
    states[:, 0, PX] = DT_SECONDS * speed_mps * torch.arange(horizon_steps)
    states[:, 0, SPEED] = speed_mps
    return states


def test_shifted_reference_moves_the_plan_into_the_new_frame():
    # Plan made at world (5, -2) heading 0.3; the car is now at world
    # (6, -1.5) heading 0.4. Node k of the reference is plan node k + 1
    horizon_steps, speed_mps = 5, 10.0
    plan = straight_plan(horizon_steps, speed_mps)
    plan_pose = torch.tensor([[5.0, -2.0, 0.3]], dtype=torch.float64)
    pose = torch.tensor([[6.0, -1.5, 0.4]], dtype=torch.float64)
    positions, headings = shifted_reference(plan, plan_pose, pose)
    for node_idx in range(horizon_steps):
        along_m = DT_SECONDS * speed_mps * (node_idx + 1)
        world = plan_pose[0, :2] + along_m * torch.tensor(
            [math.cos(0.3), math.sin(0.3)], dtype=torch.float64
        )
        offset = world - pose[0, :2]
        expected = torch.stack([
            math.cos(0.4) * offset[0] + math.sin(0.4) * offset[1],
            -math.sin(0.4) * offset[0] + math.cos(0.4) * offset[1],
        ])
        torch.testing.assert_close(positions[node_idx, 0], expected)
    torch.testing.assert_close(
        headings, torch.full((horizon_steps, 1), -0.1, dtype=torch.float64)
    )


def test_shifted_reference_without_motion_is_the_plan_one_node_on():
    horizon_steps = 4
    plan = straight_plan(horizon_steps, 5.0)
    plan[:, 0, HEADING] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    pose = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64)
    positions, headings = shifted_reference(plan, pose, pose)
    torch.testing.assert_close(positions[:-1], plan[1:, :, :2])
    # The last node: one step on from the old last node along its heading
    last = plan[-1, 0]
    expected_last = last[:2] + DT_SECONDS * 5.0 * torch.stack(
        [torch.cos(last[HEADING]), torch.sin(last[HEADING])]
    )
    torch.testing.assert_close(positions[-1, 0], expected_last)
    torch.testing.assert_close(
        headings[:, 0], torch.tensor([0.1, 0.2, 0.3, 0.3], dtype=torch.float64)
    )


def test_straight_reference_holds_speed_and_heading():
    positions, headings = straight_reference(torch.tensor([10.0, -2.0]), 4)
    torch.testing.assert_close(
        positions[..., 0],
        DT_SECONDS * torch.arange(4.0)[:, None] * torch.tensor([10.0, -2.0]),
    )
    assert bool((positions[..., 1] == 0).all())
    assert bool((headings == 0).all())


def test_planned_states_start_at_x0_and_follow_the_bicycle():
    x0 = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.05]])
    wheelbase = torch.tensor([2.7])
    controls = torch.tensor([[[1.0, 0.1]], [[0.5, -0.1]], [[0.0, 0.0]]])
    states = planned_states(x0, wheelbase, controls)
    torch.testing.assert_close(states[0], x0)
    dynamics = Bicycle(wheelbase)
    torch.testing.assert_close(states[1], dynamics(x0, controls[0]))
    torch.testing.assert_close(
        states[2], dynamics(states[1], controls[1])
    )


def test_poses_come_from_the_global_state():
    global_state = {
        "x": np.array([1.0, 2.0], dtype=np.float32),
        "y": np.array([3.0, 4.0], dtype=np.float32),
        "heading": np.array([0.5, -0.5], dtype=np.float32),
        "z": np.zeros(2, dtype=np.float32),
    }
    poses = poses_from_global_state(global_state, torch.device("cpu"))
    torch.testing.assert_close(
        poses, torch.tensor([[1.0, 3.0, 0.5], [2.0, 4.0, -0.5]])
    )


def test_shifted_reference_unwraps_the_heading_change():
    # The sim wraps headings at pi: a 0.1 rad left turn across the seam
    plan = straight_plan(3, 0.0)
    plan_pose = torch.tensor([[0.0, 0.0, math.pi - 0.05]], dtype=torch.float64)
    pose = torch.tensor([[0.0, 0.0, -math.pi + 0.05]], dtype=torch.float64)
    _, headings = shifted_reference(plan, plan_pose, pose)
    torch.testing.assert_close(
        headings, torch.full((3, 1), -0.1, dtype=torch.float64)
    )



def test_projection_normal_points_from_the_edge_into_the_road():
    # Edge straight left: n = (0, -1), the heading-normal row again
    planes, guarded = tube_rows(
        [(0.0, 3.0, 0.0, 5.0)], normal_mode=TUBE_NORMAL_PROJECTION
    )
    assert_row(planes[0], [0.0, 1.0, 3.0 - BODY_OFFSET_M])
    assert not bool(guarded.any())
    # 45 degree edge through (0, 4): foot (-2, 2), n = (1, -1) / sqrt 2
    planes, _ = tube_rows(
        [(0.0, 4.0, 0.25 * math.pi, 5.0)],
        normal_mode=TUBE_NORMAL_PROJECTION,
    )
    root_half = math.sqrt(0.5)
    offset_m = root_half * 0.5 * (EGO_LENGTH_M + EGO_WIDTH_M) + MARGIN_M
    assert_row(
        planes[0], [-root_half, root_half, 2.0 * math.sqrt(2.0) - offset_m]
    )


def test_projection_plane_is_the_same_along_a_straight_edge():
    # A heading off the edge tilts heading-normal planes, not these
    rows = [
        tube_rows(
            [(0.0, 3.0, 0.0, 20.0)], position=(along_m, 0.0),
            heading_rad=0.2, normal_mode=TUBE_NORMAL_PROJECTION,
        )[0][0]
        for along_m in (-4.0, 0.0, 6.0)
    ]
    for row in rows[1:]:
        assert_row(row, rows[0].tolist())
    torch.testing.assert_close(
        rows[0][:2], torch.tensor([0.0, 1.0], dtype=torch.float64)
    )
    heading_row = tube_rows(
        [(0.0, 3.0, 0.0, 20.0)], heading_rad=0.2
    )[0][0]
    assert abs(float(heading_row[0])) > 0.1


def test_guard_drops_a_reference_past_the_edge():
    # p_hat = (5, 4) is past the edge p_y = 3 that the car (origin) is
    # inside of: the edge point lies on p_hat's right, n points off-road
    planes, guarded = tube_rows(
        [(10.0, 3.0, 0.0, 10.0)], position=(5.0, 4.0),
        normal_mode=TUBE_NORMAL_PROJECTION,
    )
    assert guarded.tolist() == [False, True]
    assert_row(planes[1], DUMMY_ROW)
    # The heading normal has no guard
    _, guarded = tube_rows([(10.0, 3.0, 0.0, 10.0)], position=(5.0, 4.0))
    assert not bool(guarded.any())


def test_guard_drops_a_reference_on_the_edge():
    planes, guarded = tube_rows(
        [(0.0, 0.0, 0.25 * math.pi, 0.0)], position=(0.0, 0.0),
        sector_mode=TUBE_SECTOR_FULL, normal_mode=TUBE_NORMAL_PROJECTION,
    )
    # The point at p_hat lies on the heading line: no side takes it
    assert not bool(guarded.any())
    planes, guarded = tube_rows(
        [(1.0, 1.0, 0.0, 0.0)], position=(1.0, 1.0 - 1e-4),
        sector_mode=TUBE_SECTOR_FULL, normal_mode=TUBE_NORMAL_PROJECTION,
    )
    assert guarded.tolist() == [True, False]
    assert_row(planes[0], DUMMY_ROW)


def test_full_sector_takes_points_ahead_narrow_does_not():
    segments = [(5.0, 2.0, 0.0, 0.0), (5.0, 0.0, 0.0, 0.0)]
    narrow, _ = tube_rows(segments, normal_mode=TUBE_NORMAL_PROJECTION)
    assert_row(narrow[0], DUMMY_ROW)
    full, _ = tube_rows(
        segments, normal_mode=TUBE_NORMAL_PROJECTION,
        sector_mode=TUBE_SECTOR_FULL,
    )
    # (5, 2) on the left; (5, 0) on the heading line counts for no side
    direction = torch.tensor([-5.0, -2.0], dtype=torch.float64)
    direction = direction / direction.norm()
    torch.testing.assert_close(full[0, :2], -direction)
    assert_row(full[1], DUMMY_ROW)


@pytest.mark.parametrize("nearest, expected_offset_m", [
    # Side: half the width; ahead: half the length; diagonal: both
    ((0.0, 3.0), 0.5 * EGO_WIDTH_M + MARGIN_M),
    ((3.0, 0.0), 0.5 * EGO_LENGTH_M + MARGIN_M),
    ((2.0, 2.0), math.sqrt(0.5) * 0.5 * (EGO_LENGTH_M + EGO_WIDTH_M)
     + MARGIN_M),
])
def test_extent_offset_along_the_normal(nearest, expected_offset_m):
    nearest_point = torch.tensor([[nearest]], dtype=torch.float64)
    planes, guarded = projection_planes(
        torch.zeros(1, 1, 2, dtype=torch.float64),
        nearest_point,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[[0.0, 1.0]]], dtype=torch.float64),
        torch.tensor([EGO_LENGTH_M], dtype=torch.float64),
        torch.tensor([EGO_WIDTH_M], dtype=torch.float64),
        MARGIN_M,
    )
    assert not bool(guarded.any())
    # a . q* - b is the offset: the plane sits that far inside q*
    offset_m = (planes[0, 0, :2] * nearest_point[0, 0]).sum() - planes[0, 0, 2]
    torch.testing.assert_close(
        offset_m, torch.tensor(expected_offset_m, dtype=torch.float64)
    )


ROUTE_POINT_COUNT = 48
ROUTE_BACK_M = 6.0
TURN_RADIUS_M = 10.0


def straight_window(
        point_count: int = ROUTE_POINT_COUNT,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tuple: (1,R,2) route window along +x from -ROUTE_BACK_M, 1 m
            apart, and (1,R) validity, all valid.
    """
    s_m = torch.arange(point_count, dtype=torch.float64) - ROUTE_BACK_M
    window = torch.stack([s_m, torch.zeros_like(s_m)], -1)[None]
    return window, torch.ones(1, point_count, dtype=torch.bool)


def left_turn_window() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tuple: (1,R,2) route window on a left arc of radius TURN_RADIUS_M
            around (0, TURN_RADIUS_M), starting ROUTE_BACK_M of arc
            before the origin, 1 m of arc apart, and (1,R) validity.
    """
    s_m = (
        torch.arange(ROUTE_POINT_COUNT, dtype=torch.float64) - ROUTE_BACK_M
    )
    angle = s_m / TURN_RADIUS_M
    window = torch.stack([
        TURN_RADIUS_M * torch.sin(angle),
        TURN_RADIUS_M * (1.0 - torch.cos(angle)),
    ], -1)[None]
    return window, torch.ones(1, ROUTE_POINT_COUNT, dtype=torch.bool)


def anchor_nodes(
        positions: list[tuple[float, float]],
        window: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        positions (list): reference position per node [m], one agent.
        window (tuple): route window and validity of the agent.

    Returns:
        tuple: anchors (T,2), tangents (T,2) and valid (T,).
    """
    reference = torch.tensor(positions, dtype=torch.float64)[:, None]
    anchors, tangents, valid = route_anchors(reference, *window)
    return anchors[:, 0], tangents[:, 0], valid[:, 0]


def route_rows(
        segments: list[tuple[float, float, float, float]],
        heading_rad: float = 0.0,
        anchor_valid: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Route planes of one node anchored at the origin with tangent +x.

    Args:
        segments (list): see tube_rows.
        heading_rad (float): reference heading of the car.
        anchor_valid (bool): whether the node has a route anchor.

    Returns:
        tuple: (2,3) left and right plane, (2,) guarded flags.
    """
    values = torch.tensor(segments, dtype=torch.float64)
    directions = torch.stack(
        [torch.cos(values[:, 2]), torch.sin(values[:, 2])], -1
    )[None]
    planes, guarded = route_planes(
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[anchor_valid]]),
        torch.full((1, 1), heading_rad, dtype=torch.float64),
        values[None, :, :2],
        directions,
        values[None, :, 3],
        torch.ones(1, len(segments), dtype=torch.bool),
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), EGO_WIDTH_M, dtype=torch.float64),
        MARGIN_M,
    )
    return planes[0, 0], guarded[0, 0]


def test_route_anchor_on_a_straight_route_keeps_the_arc_length():
    anchors, tangents, valid = anchor_nodes(
        [(0.0, 0.0), (2.5, 0.0), (7.25, 0.0)], straight_window()
    )
    torch.testing.assert_close(
        anchors,
        torch.tensor([[0.0, 0.0], [2.5, 0.0], [7.25, 0.0]],
                     dtype=torch.float64),
    )
    torch.testing.assert_close(
        tangents, torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64)
    )
    assert bool(valid.all())


def test_route_anchor_ignores_the_reference_lateral_offset():
    anchors, tangents, valid = anchor_nodes([(4.0, -3.0)], straight_window())
    torch.testing.assert_close(anchors, torch.tensor(
        [[4.0, 0.0]], dtype=torch.float64
    ))
    torch.testing.assert_close(tangents, torch.tensor(
        [[1.0, 0.0]], dtype=torch.float64
    ))
    assert bool(valid.all())


def test_route_anchor_never_moves_back_along_the_horizon():
    anchors, _, _ = anchor_nodes(
        [(5.0, 0.0), (3.0, 0.0), (6.0, 0.0)], straight_window()
    )
    torch.testing.assert_close(anchors[:, 0], torch.tensor(
        [5.0, 5.0, 6.0], dtype=torch.float64
    ))


def test_route_tangent_follows_the_turn():
    window = left_turn_window()
    arc_m = 12.0
    angle = arc_m / TURN_RADIUS_M
    on_arc = (
        TURN_RADIUS_M * math.sin(angle),
        TURN_RADIUS_M * (1.0 - math.cos(angle)),
    )
    anchors, tangents, valid = anchor_nodes([on_arc], window)
    torch.testing.assert_close(
        anchors[0], torch.tensor(on_arc, dtype=torch.float64),
        atol=0.01, rtol=0.0,
    )
    # A chord over +-5 m of a 10 m arc is parallel to the true tangent
    expected = torch.tensor(
        [math.cos(angle), math.sin(angle)], dtype=torch.float64
    )
    torch.testing.assert_close(tangents[0], expected, atol=0.01, rtol=0.0)
    assert bool(valid.all())


def test_route_tangent_chord_stays_on_the_valid_points():
    window, window_valid = straight_window()
    window_valid[0, 10:] = False
    _, tangents, valid = anchor_nodes(
        [(3.5, 0.0)], (window, window_valid)
    )
    torch.testing.assert_close(tangents[0], torch.tensor(
        [1.0, 0.0], dtype=torch.float64
    ))
    assert bool(valid.all())


def test_route_anchor_far_from_the_route_is_invalid():
    far_m = ROUTE_ANCHOR_MAX_DISTANCE_M + 1.0
    _, _, valid = anchor_nodes([(5.0, far_m)], straight_window())
    assert not bool(valid.any())
    window, _ = straight_window()
    no_route = torch.zeros(1, ROUTE_POINT_COUNT, dtype=torch.bool)
    _, _, valid = anchor_nodes([(5.0, 0.0)], (window, no_route))
    assert not bool(valid.any())


def test_route_plane_takes_the_nearest_parallel_edge():
    planes, guarded = route_rows([
        (0.5, 3.0, 0.0, 5.0),
        (0.0, 5.0, 0.0, 5.0),
        (0.0, -4.0, math.radians(20.0), 5.0),
    ])
    assert not bool(guarded.any())
    assert_row(planes[0], [0.0, 1.0, 3.0 - BODY_OFFSET_M])
    # A 20 degree edge through (0, -4): its point nearest the anchor lies
    # 4 cos(20) from it, 1.3 m along the route, 4 cos^2(20) to the side
    lateral_m = 4.0 * math.cos(math.radians(20.0)) ** 2
    assert_row(planes[1], [0.0, -1.0, lateral_m - BODY_OFFSET_M])


def test_route_plane_skips_edges_across_the_route():
    island_tip = (0.0, 1.5, math.radians(90.0), 0.5)
    planes, _ = route_rows([island_tip, (0.0, 3.0, 0.0, 5.0)])
    assert_row(planes[0], [0.0, 1.0, 3.0 - BODY_OFFSET_M])


def test_route_plane_skips_edges_beyond_the_along_window():
    ahead_of_window = (6.0, 1.5, 0.0, 1.0)
    planes, _ = route_rows([ahead_of_window])
    assert_row(
        planes[0], [0.0, 1.0, ROUTE_PLANE_MAX_OFFSET_M - BODY_OFFSET_M]
    )


def test_route_plane_body_offset_uses_the_car_heading():
    heading_rad = math.radians(30.0)
    planes, _ = route_rows([(0.0, 4.0, 0.0, 5.0)], heading_rad)
    offset_m = (
        0.5 * EGO_LENGTH_M * math.sin(heading_rad)
        + 0.5 * EGO_WIDTH_M * math.cos(heading_rad) + MARGIN_M
    )
    assert_row(planes[0], [0.0, 1.0, 4.0 - offset_m])


def test_route_plane_guard_and_missing_anchor_give_dummies():
    # The road at the anchor is narrower than the body: no plane
    planes, guarded = route_rows([(0.0, 0.5, 0.0, 5.0)])
    assert bool(guarded[0]) and not bool(guarded[1])
    assert_row(planes[0], DUMMY_ROW)
    planes, guarded = route_rows([(0.0, 3.0, 0.0, 5.0)], anchor_valid=False)
    assert not bool(guarded.any())
    assert_row(planes[0], DUMMY_ROW)
    assert_row(planes[1], DUMMY_ROW)


def test_route_window_spacing_matches_the_sim():
    assert ROUTE_WINDOW_SPACING_M == 1.0


def test_tube_anchor_must_be_known():
    with pytest.raises(ValueError, match="tube_anchor"):
        MPCStudentConfig(constraint_type=CONSTRAINT_TUBE, tube_anchor="x")


def test_route_plane_may_exclude_the_car_but_not_its_anchor():
    # A plane anchored 2 m to the right of the car: the car is outside it,
    # its anchor 3 m inside the edge is not, so the plane is kept
    values = torch.tensor([(0.0, 1.0, 0.0, 5.0)], dtype=torch.float64)
    planes, guarded = route_planes(
        torch.tensor([[[0.0, -2.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.zeros(1, 1, dtype=torch.float64),
        values[None, :, :2],
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        values[None, :, 3],
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), EGO_WIDTH_M, dtype=torch.float64),
        MARGIN_M,
    )
    assert not bool(guarded.any())
    assert_row(planes[0, 0, 0], [0.0, 1.0, 1.0 - BODY_OFFSET_M])


def test_island_filter_drops_an_edge_the_reference_has_passed():
    """An island edge between the route and the reference position no
    longer sets the left plane when reference_positions is given; the
    curb beyond the reference does, and the reference satisfies it."""
    island_y_m, curb_y_m, reference_y_m = 1.5, 5.0, 3.0
    values = torch.tensor(
        [[0.0, island_y_m, 0.0, 2.0], [0.0, curb_y_m, 0.0, 5.0]],
        dtype=torch.float64,
    )
    directions = torch.stack(
        [torch.cos(values[:, 2]), torch.sin(values[:, 2])], -1
    )[None]
    common = (
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.zeros(1, 1, dtype=torch.float64),
        values[None, :, :2],
        directions,
        values[None, :, 3],
        torch.ones(1, 2, dtype=torch.bool),
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), EGO_WIDTH_M, dtype=torch.float64),
        MARGIN_M,
    )
    reference = torch.tensor([[[0.0, reference_y_m]]], dtype=torch.float64)
    unfiltered, _ = route_planes(*common)
    filtered, _ = route_planes(*common, reference)
    left_unfiltered, left_filtered = unfiltered[0, 0, 0], filtered[0, 0, 0]
    # Left plane (0, 1) . p <= b
    assert float(left_unfiltered[2]) < reference_y_m
    assert float(left_filtered[2]) > reference_y_m
    assert float(left_filtered[2]) < curb_y_m
    # The right side has no edge either way
    torch.testing.assert_close(unfiltered[0, 0, 1], filtered[0, 0, 1])


def test_parallel_limit_and_far_offset_are_configurable():
    """With the parallel limit at 90 degrees a crossing edge counts; a
    side without an edge gets its plane max_offset_m from the anchor."""
    crossing_y_m = 3.0
    values = torch.tensor([[0.0, crossing_y_m, math.pi / 2, 1.0]],
                          dtype=torch.float64)
    directions = torch.stack(
        [torch.cos(values[:, 2]), torch.sin(values[:, 2])], -1
    )[None]
    common = (
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.zeros(1, 1, dtype=torch.float64),
        values[None, :, :2],
        directions,
        values[None, :, 3],
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), EGO_WIDTH_M, dtype=torch.float64),
        MARGIN_M,
    )
    limited, _ = route_planes(*common)
    free, _ = route_planes(*common, None, math.pi / 2, 15.0)
    # Left: the crossing edge (its nearest point 2 m up) only without the limit
    assert float(limited[0, 0, 0, 2]) == pytest.approx(
        ROUTE_PLANE_MAX_OFFSET_M - BODY_OFFSET_M
    )
    assert float(free[0, 0, 0, 2]) == pytest.approx(2.0 - BODY_OFFSET_M)
    # Right: no edge, so the far plane at 15 m
    assert float(free[0, 0, 1, 2]) == pytest.approx(15.0 - BODY_OFFSET_M)


def test_along_window_can_be_removed():
    """A parallel curb 6 m ahead of the anchor counts only without the
    +-2 m along-route window."""
    curb_y_m = 3.0
    values = torch.tensor([[6.0, curb_y_m, 0.0, 1.0]], dtype=torch.float64)
    directions = torch.stack(
        [torch.cos(values[:, 2]), torch.sin(values[:, 2])], -1
    )[None]
    common = (
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.zeros(1, 1, dtype=torch.float64),
        values[None, :, :2],
        directions,
        values[None, :, 3],
        torch.ones(1, 1, dtype=torch.bool),
        torch.full((1,), EGO_LENGTH_M, dtype=torch.float64),
        torch.full((1,), EGO_WIDTH_M, dtype=torch.float64),
        MARGIN_M,
    )
    windowed, _ = route_planes(*common)
    unlimited, _ = route_planes(
        *common, None, ROUTE_EDGE_PARALLEL_MAX_RAD, ROUTE_PLANE_MAX_OFFSET_M,
        math.inf,
    )
    assert float(windowed[0, 0, 0, 2]) == pytest.approx(
        ROUTE_PLANE_MAX_OFFSET_M - BODY_OFFSET_M
    )
    assert float(unlimited[0, 0, 0, 2]) == pytest.approx(
        curb_y_m - BODY_OFFSET_M
    )
