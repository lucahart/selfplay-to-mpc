"""Fixed road bounds of the WarpMPC student: the tube.

Per plan node k >= 1 and side (left, right), a half-plane on the ego-frame
position through the nearest observed road-edge point on that side, its
normal pointing from that point to the reference position (projection)
or along the reference heading's normal (heading), held the ego box plus
a margin inside the edge. The reference
positions, which only place the planes, are either the agent's previous
plan shifted by one step into the current ego frame (option 2, the
default) or the rollout of the head's speed and heading targets (option
1). Nothing here is learned or differentiated.

Everything is in the ego frame of the current step. Plane rows are
(a_x, a_y, b) for a . p - s <= b, as warpmpc_solver.plane_terms reads them.

Torch only, so the student can import it without the WarpMPC venv.
"""

import math

import torch

from pufferlib.distill.bicycle import (
    DT_SECONDS,
    HEADING,
    PX,
    PY,
    SPEED,
    Bicycle,
)
from pufferlib.distill.constraints import PLANE_DUMMY_OFFSET_M
from pufferlib.distill.obstacles import OBS_NORM_XY_OFFSET_M

# Road segment layout (drive.h write_road_obs): midpoint x, y, z, half
# length, width, direction x, y
ROAD_X_IDX, ROAD_Y_IDX, ROAD_HALF_LENGTH_IDX = 0, 1, 3
ROAD_DIR_X_IDX, ROAD_DIR_Y_IDX = 5, 6
# Normalizer of the road segment half length (puffer_drive.yaml)
OBS_NORM_ROAD_SEG_LENGTH_M = 10.0
# Edge points count for side h when their angle from the reference
# heading towards h lies in this sector: points nearly ahead or behind
# belong to the road's continuation, not to its side
SECTOR_MIN_RAD = math.radians(30.0)
SECTOR_MAX_RAD = math.radians(150.0)
# Sector of a side: narrow keeps SECTOR_MIN_RAD..SECTOR_MAX_RAD, full the
# whole half-plane (the heading line excluded)
TUBE_SECTOR_NARROW = "narrow"
TUBE_SECTOR_FULL = "full"
TUBE_SECTORS = (TUBE_SECTOR_NARROW, TUBE_SECTOR_FULL)
# Plane normal: from the edge point to the reference position, or the
# left normal of the reference heading
TUBE_NORMAL_PROJECTION = "projection"
TUBE_NORMAL_HEADING = "heading"
TUBE_NORMALS = (TUBE_NORMAL_PROJECTION, TUBE_NORMAL_HEADING)
# A reference position this close to its edge point has no normal
EDGE_CONTACT_M = 1e-3
# sigma_h in plane order: left plane, right plane
SIDE_SIGNS = (1.0, -1.0)
# Safety margin mu between the body and the edge
DEFAULT_TUBE_MARGIN_M = 0.2
# Reference positions: the previous plan shifted (option 2), or the
# rollout of the head's targets (option 1)
TUBE_REFERENCE_PREVIOUS_PLAN = "previous_plan"
TUBE_REFERENCE_ROLLOUT = "rollout"
TUBE_REFERENCES = (TUBE_REFERENCE_PREVIOUS_PLAN, TUBE_REFERENCE_ROLLOUT)
# World pose of an agent: (x, y, heading)
POSE_X_IDX, POSE_Y_IDX, POSE_HEADING_IDX = 0, 1, 2
POSE_DIM = 3
# Plane anchors: the reference positions (plan) or the agent's route
# (route_anchors, route_planes)
TUBE_ANCHOR_PLAN = "plan"
TUBE_ANCHOR_ROUTE = "route"
TUBE_ANCHORS = (TUBE_ANCHOR_PLAN, TUBE_ANCHOR_ROUTE)
# Arc length between route window points (drive.h ROUTE_WINDOW_SPACING_M)
ROUTE_WINDOW_SPACING_M = 1.0
# h of the tangent chord p(s + h) - p(s - h): half the longest map lane
# segment (10 m), so the tangent turns smoothly across a polyline vertex
ROUTE_TANGENT_HALF_WINDOW_M = 5.0
# A reference position farther than this from the route window (the agent
# left its route) gets no route anchor
ROUTE_ANCHOR_MAX_DISTANCE_M = 10.0
# Edge segments count for a route plane when they run within this angle of
# the route tangent: island tips, median noses and the far edges of
# crossing roads point across the route
ROUTE_EDGE_PARALLEL_MAX_RAD = math.radians(35.0)
# Edge points count when they lie within this distance along the route
# from the anchor; the nearest one over this stretch sets the offset
ROUTE_EDGE_ALONG_WINDOW_M = 2.0
# Without a parallel edge closer than this, the plane sits here: inactive
# in practice, never wrong
ROUTE_PLANE_MAX_OFFSET_M = 8.0


def road_edge_segments(
        obs: torch.Tensor,
        partner_start_idx: int,
        backbone: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The observed road-edge segments (boundary block), in the ego frame.

    Args:
        obs (torch.Tensor): (B,obs_dim) observations.
        partner_start_idx (int): first index of the partner block: the
            ego block plus the reward and goal context (the backbone's
            ego_dim + context_dim).
        backbone (torch.nn.Module): source of the observation layout
            (DriveBackbone).

    Returns:
        tuple: centres (B,S,2) [m], unit directions (B,S,2), half-lengths
            (B,S) [m] and validity (B,S) bool.
    """
    feature_count = backbone.road_features_count
    lane_start = (
        partner_start_idx
        + backbone.obs_slots_partners_n * backbone.partner_features_count
    )
    edge_start = lane_start + backbone.obs_slots_lane_kept * feature_count
    edge_end = edge_start + backbone.obs_slots_boundary_kept * feature_count
    segments = obs[:, edge_start:edge_end].reshape(
        obs.shape[0], backbone.obs_slots_boundary_kept, feature_count
    )
    centres = segments[..., [ROAD_X_IDX, ROAD_Y_IDX]] * OBS_NORM_XY_OFFSET_M
    directions = segments[..., [ROAD_DIR_X_IDX, ROAD_DIR_Y_IDX]]
    half_lengths = (
        segments[..., ROAD_HALF_LENGTH_IDX] * OBS_NORM_ROAD_SEG_LENGTH_M
    )
    # Empty slots are all zeros
    valid = segments.abs().sum(-1) > 0
    return centres, directions, half_lengths, valid


def reference_positions(
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference positions and headings, option 1: the head's speed and
    heading targets integrated from the origin (its p_x target has weight
    0, so the targets alone are no trajectory). Detached.

    Args:
        target (torch.Tensor): (T,B,7) cost target on tau = (px, py,
            heading, speed, steering, accel, rate).

    Returns:
        tuple: positions (T,B,2) [m], p_0 = 0, and headings (T,B) [rad].
    """
    headings = target[..., HEADING].detach()
    speeds_mps = target[..., SPEED].detach()
    steps = DT_SECONDS * speeds_mps[..., None] * torch.stack(
        [torch.cos(headings), torch.sin(headings)], dim=-1
    )
    # p_{k+1} = p_k + step_k: node k sums the steps before it
    positions = torch.cumsum(steps, dim=0) - steps
    return positions, headings


def planned_states(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
    """
    States of the plan's nodes: the controls rolled out through the
    bicycle from x0, node 0 at x0.

    Args:
        x0 (torch.Tensor): (B,5) ego-frame start state.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        controls (torch.Tensor): (T,B,2) plan.

    Returns:
        torch.Tensor: (T,B,5) states (px, py, heading, speed, steering).
    """
    dynamics = Bicycle(wheelbase)
    state = x0
    states = []
    for step_controls in controls:
        states.append(state)
        state = dynamics(state, step_controls)
    return torch.stack(states)


def poses_from_global_state(
        global_state: dict,
        device: torch.device,
    ) -> torch.Tensor:
    """
    World poses from an env's get_global_agent_state().

    Args:
        global_state (dict): 'x', 'y', 'heading' arrays (B,).
        device (torch.device): device of the result.

    Returns:
        torch.Tensor: (B,3) (x, y, heading) [m, m, rad].
    """
    return torch.stack([
        torch.as_tensor(global_state[key], device=device)
        for key in ("x", "y", "heading")
    ], dim=1)


def straight_reference(
        speed_mps: torch.Tensor,
        horizon_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference without a previous plan (episode start): constant speed
    straight ahead from the current state.

    Args:
        speed_mps (torch.Tensor): (B,) current signed speed [m/s].
        horizon_steps (int): T.

    Returns:
        tuple: positions (T,B,2) [m], p_0 = 0, and headings (T,B), all 0.
    """
    times_s = DT_SECONDS * torch.arange(
        horizon_steps, dtype=speed_mps.dtype, device=speed_mps.device
    )
    positions = torch.zeros(
        horizon_steps, speed_mps.shape[0], 2,
        dtype=speed_mps.dtype, device=speed_mps.device,
    )
    positions[..., 0] = times_s[:, None] * speed_mps
    return positions, torch.zeros_like(positions[..., 0])


def shifted_reference(
        plan_states: torch.Tensor,
        plan_poses: torch.Tensor,
        poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference from the previous plan (option 2): nodes 1..T-1 of the plan
    become nodes 0..T-2, the last node is its predecessor moved one step
    at constant speed and heading, and all are carried from the frame the
    plan was made in into the current ego frame by the agent's real pose
    change (not the planned one: the executed control may differ).

    Args:
        plan_states (torch.Tensor): (T,B,5) previous plan in its own ego
            frame.
        plan_poses (torch.Tensor): (B,3) world pose (x, y, heading) the
            plan was made at.
        poses (torch.Tensor): (B,3) current world pose.

    Returns:
        tuple: positions (T,B,2) [m] and headings (T,B) [rad] in the
            current ego frame.
    """
    last = plan_states[-1]
    last_heading = last[:, HEADING]
    extrapolated = last[:, [PX, PY]] + DT_SECONDS * last[:, SPEED, None] * (
        torch.stack([torch.cos(last_heading), torch.sin(last_heading)], -1)
    )
    old_positions = torch.cat(
        [plan_states[1:, :, [PX, PY]], extrapolated[None]]
    )
    old_headings = torch.cat([plan_states[1:, :, HEADING], last_heading[None]])
    # p_current = R(-theta_1) (R(theta_0) p_old + X_0 - X_1)
    turn = plan_poses[:, POSE_HEADING_IDX] - poses[:, POSE_HEADING_IDX]
    # The sim wraps headings; keep the turn in (-pi, pi]
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))
    world_offset = (
        plan_poses[:, [POSE_X_IDX, POSE_Y_IDX]]
        - poses[:, [POSE_X_IDX, POSE_Y_IDX]]
    )
    cos_now = torch.cos(poses[:, POSE_HEADING_IDX])
    sin_now = torch.sin(poses[:, POSE_HEADING_IDX])
    offset_x = cos_now * world_offset[:, 0] + sin_now * world_offset[:, 1]
    offset_y = cos_now * world_offset[:, 1] - sin_now * world_offset[:, 0]
    cos_turn, sin_turn = torch.cos(turn), torch.sin(turn)
    positions = torch.stack([
        cos_turn * old_positions[..., 0] - sin_turn * old_positions[..., 1]
        + offset_x,
        sin_turn * old_positions[..., 0] + cos_turn * old_positions[..., 1]
        + offset_y,
    ], dim=-1)
    return positions, old_headings + turn


def projection_planes(
        positions: torch.Tensor,
        nearest: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
        ego_length_m: torch.Tensor,
        ego_width_m: torch.Tensor,
        margin_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Planes whose normal points from the edge point to the reference
    position, n = (p_hat - q*) / |p_hat - q*|, held the ego box's extent
    along n plus the margin inside the edge: n . p >= n . q* + offset,
    offset = (l_e / 2)|n . t| + (w_e / 2)|n . nu| + mu. Guarded: a
    reference position on the edge, or past it while the car (the origin)
    is not, gives no plane.

    Args:
        positions (torch.Tensor): (T,B,2) reference positions [m].
        nearest (torch.Tensor): (T,B,2) selected edge points q* [m].
        tangent (torch.Tensor): (T,B,2) reference directions t.
        normal (torch.Tensor): (T,B,2) their left normals nu.
        ego_length_m (torch.Tensor): (B,) ego length [m].
        ego_width_m (torch.Tensor): (B,) ego width [m].
        margin_m (float): mu [m].

    Returns:
        tuple: planes (T,B,3) (a_x, a_y, b) with a = -n, and guarded
            (T,B) bool, rows the guard dropped.
    """
    away = positions - nearest
    distance_m = away.norm(dim=-1, keepdim=True)
    on_edge = distance_m[..., 0] < EDGE_CONTACT_M
    into_road = away / distance_m.clamp_min(EDGE_CONTACT_M)
    # n . (p_0 - q*) with p_0 = 0: the car is on the road side of the plane
    car_inside = -(into_road * nearest).sum(-1) > 0.0
    offset_m = (
        0.5 * ego_length_m * (into_road * tangent).sum(-1).abs()
        + 0.5 * ego_width_m * (into_road * normal).sum(-1).abs()
        + margin_m
    )
    bound = (into_road * nearest).sum(-1) + offset_m
    planes = torch.cat([-into_road, -bound[..., None]], dim=-1)
    return planes, on_edge | ~car_inside


def tube_planes(
        positions: torch.Tensor,
        headings: torch.Tensor,
        centres: torch.Tensor,
        directions: torch.Tensor,
        half_lengths: torch.Tensor,
        valid: torch.Tensor,
        ego_length_m: torch.Tensor,
        ego_width_m: torch.Tensor,
        margin_m: float,
        normal_mode: str = TUBE_NORMAL_PROJECTION,
        sector_mode: str = TUBE_SECTOR_NARROW,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Left and right plane per node: project the reference position onto
    every edge segment, keep the projections on the side (in its sector),
    take the nearest, and put a plane through it. Heading normal: the
    plane runs parallel to the reference heading, w_e / 2 + margin inside
    the edge point. Projection normal: projection_planes. A side without
    an edge point, or dropped by the guard, gets the dummy row (never
    active).

    Args:
        positions (torch.Tensor): (T,B,2) reference positions [m].
        headings (torch.Tensor): (T,B) reference headings [rad].
        centres (torch.Tensor): (B,S,2) segment centres [m].
        directions (torch.Tensor): (B,S,2) unit segment directions.
        half_lengths (torch.Tensor): (B,S) segment half-lengths [m].
        valid (torch.Tensor): (B,S) bool, observed segments.
        ego_length_m (torch.Tensor): (B,) ego length [m].
        ego_width_m (torch.Tensor): (B,) ego width [m].
        margin_m (float): mu, distance the body keeps from the edge [m].
        normal_mode (str): one of TUBE_NORMALS.
        sector_mode (str): one of TUBE_SECTORS.

    Returns:
        tuple: planes (T,B,2,3) (a_x, a_y, b), left then right, and
            guarded (T,B,2) bool, rows the projection guard dropped.
    """
    tangent = torch.stack([torch.cos(headings), torch.sin(headings)], -1)
    normal = torch.stack([-torch.sin(headings), torch.cos(headings)], -1)
    # (T,B,S): every node against every segment of its row
    along = (
        (positions[:, :, None] - centres[None]) * directions[None]
    ).sum(-1)
    clipped = torch.maximum(
        torch.minimum(along, half_lengths[None]), -half_lengths[None]
    )
    projections = centres[None] + clipped[..., None] * directions[None]
    delta = projections - positions[:, :, None]
    forward_m = (delta * tangent[:, :, None]).sum(-1)
    lateral_m = (delta * normal[:, :, None]).sum(-1)
    distance_m = delta.norm(dim=-1)
    dummy = torch.tensor(
        [0.0, 1.0, PLANE_DUMMY_OFFSET_M],
        dtype=positions.dtype, device=positions.device,
    )
    body_offset_m = 0.5 * ego_width_m + margin_m
    planes, guards = [], []
    for side_sign in SIDE_SIGNS:
        # Full: the whole half-plane of the side, the heading line excluded
        in_sector = valid[None] & (side_sign * lateral_m > 0.0)
        if sector_mode == TUBE_SECTOR_NARROW:
            angle = torch.atan2(side_sign * lateral_m, forward_m)
            in_sector = (
                in_sector
                & (angle >= SECTOR_MIN_RAD) & (angle <= SECTOR_MAX_RAD)
            )
        nearest_distance_m, nearest_idx = torch.where(
            in_sector, distance_m, math.inf
        ).min(dim=-1)
        nearest = torch.gather(
            projections, 2,
            nearest_idx[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        has_edge = torch.isfinite(nearest_distance_m)
        guarded = torch.zeros_like(has_edge)
        side_normal = side_sign * normal
        offset_m = (side_normal * nearest).sum(-1) - body_offset_m
        plane = torch.cat([side_normal, offset_m[..., None]], dim=-1)
        if normal_mode == TUBE_NORMAL_PROJECTION:
            plane, dropped = projection_planes(
                positions, nearest, tangent, normal, ego_length_m,
                ego_width_m, margin_m,
            )
            guarded = has_edge & dropped
        keep = (has_edge & ~guarded)[..., None]
        planes.append(torch.where(keep, plane, dummy))
        guards.append(guarded)
    return torch.stack(planes, dim=2), torch.stack(guards, dim=2)


def window_points(
        window: torch.Tensor,
        position_idx: torch.Tensor,
    ) -> torch.Tensor:
    """
    Points of the route window at fractional indices, linearly
    interpolated (the window is uniform in arc length).

    Args:
        window (torch.Tensor): (B,R,2) route window [m].
        position_idx (torch.Tensor): (T,B) fractional indices in
            [0, R - 1].

    Returns:
        torch.Tensor: (T,B,2) points [m].
    """
    point_count = window.shape[1]
    lower_idx = position_idx.floor().long().clamp(0, point_count - 2)
    fraction = (position_idx - lower_idx)[..., None]
    expanded = window[None].expand(position_idx.shape[0], -1, -1, -1)
    lower = torch.gather(
        expanded, 2, lower_idx[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    upper = torch.gather(
        expanded, 2, (lower_idx + 1)[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    return lower + fraction * (upper - lower)


def route_anchors(
        positions: torch.Tensor,
        window: torch.Tensor,
        window_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Node k's anchor: its reference position projected onto the route
    window, keeping only the arc length s_k, never behind node k - 1's.
    The reference's lateral wobble then has no effect, only its progress
    along the road. Tangent: the chord p(s_k + h) - p(s_k - h), the
    length-weighted mean direction of the route over that stretch; its
    ends are kept on the valid part of the window.

    Args:
        positions (torch.Tensor): (T,B,2) reference positions [m].
        window (torch.Tensor): (B,R,2) route window in the ego frame [m]
            (Drive.get_route_window).
        window_valid (torch.Tensor): (B,R) bool; valid points are
            contiguous.

    Returns:
        tuple: anchors (T,B,2) [m], unit tangents (T,B,2), and valid (T,B)
            bool: a row with a route whose reference position lies within
            ROUTE_ANCHOR_MAX_DISTANCE_M of it.
    """
    starts, ends = window[:, :-1], window[:, 1:]
    segment_valid = window_valid[:, :-1] & window_valid[:, 1:]
    segments = ends - starts
    length_sq = (segments * segments).sum(-1).clamp_min(EDGE_CONTACT_M)
    # (T,B,R-1): every node against every segment of its route
    along = (
        ((positions[:, :, None] - starts[None]) * segments[None]).sum(-1)
        / length_sq[None]
    ).clamp(0.0, 1.0)
    projections = starts[None] + along[..., None] * segments[None]
    distance_m = torch.where(
        segment_valid[None],
        (projections - positions[:, :, None]).norm(dim=-1),
        math.inf,
    )
    nearest_distance_m, segment_idx = distance_m.min(dim=-1)
    position_idx = segment_idx + torch.gather(
        along, 2, segment_idx[..., None]
    ).squeeze(2)
    # Monotone along the horizon: a bend cannot pull a node back
    position_idx = torch.cummax(position_idx, dim=0).values
    point_count = window.shape[1]
    first_valid_idx = window_valid.float().argmax(dim=1)
    last_valid_idx = point_count - 1 - window_valid.flip(1).float().argmax(1)
    half_window_idx = ROUTE_TANGENT_HALF_WINDOW_M / ROUTE_WINDOW_SPACING_M
    low_idx = torch.maximum(
        position_idx - half_window_idx, first_valid_idx[None].float()
    )
    high_idx = torch.minimum(
        position_idx + half_window_idx, last_valid_idx[None].float()
    )
    chord = window_points(window, high_idx) - window_points(window, low_idx)
    tangents = chord / chord.norm(dim=-1, keepdim=True).clamp_min(
        EDGE_CONTACT_M
    )
    valid = nearest_distance_m <= ROUTE_ANCHOR_MAX_DISTANCE_M
    return window_points(window, position_idx), tangents, valid


def route_planes(
        anchors: torch.Tensor,
        tangents: torch.Tensor,
        anchor_valid: torch.Tensor,
        headings: torch.Tensor,
        centres: torch.Tensor,
        directions: torch.Tensor,
        half_lengths: torch.Tensor,
        valid: torch.Tensor,
        ego_length_m: torch.Tensor,
        ego_width_m: torch.Tensor,
        margin_m: float,
        reference_positions: torch.Tensor | None = None,
        parallel_max_rad: float = ROUTE_EDGE_PARALLEL_MAX_RAD,
        max_offset_m: float = ROUTE_PLANE_MAX_OFFSET_M,
        along_window_m: float = ROUTE_EDGE_ALONG_WINDOW_M,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Left and right plane per node, parallel to the route tangent:
    n_h = sigma_h nu (nu the tangent's left normal), d_h the lateral
    distance from the anchor to the nearest point of an edge segment on
    side h that runs within ROUTE_EDGE_PARALLEL_MAX_RAD of the tangent and
    lies within ROUTE_EDGE_ALONG_WINDOW_M of the anchor along it, capped
    at ROUTE_PLANE_MAX_OFFSET_M. Row n_h . p <= n_h . anchor + d_h -
    offset, offset = (l_e / 2)|n_h . t| + (w_e / 2)|n_h . nu_car| + mu with
    the reference heading t, as projection_planes. Guarded: a plane its
    own anchor violates (the road there is narrower than the body) gives
    the dummy row, as does a node without a route anchor. The car (the
    origin) may lie outside a far node's plane: in a bend, the tangent
    line ahead passes beside the car, and the plane binds node k only.

    Args:
        anchors (torch.Tensor): (T,B,2) route anchors [m].
        tangents (torch.Tensor): (T,B,2) unit route tangents.
        anchor_valid (torch.Tensor): (T,B) bool.
        headings (torch.Tensor): (T,B) reference headings [rad].
        centres (torch.Tensor): (B,S,2) segment centres [m].
        directions (torch.Tensor): (B,S,2) unit segment directions.
        half_lengths (torch.Tensor): (B,S) segment half-lengths [m].
        valid (torch.Tensor): (B,S) bool, observed segments.
        ego_length_m (torch.Tensor): (B,) ego length [m].
        ego_width_m (torch.Tensor): (B,) ego width [m].
        margin_m (float): mu, distance the body keeps from the edge [m].
        reference_positions (torch.Tensor | None): (T,B,2) reference
            positions [m]; when given, an edge point counts for side h only
            if it also lies on side h of node k's reference position (the
            island filter: an island edge the plan has already passed lies
            on the other side of the reference and would give a plane the
            plan violates from the start).
        parallel_max_rad (float): edge segments count when they run within
            this angle of the tangent; pi / 2 or more keeps every segment.
        max_offset_m (float): d_h without a closer edge [m], measured from
            the anchor (on the route).
        along_window_m (float): edge points count within this distance
            along the tangent from the anchor; math.inf keeps every one.

    Returns:
        tuple: planes (T,B,2,3) (a_x, a_y, b), left then right, and
            guarded (T,B,2) bool, anchored rows their anchor violates.
    """
    normal = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)
    car_tangent = torch.stack([torch.cos(headings), torch.sin(headings)], -1)
    car_normal = torch.stack(
        [-torch.sin(headings), torch.cos(headings)], -1
    )
    parallel = (
        (directions[None] * tangents[:, :, None]).sum(-1).abs()
        >= math.cos(min(parallel_max_rad, math.pi / 2))
    )
    # (T,B,S): every anchor against every segment of its row
    along = ((anchors[:, :, None] - centres[None]) * directions[None]).sum(-1)
    clipped = torch.maximum(
        torch.minimum(along, half_lengths[None]), -half_lengths[None]
    )
    projections = centres[None] + clipped[..., None] * directions[None]
    delta = projections - anchors[:, :, None]
    along_route_m = (delta * tangents[:, :, None]).sum(-1)
    lateral_m = (delta * normal[:, :, None]).sum(-1)
    candidate = (
        valid[None] & parallel
        & (along_route_m.abs() <= along_window_m)
    )
    dummy = torch.tensor(
        [0.0, 1.0, PLANE_DUMMY_OFFSET_M],
        dtype=anchors.dtype, device=anchors.device,
    )
    reference_lateral_m = None
    if reference_positions is not None:
        reference_lateral_m = (
            (projections - reference_positions[:, :, None])
            * normal[:, :, None]
        ).sum(-1)
    planes, guards = [], []
    for side_sign in SIDE_SIGNS:
        side_lateral_m = side_sign * lateral_m
        side_candidate = candidate & (side_lateral_m > 0.0)
        if reference_lateral_m is not None:
            side_candidate = side_candidate & (
                side_sign * reference_lateral_m > 0.0
            )
        offset_side_m = torch.where(
            side_candidate, side_lateral_m, math.inf
        ).min(dim=-1).values.clamp_max(max_offset_m)
        side_normal = side_sign * normal
        body_offset_m = (
            0.5 * ego_length_m * (side_normal * car_tangent).sum(-1).abs()
            + 0.5 * ego_width_m * (side_normal * car_normal).sum(-1).abs()
            + margin_m
        )
        bound = (
            (side_normal * anchors).sum(-1) + offset_side_m - body_offset_m
        )
        guarded = anchor_valid & (offset_side_m < body_offset_m)
        plane = torch.cat([side_normal, bound[..., None]], dim=-1)
        keep = (anchor_valid & ~guarded)[..., None]
        planes.append(torch.where(keep, plane, dummy))
        guards.append(guarded)
    return torch.stack(planes, dim=2), torch.stack(guards, dim=2)


def count_fallback_planes(
        planes: torch.Tensor,
        anchors: torch.Tensor,
        anchor_valid: torch.Tensor,
        guarded: torch.Tensor,
        ego_length_m: torch.Tensor,
        ego_width_m: torch.Tensor,
        margin_m: float,
        max_offset_m: float,
    ) -> int:
    """
    Closed-loop route rows (node >= 1, side) with no edge in reach, whose
    plane sits at the max_offset_m cap (an evaluation diagnostic).

    The body offset is at most (l_e + w_e) / 2 + mu, so a row whose
    offset from its anchor reaches max_offset_m minus that bound is
    counted; an edge that close to the cap counts with it.

    Args:
        planes (torch.Tensor): (T,B,2,3) route_planes rows.
        anchors (torch.Tensor): (T,B,2) route anchors [m].
        anchor_valid (torch.Tensor): (T,B) bool.
        guarded (torch.Tensor): (T,B,2) bool, route_planes guards.
        ego_length_m (torch.Tensor): (B,) ego length [m].
        ego_width_m (torch.Tensor): (B,) ego width [m].
        margin_m (float): mu [m].
        max_offset_m (float): the route plane offset cap [m].

    Returns:
        int: counted rows.
    """
    offset_m = planes[..., 2] - (planes[..., :2] * anchors[:, :, None]).sum(-1)
    body_bound_m = 0.5 * (ego_length_m + ego_width_m) + margin_m
    at_cap = offset_m >= max_offset_m - body_bound_m[None, :, None]
    counted = anchor_valid[..., None] & ~guarded & at_cap
    return int(counted[1:].sum())
