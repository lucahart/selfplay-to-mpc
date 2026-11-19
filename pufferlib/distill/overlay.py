"""What the MPC student planned with, packed for the video overlay.

Per agent one row of OVERLAY_FLOATS floats in the ego frame of the step the
plan was made at (the MPC's frame), as `vec_set_overlay` (env_binding.h)
reads it and `draw_overlay` (render.h) draws it; the constants mirror
datatypes.h. Layout:

    [valid, node_count, plane_count, ellipse_count,
     per node k < OVERLAY_MAX_NODES:
         px_k, py_k, target_y_k, q_y_k,
         per plane h < OVERLAY_MAX_PLANES: a_x, a_y, b,
         per ellipse i < OVERLAY_MAX_ELLIPSES:
             px, py, cos, sin, semi-axis along, across]

Torch only, so the main venv can import it.
"""

import torch

from pufferlib.distill.tube import planned_states

# Mirrored in datatypes.h
OVERLAY_MAX_NODES = 20
OVERLAY_MAX_PLANES = 3
OVERLAY_MAX_ELLIPSES = 8
OVERLAY_HEADER = 4
PLANE_FLOATS = 3
ELLIPSE_FLOATS = 6
NODE_PLANES_START = 4
NODE_ELLIPSES_START = NODE_PLANES_START + PLANE_FLOATS * OVERLAY_MAX_PLANES
OVERLAY_NODE_FLOATS = (
    NODE_ELLIPSES_START + ELLIPSE_FLOATS * OVERLAY_MAX_ELLIPSES
)
OVERLAY_FLOATS = OVERLAY_HEADER + OVERLAY_MAX_NODES * OVERLAY_NODE_FLOATS
OVERLAY_VALID_IDX = 0
OVERLAY_NODE_COUNT_IDX = 1
OVERLAY_PLANE_COUNT_IDX = 2
OVERLAY_ELLIPSE_COUNT_IDX = 3
NODE_POSITION_SLICE = slice(0, 2)
NODE_TARGET_Y_IDX = 2
NODE_WEIGHT_Y_IDX = 3


def planned_positions(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
    """
    Positions of the plan's nodes: the controls rolled out through the
    bicycle from x0, node 0 at x0.

    Args:
        x0 (torch.Tensor): (B,5) ego-frame start state.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        controls (torch.Tensor): (T,B,2) plan.

    Returns:
        torch.Tensor: (B,T,2) positions [m].
    """
    states = planned_states(x0, wheelbase, controls)
    return states[..., :2].transpose(0, 1)


def pack_overlay(
        positions: torch.Tensor,
        target_y: torch.Tensor,
        weight_y: torch.Tensor,
        planes: torch.Tensor | None,
        ellipses: torch.Tensor | None = None,
    ) -> torch.Tensor:
    """
    Overlay rows of a solve.

    Args:
        positions (torch.Tensor): (B,T,2) planned positions [m].
        target_y (torch.Tensor): (T,B) lateral target of the cost [m].
        weight_y (torch.Tensor): (T,B) lateral weight of the cost.
        planes (torch.Tensor | None): (T,B,H,3) (a_x, a_y, b), or None
            without planes.
        ellipses (torch.Tensor | None): (T,B,K,6) partner ellipses
            (obstacles.py layout), or None without.

    Returns:
        torch.Tensor: (B,OVERLAY_FLOATS) float32 rows.
    """
    row_count, horizon_steps = positions.shape[:2]
    plane_count = 0 if planes is None else planes.shape[2]
    ellipse_count = 0 if ellipses is None else ellipses.shape[2]
    if horizon_steps > OVERLAY_MAX_NODES:
        raise ValueError(
            f"the overlay holds at most {OVERLAY_MAX_NODES} nodes"
        )
    if plane_count > OVERLAY_MAX_PLANES:
        raise ValueError(
            f"the overlay holds at most {OVERLAY_MAX_PLANES} planes"
        )
    if ellipse_count > OVERLAY_MAX_ELLIPSES:
        raise ValueError(
            f"the overlay holds at most {OVERLAY_MAX_ELLIPSES} ellipses"
        )
    rows = torch.zeros(
        row_count, OVERLAY_FLOATS, dtype=torch.float32,
        device=positions.device,
    )
    rows[:, OVERLAY_VALID_IDX] = 1.0
    rows[:, OVERLAY_NODE_COUNT_IDX] = float(horizon_steps)
    rows[:, OVERLAY_PLANE_COUNT_IDX] = float(plane_count)
    rows[:, OVERLAY_ELLIPSE_COUNT_IDX] = float(ellipse_count)
    nodes = rows[:, OVERLAY_HEADER:].view(
        row_count, OVERLAY_MAX_NODES, OVERLAY_NODE_FLOATS
    )
    nodes[:, :horizon_steps, NODE_POSITION_SLICE] = positions
    nodes[:, :horizon_steps, NODE_TARGET_Y_IDX] = target_y.transpose(0, 1)
    nodes[:, :horizon_steps, NODE_WEIGHT_Y_IDX] = weight_y.transpose(0, 1)
    if planes is not None:
        planes_end = NODE_PLANES_START + PLANE_FLOATS * plane_count
        nodes[:, :horizon_steps, NODE_PLANES_START:planes_end] = (
            planes.transpose(0, 1).flatten(2)
        )
    if ellipses is not None:
        ellipses_end = NODE_ELLIPSES_START + ELLIPSE_FLOATS * ellipse_count
        nodes[:, :horizon_steps, NODE_ELLIPSES_START:ellipses_end] = (
            ellipses.transpose(0, 1).flatten(2)
        )
    return rows
