"""The MPC student's problem as a WarpMPC batched SQP.

Same problem as `solve` + `target_cost` in bicycle.py: per step the cost
0.5 * sum(q * (tau - target)^2) on tau = [x; u], the classic bicycle as
equality rows, a symmetric control box and x_0 fixed to the ego state.
WarpMPC's horizon N counts intervals and has N + 1 nodes z_k = [x_k; u_k];
with N = T - 1 the nodes are tau_0 ... tau_{T-1} and x_T is not a
variable, as in mpc.pytorch (u_{T-1} only moves the uncosted x_T).

Per-row parameters, node after node:
    first    (k = 0):         q_0, target_0, x0, wheelbase   (20)
    middle   (0 < k < T-1):   q_k, target_k, wheelbase       (15)
    terminal (k = T-1):       q_k, target_k                  (14)

With C constraint rows (constraints.py: planes, partner ellipses, or the
tube's two road planes then its ellipses) every node k >= 1 also carries
C slacks s >= 0 priced in the cost, so it is z_k = [x; u; s], and its
parameters end with its constraint values (constraint_value_count).
Node 0 is pinned to x0, so it has no constraint rows and no slacks
(WarpMPC allows a different width per node). The slacks keep the QP
feasible when x0 already violates a row.

Requires the WarpMPC venv (JAX, Warp, CasADi); import this module only
when the WarpMPC solver is selected.
"""

import contextlib

import casadi as ca
import jax
import jax.dlpack
import numpy as np
import torch
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    CasadiStageFunction,
    SparseMPCProblem,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)

from pufferlib.distill.bicycle import (
    DT_SECONDS,
    MAX_SPEED_MPS,
    NU,
    NX,
    REAR_AXLE_RATIO,
    STEERING_LIMIT_RAD,
    Bicycle,
)
from pufferlib.distill.constraints import (
    CONSTRAINT_ELLIPSE,
    CONSTRAINT_NONE,
    CONSTRAINT_PLANE,
    CONSTRAINT_TUBE,
    PLANE_PARAM_COUNT,
    TUBE_PLANE_COUNT,
    TUBE_PLANE_VALUE_COUNT,
    constraint_count_from_params,
    constraint_slack_count,
    constraint_value_count,
    dummy_constraints,
    validate_constraints,
)
from pufferlib.distill.obstacles import OBSTACLE_PARAM_COUNT

TAU_DIM = NX + NU
FIRST_PARAM_COUNT = 2 * TAU_DIM + NX + 1
MIDDLE_PARAM_COUNT = 2 * TAU_DIM + 1
TERMINAL_PARAM_COUNT = 2 * TAU_DIM
# The stage layout needs a first and a terminal node
MIN_HORIZON_STEPS = 2
# Iterative refinement of the adjoint solve: WarpMPC's default of 100 cost
# 26 ms of a 28 ms backward at 1024 rows. Against finite differences the
# median directional error is 2.6 % at 0 steps and 0.35 % from 2 steps on,
# the same as at 100
DEFAULT_REFINEMENT_ITERATIONS = 2
# ADMM penalty of the inequality rows (equality rows get 1e3 times it).
# OSQP's default 0.1 left 25 ADMM steps with controls outside the box on
# the costs training reaches (10 % of rows, up to 2.3 times the box), and
# training exploited it until the loss diverged; 3.0 keeps every row in
# the box with the same agreement with iLQR
ADMM_RHO = 3.0
# ADMM proximal term on the variables (OSQP's sigma); it does not move the
# optimum. At OSQP's default 1e-6, standstill rows whose state weights the
# head drove to ~1e-14 have no curvature on states the controls cannot move
# yet, the fp32 factorization breaks down and the solve returns NaN (a
# T = 10 DAgger run); 1e-3 solves them and matches iLQR as closely
ADMM_SIGMA = 1e-3
# An SQP step is rejected for a row (the row keeps its iterate) when it
# leaves the box by more than this fraction or is not finite: on rare rows
# with extreme costs 5 SQP steps at T = 10 diverged to |u| ~ 1e9 times the
# box, where the filter line search accepted a step of an unconverged QP
STEP_BOX_TOLERANCE = 0.05
# Compiled row counts are powers of two from this size on; the solve time
# is nearly flat in the rows (8.7 / 9.3 / 10.4 ms forward + backward at
# 1k / 4k / 16k rows), so padding is cheaper than one compile per count
MIN_PADDED_ROW_COUNT = 256
# Padding rows: unit weights, zero targets, standstill, a typical car; they
# solve without NaN and are cut off before anything reads them
PAD_WHEELBASE_M = 2.7
SOLVER_DTYPES = {torch.float32: np.float32, torch.float64: np.float64}
# Bound of one-sided rows (WarpMPC's OSQP infinity)
ROW_INFINITY = 1e30
# Price of a constraint slack s: linear * s + 0.5 * quadratic * s^2. The
# linear term makes the penalty exact while the row's multiplier stays
# below it, the quadratic one gives the QP curvature on s.
# Planes, T = 10, 25 ADMM steps: the quadratic price sets
# the ADMM conditioning, the linear one how hard the plane is. At 100 /
# 1000 a plane that braking cannot satisfy (p_x <= 0 at 10 m/s) left the
# first QP step at 1.48 times the accel box: the slack curvature dwarfs
# ADMM_RHO, every step was rejected and the row froze at its warm start
# without gradient. 100 / 10 keeps the step in the box and still holds
# p_y <= 0.3 m against the head's strongest pull (lateral weight 10,
# target 5 m: 0.315 m); a linear price of 30 lets it reach 1.3 m, 10
# lets it reach 2.6 m
PLANE_SLACK_LINEAR_WEIGHT = 100.0
PLANE_SLACK_QUADRATIC_WEIGHT = 10.0
# Ellipses keep the price they were measured with
ELLIPSE_SLACK_LINEAR_WEIGHT = 100.0
ELLIPSE_SLACK_QUADRATIC_WEIGHT = 1000.0
# The ellipse row is the normalized distance sqrt(v + eps), not v: the
# ADMM adds rho * a a^T of every row, active or not, to its matrix (no
# OSQP scaling), and v's gradient grows with the distance (2 d / a^2, ~1e3
# for a dummy 500 m away), which moved the plans of rows no ellipse
# touched by up to the full box. eps keeps it smooth at the centre
ELLIPSE_DISTANCE_EPS = 1e-4

# Compiled solvers by (T, rows, bound, SQP, ADMM, dtype, constraint type,
# constraint count); module level so
# students can be deep-copied and pickled without the compiled functions
_SOLVER_CACHE = {}


def node_dim(
        constraint_count: int,
    ) -> int:
    """
    Variables of a node k >= 1: tau = [x; u] and one slack per constraint
    row (node 0 has tau only).

    Args:
        constraint_count (int): C, slacks per node
            (constraints.constraint_slack_count).

    Returns:
        int: 7 + C.
    """
    return TAU_DIM + constraint_count


def z_dim(
        horizon_steps: int,
        constraint_count: int,
    ) -> int:
    """
    Variables per row: tau_0, then tau_k and C slacks for k = 1..T-1.

    Args:
        horizon_steps (int): T.
        constraint_count (int): C.

    Returns:
        int: 7 + (T - 1) * (7 + C).
    """
    return TAU_DIM + (horizon_steps - 1) * node_dim(constraint_count)


def param_count(
        horizon_steps: int,
        constraint_type: str = CONSTRAINT_NONE,
        constraint_count: int = 0,
        node_slack_price: bool = False,
    ) -> int:
    """
    Parameters per row for a horizon.

    Args:
        horizon_steps (int): T, number of tau nodes.
        constraint_type (str): one of constraints.CONSTRAINT_TYPES.
        constraint_count (int): rows per node; ellipses for the tube.
        node_slack_price (bool): one slack price per node k >= 1 after
            its constraint values.

    Returns:
        int: 20 + 15 * (T - 2) + 14 + (T - 1) * V, V the constraint
            values per node (plus the price).
    """
    middle_count = horizon_steps - MIN_HORIZON_STEPS
    constraint_values = constraint_value_count(
        constraint_type, constraint_count
    ) + int(node_slack_price)
    return (
        FIRST_PARAM_COUNT
        + middle_count * MIDDLE_PARAM_COUNT
        + TERMINAL_PARAM_COUNT
        + (horizon_steps - 1) * constraint_values
    )


def tau_nodes(
        z: torch.Tensor,
        horizon_steps: int,
        constraint_count: int,
    ) -> torch.Tensor:
    """
    tau = [x; u] of every node from a solution (slacks dropped).

    Args:
        z (torch.Tensor): (R,z_dim) solution.
        horizon_steps (int): T.
        constraint_count (int): C.

    Returns:
        torch.Tensor: (R,T,7) tau.
    """
    later = z[:, TAU_DIM:].reshape(
        z.shape[0], horizon_steps - 1, node_dim(constraint_count)
    )
    return torch.cat([z[:, None, :TAU_DIM], later[..., :TAU_DIM]], dim=1)


def node_slacks(
        z: torch.Tensor,
        horizon_steps: int,
        constraint_count: int,
    ) -> torch.Tensor:
    """
    Slacks of nodes 1..T-1 from a solution.

    Args:
        z (torch.Tensor): (R,z_dim) solution.
        horizon_steps (int): T.
        constraint_count (int): C.

    Returns:
        torch.Tensor: (R,T-1,C) slacks.
    """
    later = z[:, TAU_DIM:].reshape(
        z.shape[0], horizon_steps - 1, node_dim(constraint_count)
    )
    return later[..., TAU_DIM:]


def bicycle_step_casadi(
        x: ca.SX,
        u: ca.SX,
        wheelbase_m: ca.SX,
    ) -> ca.SX:
    """
    Bicycle.forward (bicycle.py) in CasADi: same order and clamps.

    Args:
        x (ca.SX): (5,) state (px, py, heading, speed, steering).
        u (ca.SX): (2,) control (accel [m/s^2], steering_rate [rad/s]).
        wheelbase_m (ca.SX): scalar wheelbase [m].

    Returns:
        ca.SX: (5,) next state.
    """
    px, py, heading, speed, steering = (x[idx] for idx in range(NX))
    accel, steering_rate = u[0], u[1]
    # drive.h updates steering and speed first; the pose uses the new ones
    next_steering = ca.fmin(
        ca.fmax(steering + steering_rate * DT_SECONDS, -STEERING_LIMIT_RAD),
        STEERING_LIMIT_RAD,
    )
    next_speed = ca.fmin(
        ca.fmax(speed + accel * DT_SECONDS, -MAX_SPEED_MPS),
        MAX_SPEED_MPS,
    )
    tan_steering = ca.tan(next_steering)
    beta = ca.atan(REAR_AXLE_RATIO * tan_steering)  # slip angle
    yaw_rate = next_speed * ca.cos(beta) * tan_steering / wheelbase_m
    # Position moves along the old heading plus slip; heading changes after
    course = heading + beta
    return ca.vertcat(
        px + next_speed * ca.cos(course) * DT_SECONDS,
        py + next_speed * ca.sin(course) * DT_SECONDS,
        heading + yaw_rate * DT_SECONDS,
        next_speed,
        next_steering,
    )


def slack_cost(
        slack: ca.SX,
        linear_weight: float,
        quadratic_weight: float,
    ) -> ca.SX:
    """
    Price of a node's slacks.

    Args:
        slack (ca.SX): (C,) slacks.
        linear_weight (float): price per unit of slack.
        quadratic_weight (float): curvature of the price.

    Returns:
        ca.SX: scalar linear_weight * sum(s)
            + 0.5 * quadratic_weight * sum(s^2).
    """
    return (
        linear_weight * ca.sum1(slack)
        + 0.5 * quadratic_weight * ca.sum1(slack ** 2)
    )


def plane_terms(
        x: ca.SX,
        slack: ca.SX,
        plane_params: ca.SX,
        slack_weight_scale: float = 1.0,
        slack_prices: tuple[float, float] | None = None,
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
    """
    Slack cost and rows of the H planes of one node:
    a . p - b - s <= 0 and s >= 0, p = (px, py).

    Args:
        x (ca.SX): (5,) state.
        slack (ca.SX): (H,) slacks.
        plane_params (ca.SX): (H*3,) per plane (a_px, a_py, b), a unit.
        slack_weight_scale (float): factor on both slack prices.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price replacing the scaled constants; None keeps them.

    Returns:
        tuple: slack cost (scalar), rows (2H,), lower (2H,), upper (2H,).
    """
    plane_count = slack.shape[0]
    values = []
    for plane_idx in range(plane_count):
        offset = plane_idx * PLANE_PARAM_COUNT
        normal_x, normal_y, plane_offset = (
            plane_params[offset + idx] for idx in range(PLANE_PARAM_COUNT)
        )
        # b inside the row, not in its bound, so its derivative is the
        # row's parameter derivative like a's
        values.append(
            normal_x * x[0] + normal_y * x[1] - plane_offset
            - slack[plane_idx]
        )
    rows = ca.vertcat(*values, slack)
    lower = ca.vertcat(ca.DM.ones(plane_count) * -ROW_INFINITY,
                       ca.DM.zeros(plane_count))
    upper = ca.vertcat(ca.DM.zeros(plane_count),
                       ca.DM.ones(plane_count) * ROW_INFINITY)
    if slack_prices is None:
        slack_prices = (
            slack_weight_scale * PLANE_SLACK_LINEAR_WEIGHT,
            slack_weight_scale * PLANE_SLACK_QUADRATIC_WEIGHT,
        )
    cost = slack_cost(slack, *slack_prices)
    return cost, rows, lower, upper


def obstacle_terms(
        x: ca.SX,
        slack: ca.SX,
        obstacle_params: ca.SX,
        slack_weight_scale: float = 1.0,
        slack_prices: tuple[float, float] | None = None,
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
    """
    Slack cost and rows of the K ellipses of one node:
    sqrt(ellipse_value + eps) + s >= 1 and s >= 0.

    Args:
        x (ca.SX): (5,) state.
        slack (ca.SX): (K,) slacks.
        obstacle_params (ca.SX): (K*6,) per obstacle (px, py, cos, sin,
            semi-axis along its heading, across it) [m].
        slack_weight_scale (float): factor on both slack prices.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price replacing the scaled constants; None keeps them.

    Returns:
        tuple: slack cost (scalar), rows (2K,), lower (2K,), upper (2K,).
    """
    obstacle_count = slack.shape[0]
    values = []
    for obstacle_idx in range(obstacle_count):
        offset = obstacle_idx * OBSTACLE_PARAM_COUNT
        px, py, cos_heading, sin_heading, semi_long, semi_lat = (
            obstacle_params[offset + idx]
            for idx in range(OBSTACLE_PARAM_COUNT)
        )
        dx, dy = x[0] - px, x[1] - py
        along = (dx * cos_heading + dy * sin_heading) / semi_long
        across = (dy * cos_heading - dx * sin_heading) / semi_lat
        distance = ca.sqrt(along ** 2 + across ** 2 + ELLIPSE_DISTANCE_EPS)
        values.append(distance + slack[obstacle_idx])
    rows = ca.vertcat(*values, slack)
    lower = ca.vertcat(ca.DM.ones(obstacle_count),
                       ca.DM.zeros(obstacle_count))
    upper = ca.DM.ones(2 * obstacle_count) * ROW_INFINITY
    if slack_prices is None:
        slack_prices = (
            slack_weight_scale * ELLIPSE_SLACK_LINEAR_WEIGHT,
            slack_weight_scale * ELLIPSE_SLACK_QUADRATIC_WEIGHT,
        )
    cost = slack_cost(slack, *slack_prices)
    return cost, rows, lower, upper


def tube_terms(
        x: ca.SX,
        slack: ca.SX,
        tube_params: ca.SX,
        slack_weight_scale: float = 1.0,
        slack_prices: tuple[float, float] | None = None,
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
    """
    Slack cost and rows of one tube node: the left and right road planes
    (plane_terms), then K partner ellipses (obstacle_terms).

    Args:
        x (ca.SX): (5,) state.
        slack (ca.SX): (2+K,) slacks, planes first.
        tube_params (ca.SX): (6+6K,) plane then ellipse parameters.
        slack_weight_scale (float): factor on both slack prices.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price of planes and ellipses alike; None keeps the constants.

    Returns:
        tuple: slack cost (scalar), rows (2(2+K),), lower and upper.
    """
    plane_cost, plane_rows, plane_lower, plane_upper = plane_terms(
        x,
        slack[:TUBE_PLANE_COUNT],
        tube_params[:TUBE_PLANE_VALUE_COUNT],
        slack_weight_scale,
        slack_prices,
    )
    if slack.shape[0] == TUBE_PLANE_COUNT:
        return plane_cost, plane_rows, plane_lower, plane_upper
    ellipse_cost, ellipse_rows, ellipse_lower, ellipse_upper = (
        obstacle_terms(
            x,
            slack[TUBE_PLANE_COUNT:],
            tube_params[TUBE_PLANE_VALUE_COUNT:],
            slack_weight_scale,
            slack_prices,
        )
    )
    return (
        plane_cost + ellipse_cost,
        ca.vertcat(plane_rows, ellipse_rows),
        ca.vertcat(plane_lower, ellipse_lower),
        ca.vertcat(plane_upper, ellipse_upper),
    )


def constraint_terms(
        constraint_type: str,
        x: ca.SX,
        slack: ca.SX,
        constraint_params: ca.SX,
        slack_weight_scale: float = 1.0,
        slack_prices: tuple[float, float] | None = None,
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
    """
    Slack cost and rows of one node's constraints of the given type.

    Args:
        constraint_type (str): one of constraints.CONSTRAINT_TYPES.
        x (ca.SX): (5,) state.
        slack (ca.SX): (C,) slacks.
        constraint_params (ca.SX): (V,) parameters.
        slack_weight_scale (float): factor on every slack price.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price of every slack; None keeps the scaled constants.

    Returns:
        tuple: slack cost (scalar), rows (2C,), lower (2C,), upper (2C,);
            zero cost and no rows for CONSTRAINT_NONE.
    """
    if constraint_type == CONSTRAINT_PLANE:
        return plane_terms(
            x, slack, constraint_params, slack_weight_scale, slack_prices
        )
    if constraint_type == CONSTRAINT_ELLIPSE:
        return obstacle_terms(
            x, slack, constraint_params, slack_weight_scale, slack_prices
        )
    if constraint_type == CONSTRAINT_TUBE:
        return tube_terms(
            x, slack, constraint_params, slack_weight_scale, slack_prices
        )
    return ca.SX(0.0), ca.SX(0, 1), ca.DM(0, 1), ca.DM(0, 1)


def bicycle_stage_functions(
        control_bound: np.ndarray,
        constraint_type: str = CONSTRAINT_NONE,
        constraint_count: int = 0,
        slack_weight_scale: float = 1.0,
        plan_control_bound: np.ndarray | None = None,
        slack_prices: tuple[float, float] | None = None,
        node_slack_price: bool = False,
        slack_quadratic_fraction: float = 0.0,
    ) -> tuple[CasadiStageFunction, CasadiStageFunction, CasadiStageFunction]:
    """
    First, middle and terminal stage functions of the student's problem.
    Constraint rows: dynamics defect (equality), control box, on the first
    node x_0 = x0 (equality); on later nodes the rows of constraint_terms.
    The first node has no slacks: its z is tau, its z_next tau and slacks.

    Args:
        control_bound (np.ndarray): (2,) symmetric box (accel [m/s^2],
            steering_rate [rad/s]).
        constraint_type (str): one of constraints.CONSTRAINT_TYPES.
        constraint_count (int): rows per node k >= 1 (0 for none);
            ellipses for the tube.
        slack_weight_scale (float): factor on every slack price.
        plan_control_bound (np.ndarray | None): (2,) symmetric box of
            nodes k >= 1; None for control_bound on every node.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price of every slack; None keeps the scaled constants.
        node_slack_price (bool): the last parameter of nodes k >= 1 is
            their linear slack price p (quadratic: fraction * p),
            replacing slack_prices.
        slack_quadratic_fraction (float): quadratic price per unit of p.

    Returns:
        tuple: first, middle and terminal CasadiStageFunction.
    """
    first_bound = ca.DM(np.asarray(control_bound, dtype=np.float64))
    if plan_control_bound is None:
        plan_control_bound = control_bound
    bound = ca.DM(np.asarray(plan_control_bound, dtype=np.float64))
    defect_zero = ca.DM.zeros(NX)

    def tracking_cost(
            z: ca.SX,
            q: ca.SX,
            target: ca.SX,
        ) -> ca.SX:
        """
        Diagonal tracking cost of one node.

        Args:
            z (ca.SX): (7,) tau = [x; u] of the node.
            q (ca.SX): (7,) weights.
            target (ca.SX): (7,) target.

        Returns:
            ca.SX: scalar 0.5 * sum(q * (z - target)^2).
        """
        return 0.5 * ca.sum1(q * (z - target) ** 2)

    constraint_values = constraint_value_count(
        constraint_type, constraint_count
    )
    slack_count = constraint_slack_count(constraint_type, constraint_count)
    first_z = ca.SX.sym("z_first", TAU_DIM)
    z = ca.SX.sym("z", node_dim(slack_count))
    z_next = ca.SX.sym("z_next", node_dim(slack_count))
    tau, x, u = z[:TAU_DIM], z[:NX], z[NX:TAU_DIM]
    slack = z[TAU_DIM:]

    first_p = ca.SX.sym("p_first", FIRST_PARAM_COUNT)
    q, target = first_p[:TAU_DIM], first_p[TAU_DIM:2 * TAU_DIM]
    x0 = first_p[2 * TAU_DIM:2 * TAU_DIM + NX]
    wheelbase = first_p[FIRST_PARAM_COUNT - 1]
    first_x, first_u = first_z[:NX], first_z[NX:]
    first = ca.Function(
        "bicycle_first",
        [first_z, z_next, first_p],
        [
            tracking_cost(first_z, q, target),
            ca.vertcat(
                z_next[:NX] - bicycle_step_casadi(first_x, first_u, wheelbase),
                first_u,
                first_x,
            ),
            ca.vertcat(defect_zero, -first_bound, x0),
            ca.vertcat(defect_zero, first_bound, x0),
        ],
    )

    price_count = int(node_slack_price)
    middle_p = ca.SX.sym(
        "p_middle", MIDDLE_PARAM_COUNT + constraint_values + price_count
    )
    q, target = middle_p[:TAU_DIM], middle_p[TAU_DIM:2 * TAU_DIM]
    wheelbase = middle_p[MIDDLE_PARAM_COUNT - 1]
    node_prices = slack_prices
    if node_slack_price:
        price = middle_p[MIDDLE_PARAM_COUNT + constraint_values]
        node_prices = (price, slack_quadratic_fraction * price)
    slack_price, rows, lower, upper = constraint_terms(
        constraint_type,
        x,
        slack,
        middle_p[MIDDLE_PARAM_COUNT:MIDDLE_PARAM_COUNT + constraint_values],
        slack_weight_scale,
        node_prices,
    )
    middle = ca.Function(
        "bicycle_middle",
        [z, z_next, middle_p],
        [
            tracking_cost(tau, q, target) + slack_price,
            ca.vertcat(z_next[:NX] - bicycle_step_casadi(x, u, wheelbase),
                       u, rows),
            ca.vertcat(defect_zero, -bound, lower),
            ca.vertcat(defect_zero, bound, upper),
        ],
    )

    terminal_p = ca.SX.sym(
        "p_terminal", TERMINAL_PARAM_COUNT + constraint_values + price_count
    )
    q, target = terminal_p[:TAU_DIM], terminal_p[TAU_DIM:2 * TAU_DIM]
    node_prices = slack_prices
    if node_slack_price:
        price = terminal_p[TERMINAL_PARAM_COUNT + constraint_values]
        node_prices = (price, slack_quadratic_fraction * price)
    slack_price, rows, lower, upper = constraint_terms(
        constraint_type,
        x,
        slack,
        terminal_p[
            TERMINAL_PARAM_COUNT:TERMINAL_PARAM_COUNT + constraint_values
        ],
        slack_weight_scale,
        node_prices,
    )
    terminal = ca.Function(
        "bicycle_terminal",
        [z, terminal_p],
        [
            tracking_cost(tau, q, target) + slack_price,
            ca.vertcat(u, rows),
            ca.vertcat(-bound, lower),
            ca.vertcat(bound, upper),
        ],
    )
    return (
        CasadiStageFunction.from_function(first, has_next=True),
        CasadiStageFunction.from_function(middle, has_next=True),
        CasadiStageFunction.from_function(terminal, has_next=False),
    )


def pack_params(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        q_diag: torch.Tensor,
        target: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
    """
    Row-major WarpMPC parameters from the student's cost terms.

    Args:
        x0 (torch.Tensor): (B,5) ego-frame start state.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        q_diag (torch.Tensor): (T,B,7) weights on tau.
        target (torch.Tensor): (T,B,7) target for tau.
        constraints (torch.Tensor | None): (T,B,C,P) constraint
            parameters (constraints.py), (T,B,V) for the tube, step 0
            unused; None for none.

    Returns:
        torch.Tensor: (B,param_count(T, type, C)) parameters,
            differentiable with respect to q_diag, target and constraints.
    """
    horizon_steps, row_count = q_diag.shape[:2]
    if constraints is None:
        constraints = q_diag.new_zeros(horizon_steps, row_count, 0)
    # Node 0 is pinned to x0 and has no constraint rows
    constraints = constraints.reshape(horizon_steps, row_count, -1)
    wheelbase_column = wheelbase[:, None]
    blocks = [q_diag[0], target[0], x0, wheelbase_column]
    for step_idx in range(1, horizon_steps - 1):
        blocks += [
            q_diag[step_idx], target[step_idx], wheelbase_column,
            constraints[step_idx],
        ]
    blocks += [q_diag[-1], target[-1], constraints[-1]]
    return torch.cat(blocks, dim=1)


def rollout_warm_start(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        u_init: torch.Tensor,
        constraint_count: int = 0,
    ) -> torch.Tensor:
    """
    Dynamically consistent SQP start point: u_init rolled out through
    Bicycle.forward, as mpc.pytorch starts its iLQR; slacks start at 0.

    Args:
        x0 (torch.Tensor): (B,5) start state.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        u_init (torch.Tensor): (T,B,2) controls.
        constraint_count (int): C, slacks per node k >= 1
            (constraints.constraint_slack_count).

    Returns:
        torch.Tensor: (B,z_dim(T, C)) z = [x_0, u_0, x_1, u_1, s_1, ...,
            x_{T-1}, u_{T-1}, s_{T-1}].
    """
    dynamics = Bicycle(wheelbase)
    slack = x0.new_zeros(x0.shape[0], constraint_count)
    nodes = []
    state = x0
    for step_idx, controls in enumerate(u_init):
        node = [state, controls] if step_idx == 0 else [
            state, controls, slack
        ]
        nodes.append(torch.cat(node, dim=1))
        state = dynamics(state, controls)
    return torch.cat(nodes, dim=1)


class CompiledBicycleSolver:
    """WarpMPC SQP for one horizon, row count, iteration counts and dtype."""

    def __init__(
            self,
            horizon_steps: int,
            row_count: int,
            control_bound: np.ndarray,
            sqp_iterations: int,
            admm_iterations: int,
            dtype: np.dtype = np.float32,
            refinement_iterations: int = DEFAULT_REFINEMENT_ITERATIONS,
            constraint_type: str = CONSTRAINT_NONE,
            constraint_count: int = 0,
            slack_weight_scale: float = 1.0,
            plan_control_bound: np.ndarray | None = None,
            slack_prices: tuple[float, float] | None = None,
            node_slack_price: bool = False,
            slack_quadratic_fraction: float = 0.0,
        ) -> None:
        """
        Build the stage functions and compile the batched SQP with OSQP
        adjoint derivatives.

        Args:
            horizon_steps (int): T, number of tau nodes (>= 2).
            row_count (int): batch rows every call must have.
            control_bound (np.ndarray): (2,) symmetric control box.
            sqp_iterations (int): SQP steps per solve (>= 1).
            admm_iterations (int): ADMM iterations per QP (>= 1).
            dtype (np.dtype): float32, or float64 with jax_enable_x64.
            refinement_iterations (int): iterative-refinement steps of the
                adjoint linear solve in the backward pass.
            constraint_type (str): one of constraints.CONSTRAINT_TYPES.
            constraint_count (int): rows with slack per node k >= 1;
                ellipses for the tube (its planes come on top).
            slack_weight_scale (float): factor on every slack price.
            plan_control_bound (np.ndarray | None): (2,) box of nodes
                k >= 1; None for control_bound on every node.
            slack_prices (tuple[float, float] | None): (linear,
                quadratic) price of every slack; None keeps the scaled
                constants.
            node_slack_price (bool): a linear slack price parameter per
                node k >= 1 (bicycle_stage_functions).
            slack_quadratic_fraction (float): quadratic price per unit of
                the node price.
        """
        if horizon_steps < MIN_HORIZON_STEPS:
            raise ValueError(
                f"horizon_steps must be at least {MIN_HORIZON_STEPS}"
            )
        if row_count <= 0 or sqp_iterations <= 0 or admm_iterations <= 0:
            raise ValueError("row and iteration counts must be positive")
        validate_constraints(constraint_type, constraint_count)
        self.horizon_steps = horizon_steps
        self.row_count = row_count
        self.sqp_iterations = sqp_iterations
        self.slack_count = constraint_slack_count(
            constraint_type, constraint_count
        )
        if plan_control_bound is None:
            plan_control_bound = control_bound
        self.control_bound = jax.numpy.asarray(control_bound, dtype=dtype)
        self.plan_control_bound = jax.numpy.asarray(
            plan_control_bound, dtype=dtype
        )
        first, middle, terminal = bicycle_stage_functions(
            control_bound,
            constraint_type,
            constraint_count,
            slack_weight_scale,
            plan_control_bound,
            slack_prices,
            node_slack_price,
            slack_quadratic_fraction,
        )
        problem = SparseMPCProblem.from_stage_functions(
            horizon=horizon_steps - 1,
            first=first,
            intermediate=middle,
            terminal=terminal,
        )
        # Warm starting carries the ADMM iterate from one SQP step to the
        # next within a solve; every solve starts from zeros. rho_is_vec
        # gives equality rows (dynamics, x0) a 1e3 times larger rho:
        # without it 3 SQP x 50 ADMM steps missed iLQR's first steering
        # rate by up to 44 % of the box, with it by 0.04 %
        settings = OSQPSettings(
            rho=ADMM_RHO,
            sigma=ADMM_SIGMA,
            max_iter=admm_iterations,
            warm_starting=True,
            rho_is_vec=True,
        )
        # Kept for problem-size reports (variables, rows, KKT nonzeros)
        self.plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=settings,
            derivatives=True,
        )
        expected_params = param_count(
            horizon_steps, constraint_type, constraint_count,
            node_slack_price,
        )
        if self.plan.n_parameters != expected_params:
            raise ValueError("stage parameter layout does not match")
        self.solver = compile_sparse_mpc_sqp(
            problem,
            self.plan,
            dtype=dtype,
            osqp_settings=settings,
            qdldl_backend="warp",
            derivatives=True,
            derivative_refinement_iters=refinement_iterations,
        )

    def accepted_rows(
            self,
            z_next: jax.Array,
        ) -> jax.Array:
        """
        Rows whose new SQP iterate is finite and inside the box (up to
        STEP_BOX_TOLERANCE).

        Args:
            z_next (jax.Array): (R,z_dim) candidate iterate.

        Returns:
            jax.Array: (R,1) bool.
        """
        later = z_next[:, TAU_DIM:].reshape(
            self.row_count, self.horizon_steps - 1,
            node_dim(self.slack_count),
        )
        first_ratio = jax.numpy.max(
            jax.numpy.abs(z_next[:, NX:TAU_DIM]) / self.control_bound, axis=1
        )
        later_ratio = jax.numpy.max(
            jax.numpy.abs(later[:, :, NX:TAU_DIM])
            / self.plan_control_bound,
            axis=(1, 2),
        )
        inside = jax.numpy.maximum(first_ratio, later_ratio) <= (
            1.0 + STEP_BOX_TOLERANCE
        )
        finite = jax.numpy.all(jax.numpy.isfinite(z_next), axis=1)
        return (inside & finite)[:, None]

    def solve(
            self,
            z_init: jax.Array,
            params: jax.Array,
        ) -> jax.Array:
        """
        sqp_iterations SQP steps. Only the last step is differentiable,
        taken from the frozen previous iterate, so its derivative is the
        implicit derivative of the (near-)optimum, as in mpc.pytorch.

        Args:
            z_init (jax.Array): (R,z_dim) start point.
            params (jax.Array): (R,P) parameters.

        Returns:
            jax.Array: (R,z_dim) solution z.
        """
        state = self.solver.init_state(self.row_count)
        frozen_params = jax.lax.stop_gradient(params)
        z = z_init
        for _ in range(self.sqp_iterations - 1):
            result, state = self.solver.step(z, frozen_params, state=state)
            z = jax.numpy.where(
                self.accepted_rows(result.z_next), result.z_next, z
            )
        state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
        result, _ = self.solver.step(
            jax.lax.stop_gradient(z), params, state=state
        )
        # Value of the line-search step, derivative of the full QP step:
        # the accepted step length scales z_next's derivative, and near
        # the optimum the filter often accepts less than 1 (the gradient
        # norm fell to 0.54 of mpc.pytorch's at 5 SQP steps)
        direction = result.solve.direction
        stepped = (
            jax.lax.stop_gradient(result.z_next)
            + direction
            - jax.lax.stop_gradient(direction)
        )
        # A rejected row keeps its iterate and gets no gradient
        return jax.numpy.where(
            self.accepted_rows(result.z_next),
            stepped,
            jax.lax.stop_gradient(z),
        )


def padded_row_count(
        row_count: int,
    ) -> int:
    """
    Compiled row count for a batch: the next power of two, at least
    MIN_PADDED_ROW_COUNT.

    Args:
        row_count (int): rows to solve (> 0).

    Returns:
        int: padded row count.
    """
    return max(MIN_PADDED_ROW_COUNT, 1 << (row_count - 1).bit_length())


def cached_solver(
        horizon_steps: int,
        row_count: int,
        control_bound: tuple[float, float],
        sqp_iterations: int,
        admm_iterations: int,
        dtype: np.dtype,
        constraint_type: str = CONSTRAINT_NONE,
        constraint_count: int = 0,
        slack_weight_scale: float = 1.0,
        plan_control_bound: tuple[float, float] | None = None,
        slack_prices: tuple[float, float] | None = None,
        node_slack_price: bool = False,
        slack_quadratic_fraction: float = 0.0,
    ) -> CompiledBicycleSolver:
    """
    Compiled solver for these settings, built on first use.

    Args:
        horizon_steps (int): T.
        row_count (int): padded row count.
        control_bound (tuple[float, float]): symmetric box.
        sqp_iterations (int): SQP steps per solve.
        admm_iterations (int): ADMM iterations per QP.
        dtype (np.dtype): float32 or float64.
        constraint_type (str): one of constraints.CONSTRAINT_TYPES.
        constraint_count (int): rows per node k >= 1; ellipses for the
            tube.
        slack_weight_scale (float): factor on every slack price.
        plan_control_bound (tuple[float, float] | None): box of nodes
            k >= 1; None for control_bound on every node.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price of every slack; None keeps the scaled constants.
        node_slack_price (bool): a slack price parameter per node.
        slack_quadratic_fraction (float): quadratic price per unit of it.

    Returns:
        CompiledBicycleSolver: the cached solver.
    """
    if plan_control_bound is None:
        plan_control_bound = control_bound
    key = (
        horizon_steps, row_count, control_bound, sqp_iterations,
        admm_iterations, np.dtype(dtype), constraint_type, constraint_count,
        slack_weight_scale, plan_control_bound, slack_prices,
        node_slack_price, slack_quadratic_fraction,
    )
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = CompiledBicycleSolver(
            horizon_steps, row_count, np.asarray(control_bound),
            sqp_iterations, admm_iterations, dtype,
            constraint_type=constraint_type,
            constraint_count=constraint_count,
            slack_weight_scale=slack_weight_scale,
            plan_control_bound=np.asarray(plan_control_bound),
            slack_prices=slack_prices,
            node_slack_price=node_slack_price,
            slack_quadratic_fraction=slack_quadratic_fraction,
        )
    return _SOLVER_CACHE[key]


def torch_to_jax(
        tensor: torch.Tensor,
    ) -> jax.Array:
    """
    Zero-copy view of a CUDA tensor as a JAX array.

    Args:
        tensor (torch.Tensor): CUDA tensor.

    Returns:
        jax.Array: same memory, same shape.
    """
    # JAX runs on its own stream: torch's pending writes must land first
    torch.cuda.current_stream(tensor.device).synchronize()
    return jax.dlpack.from_dlpack(tensor.detach().contiguous())


def jax_to_torch(
        array: jax.Array,
    ) -> torch.Tensor:
    """
    Zero-copy view of a JAX array as a torch tensor.

    Args:
        array (jax.Array): device array.

    Returns:
        torch.Tensor: same memory, same shape.
    """
    # torch reads on its own stream: JAX's kernels must be done
    return torch.from_dlpack(jax.block_until_ready(array))


def precision_scope(
        dtype: torch.dtype,
    ) -> contextlib.AbstractContextManager:
    """
    jax_enable_x64 around float64 solves only (the float64 retry of rows
    whose float32 gradient failed), so float32 solves keep JAX's default
    32-bit types.

    Args:
        dtype (torch.dtype): dtype of the solve.

    Returns:
        contextlib.AbstractContextManager: the scope.
    """
    if dtype == torch.float64:
        return jax.enable_x64(True)
    return contextlib.nullcontext()


class WarpMPCSolveFn(torch.autograd.Function):
    """CompiledBicycleSolver.solve as a torch op on CUDA tensors; the
    backward is WarpMPC's OSQP adjoint through jax.vjp."""

    @staticmethod
    def forward(
            ctx,
            solver: CompiledBicycleSolver,
            z_init: torch.Tensor,
            params: torch.Tensor,
        ) -> torch.Tensor:
        """
        Solve and keep the vjp closure for the backward.

        Args:
            ctx: autograd context.
            solver (CompiledBicycleSolver): compiled for params' rows.
            z_init (torch.Tensor): (R,z_dim) start point, no gradient.
            params (torch.Tensor): (R,P) parameters.

        Returns:
            torch.Tensor: (R,z_dim) solution z.
        """
        ctx.solve_dtype = params.dtype
        z_init_jax = torch_to_jax(z_init)
        z, ctx.vjp_fn = jax.vjp(
            lambda params_jax: solver.solve(z_init_jax, params_jax),
            torch_to_jax(params),
        )
        return jax_to_torch(z)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
            ctx,
            z_grad: torch.Tensor,
        ) -> tuple[None, None, torch.Tensor]:
        """
        Pull the solution's gradient back to the parameters.

        Args:
            ctx: autograd context.
            z_grad (torch.Tensor): (R,z_dim) dL/dz.

        Returns:
            tuple: no gradient for solver and z_init, dL/dparams (R,P).
        """
        with precision_scope(ctx.solve_dtype):
            (params_grad,) = ctx.vjp_fn(torch_to_jax(z_grad))
        # The closure holds the forward's device buffers; drop them now
        # rather than with the graph (cf. lqr_step_releasing_closures)
        del ctx.vjp_fn
        return None, None, jax_to_torch(params_grad)


def solve_cost_warpmpc(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        q_diag: torch.Tensor,
        target: torch.Tensor,
        u_init: torch.Tensor | None,
        control_bound: torch.Tensor,
        sqp_iterations: int,
        admm_iterations: int,
        constraint_type: str = CONSTRAINT_NONE,
        constraints: torch.Tensor | None = None,
        slack_weight_scale: float = 1.0,
        plan_control_bound: torch.Tensor | None = None,
        slack_prices: tuple[float, float] | None = None,
        node_slack_price: torch.Tensor | None = None,
        slack_quadratic_fraction: float = 0.0,
    ) -> torch.Tensor:
    """
    The student's MPC solved by WarpMPC: pad to the compiled row count,
    solve, cut the padding. Differentiable with respect to q_diag,
    target and constraints when grad is enabled.

    Args:
        x0 (torch.Tensor): (B,5) ego-frame start state, on CUDA.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        q_diag (torch.Tensor): (T,B,7) weights on tau.
        target (torch.Tensor): (T,B,7) target for tau.
        u_init (torch.Tensor | None): (T,B,2) warm start or None (zeros).
        control_bound (torch.Tensor): (2,) symmetric box.
        sqp_iterations (int): SQP steps per solve.
        admm_iterations (int): ADMM iterations per QP.
        constraint_type (str): one of constraints.CONSTRAINT_TYPES.
        constraints (torch.Tensor | None): (T,B,C,P) parameters of
            constraint_type, (T,B,V) for the tube (step 0 unused), None
            for CONSTRAINT_NONE.
        slack_weight_scale (float): factor on every slack price.
        plan_control_bound (torch.Tensor | None): (2,) box of nodes
            k >= 1; None for control_bound on every node.
        slack_prices (tuple[float, float] | None): (linear, quadratic)
            price of every slack; None keeps the scaled constants.
        node_slack_price (torch.Tensor | None): (T,B) linear slack price
            per node and row (node 0 unused), differentiable; replaces
            slack_prices. Needs constraints.
        slack_quadratic_fraction (float): quadratic price per unit of
            node_slack_price.

    Returns:
        torch.Tensor: (T,B,2) planned controls.
    """
    if not x0.is_cuda:
        raise ValueError("the WarpMPC solver needs CUDA tensors")
    if node_slack_price is not None and constraints is None:
        raise ValueError("node_slack_price needs constraints")
    horizon_steps, row_count = q_diag.shape[:2]
    pad_count = padded_row_count(row_count) - row_count
    dtype, device = q_diag.dtype, q_diag.device
    pad_rows = torch.zeros(pad_count, NX, dtype=dtype, device=device)
    pad_steps = torch.zeros(
        horizon_steps, pad_count, TAU_DIM, dtype=dtype, device=device
    )
    x0 = torch.cat([x0, pad_rows])
    wheelbase = torch.cat(
        [wheelbase, torch.full_like(pad_rows[:, 0], PAD_WHEELBASE_M)]
    )
    constraint_count = 0
    if constraints is not None:
        constraint_count = constraint_count_from_params(
            constraint_type, constraints
        )
        constraints = torch.cat([
            constraints,
            dummy_constraints(
                constraint_type, horizon_steps, pad_count, constraint_count,
                dtype, device,
            ),
        ], dim=1)
        if node_slack_price is not None:
            # Padding rows price their (inactive) slacks at 1
            prices = torch.cat([
                node_slack_price,
                node_slack_price.new_ones(horizon_steps, pad_count),
            ], dim=1)
            constraints = torch.cat([
                constraints.reshape(horizon_steps, constraints.shape[1], -1),
                prices[..., None],
            ], dim=-1)
    params = pack_params(
        x0,
        wheelbase,
        torch.cat([q_diag, pad_steps + 1.0], dim=1),
        torch.cat([target, pad_steps], dim=1),
        constraints,
    )
    if u_init is None:
        u_init = torch.zeros(
            horizon_steps, row_count, NU, dtype=dtype, device=device
        )
    # A warm start outside the box (a plan from a diverged solve) seeds
    # the next solve far from any feasible plan; mpc.pytorch's plans stay
    # in the box because its forward pass clips them
    if plan_control_bound is None:
        plan_control_bound = control_bound
    node_bound = torch.cat([
        control_bound[None],
        plan_control_bound[None].expand(horizon_steps - 1, NU),
    ])[:, None, :]
    u_init = torch.minimum(torch.maximum(u_init, -node_bound), node_bound)
    slack_count = constraint_slack_count(constraint_type, constraint_count)
    with torch.no_grad():
        z_init = rollout_warm_start(
            x0, wheelbase, torch.cat([u_init, pad_steps[..., :NU]], dim=1),
            slack_count,
        )
    with precision_scope(dtype):
        solver = cached_solver(
            horizon_steps,
            params.shape[0],
            tuple(control_bound.tolist()),
            sqp_iterations,
            admm_iterations,
            SOLVER_DTYPES[dtype],
            constraint_type,
            constraint_count,
            slack_weight_scale,
            tuple(plan_control_bound.tolist()),
            slack_prices,
            node_slack_price is not None,
            slack_quadratic_fraction,
        )
        if params.requires_grad:
            z = WarpMPCSolveFn.apply(solver, z_init, params)
        else:
            z = jax_to_torch(
                solver.solve(torch_to_jax(z_init), torch_to_jax(params))
            )
    if not bool(torch.isfinite(z).all()):
        raise ValueError("non-finite solution from the WarpMPC solve")
    nodes = tau_nodes(z, horizon_steps, slack_count)
    return nodes[:row_count, :, NX:].transpose(0, 1)
