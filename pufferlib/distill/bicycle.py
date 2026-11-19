"""PufferDrive CLASSIC bicycle (drive.h move_dynamics) for mpc.pytorch.

Adapted from mpc-lab/common.py. Differences from that file: slip angle is
atan (not tanh), steering is a state with its rate as the control, steering
is clipped at STEERING_LIMIT_RAD, the wheelbase is per agent (0.6 * length)
and the speed clip is MAX_SPEED_MPS.

State x = (px, py, heading, speed, steering),
control u = (accel, steering_rate).
Step order matches drive.h: steering first, then speed, then the pose from
the new speed and steering but the old heading. Heading is not wrapped
(drive.h wraps it with normalize_heading).
"""

import torch
from mpc import mpc
from mpc.lqr_step import LQRStep
from mpc import util
from mpc.mpc import GradMethods, LinDx, QuadCost
from torch.autograd import Variable

# Mirrored from drive.h
DT_SECONDS = 0.1
MAX_SPEED_MPS = 40.0
STEERING_LIMIT_RAD = 0.667
MAX_STEERING_RATE_RADPS = 0.6
MAX_ACCEL_MPS2 = 4.0
REAR_AXLE_RATIO = 0.5
WHEELBASE_PER_LENGTH = 0.6
ACCELERATION_VALUES = (-4.0, -2.667, -1.333, -0.0, 1.333, 2.667, 4.0)
STEERING_VALUES = (
    -0.667, -0.500, -0.333, -0.167, 0.000, 0.167, 0.333, 0.500, 0.667
)

# Ego observation layout (drive.h write_ego_obs)
EGO_SPEED_IDX = 0
EGO_LENGTH_IDX = 2
EGO_STEERING_IDX = 3
OBS_NORM_VEH_LENGTH_M = 15.0

NX, NU = 5, 2
PX, PY, HEADING, SPEED, STEERING = range(NX)
ACCEL, STEERING_RATE = range(NU)
U_LO = torch.tensor([-MAX_ACCEL_MPS2, -MAX_STEERING_RATE_RADPS])
U_HI = torch.tensor([MAX_ACCEL_MPS2, MAX_STEERING_RATE_RADPS])
SOLVER_EPS = 1e-4


class Bicycle(torch.nn.Module):
    """PufferDrive classic kinematic bicycle with analytic Jacobians."""

    def __init__(
            self,
            wheelbase_meters: torch.Tensor,
            dt_seconds: float = DT_SECONDS,
        ) -> None:
        """
        Store the per-agent wheelbase and the step length.

        Args:
            wheelbase_meters (torch.Tensor): (B,) wheelbase per agent [m].
            dt_seconds (float): step length [s].
        """
        super().__init__()
        self.wheelbase_meters = wheelbase_meters
        self.dt_seconds = dt_seconds

    def _tiled_wheelbase(
            self,
            row_count: int,
        ) -> torch.Tensor:
        """
        Wheelbase per batch row. mpc.pytorch calls the dynamics on (B,)
        batches during rollouts and on time-major flattened ((T-1)*B,)
        batches during linearization.

        Args:
            row_count (int): number of rows in the batch, a multiple of B.

        Returns:
            torch.Tensor: (row_count,) wheelbase per row [m].
        """
        agent_count = self.wheelbase_meters.shape[0]
        assert row_count % agent_count == 0, (
            "batch rows must be a multiple of the agent count"
        )
        return self.wheelbase_meters.repeat(row_count // agent_count)

    def forward(
            self,
            x: torch.Tensor,
            u: torch.Tensor,
        ) -> torch.Tensor:
        """
        One classic step, in drive.h's order.

        Args:
            x (torch.Tensor): (N,5) state
                (px, py, heading, speed, steering).
            u (torch.Tensor): (N,2) control
                (accel [m/s^2], steering_rate [rad/s]).

        Returns:
            torch.Tensor: (N,5) next state.
        """
        px, py, heading, speed, steering = x.unbind(-1)
        accel, steering_rate = u.unbind(-1)
        dt = self.dt_seconds
        wheelbase = self._tiled_wheelbase(x.shape[0])

        # drive.h updates steering and speed first; the pose uses the new ones
        next_steering = (steering + steering_rate * dt).clamp(
            -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD
        )
        next_speed = (speed + accel * dt).clamp(-MAX_SPEED_MPS, MAX_SPEED_MPS)
        tan_steering = torch.tan(next_steering)
        beta = torch.atan(REAR_AXLE_RATIO * tan_steering)  # slip angle
        yaw_rate = next_speed * torch.cos(beta) * tan_steering / wheelbase
        # Position moves along the old heading plus slip; heading changes after
        course = heading + beta
        return torch.stack(
            [
                px + next_speed * torch.cos(course) * dt,
                py + next_speed * torch.sin(course) * dt,
                heading + yaw_rate * dt,
                next_speed,
                next_steering,
            ],
            -1,
        )

    def grad_input(
            self,
            x: torch.Tensor,
            u: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic Jacobians of forward, used by mpc.pytorch with
        GradMethods.ANALYTIC.

        Args:
            x (torch.Tensor): (N,5) state
                (px, py, heading, speed, steering).
            u (torch.Tensor): (N,2) control (accel, steering_rate).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A = df/dx (N,5,5) and
                B = df/du (N,5,2).
        """
        px, py, heading, speed, steering = x.unbind(-1)
        accel, steering_rate = u.unbind(-1)
        dt = self.dt_seconds
        wheelbase = self._tiled_wheelbase(x.shape[0])

        # Each clip gives a 0/1 mask: a clipped value ignores its inputs
        raw_steering = steering + steering_rate * dt
        steering_mask = (raw_steering.abs() < STEERING_LIMIT_RAD).to(x.dtype)
        next_steering = raw_steering.clamp(
            -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD
        )
        raw_speed = speed + accel * dt
        speed_mask = (raw_speed.abs() < MAX_SPEED_MPS).to(x.dtype)
        next_speed = raw_speed.clamp(-MAX_SPEED_MPS, MAX_SPEED_MPS)

        tan_steering = torch.tan(next_steering)
        sec2_steering = 1 + tan_steering**2
        beta = torch.atan(REAR_AXLE_RATIO * tan_steering)
        # d(atan(r tan d))/dd = r sec^2 d / (1 + r^2 tan^2 d)
        dbeta = (
            REAR_AXLE_RATIO * sec2_steering
            / (1 + (REAR_AXLE_RATIO * tan_steering) ** 2)
        )
        cos_course = torch.cos(heading + beta)
        sin_course = torch.sin(heading + beta)
        cos_beta, sin_beta = torch.cos(beta), torch.sin(beta)

        # Partials w.r.t. the input speed and steering (through the clips)
        dpx_dspeed = speed_mask * cos_course * dt
        dpy_dspeed = speed_mask * sin_course * dt
        dheading_dspeed = (
            speed_mask * cos_beta * tan_steering / wheelbase * dt
        )
        dpx_dsteering = steering_mask * -next_speed * sin_course * dbeta * dt
        dpy_dsteering = steering_mask * next_speed * cos_course * dbeta * dt
        dheading_dsteering = (
            steering_mask * next_speed / wheelbase * dt
            * (cos_beta * sec2_steering - sin_beta * dbeta * tan_steering)
        )

        row_count = x.shape[0]
        A = torch.eye(NX, dtype=x.dtype, device=x.device).repeat(
            row_count, 1, 1
        )
        A[:, PX, HEADING] = -next_speed * sin_course * dt
        A[:, PY, HEADING] = next_speed * cos_course * dt
        A[:, PX, SPEED] = dpx_dspeed
        A[:, PY, SPEED] = dpy_dspeed
        A[:, HEADING, SPEED] = dheading_dspeed
        A[:, SPEED, SPEED] = speed_mask
        A[:, PX, STEERING] = dpx_dsteering
        A[:, PY, STEERING] = dpy_dsteering
        A[:, HEADING, STEERING] = dheading_dsteering
        A[:, STEERING, STEERING] = steering_mask

        # Controls enter only through speed and steering, each times dt
        B = torch.zeros(row_count, NX, NU, dtype=x.dtype, device=x.device)
        B[:, :, ACCEL] = A[:, :, SPEED] * dt
        B[:, :, STEERING_RATE] = A[:, :, STEERING] * dt
        return A, B


def wheelbase_from_length(
        length_meters: torch.Tensor,
    ) -> torch.Tensor:
    """
    Wheelbase as drive.h sets it (0.6 * vehicle length).

    Args:
        length_meters (torch.Tensor): (B,) vehicle length [m].

    Returns:
        torch.Tensor: (B,) wheelbase [m].
    """
    return WHEELBASE_PER_LENGTH * length_meters


def ego_state_from_obs(
        obs: torch.Tensor,
        obs_norm_veh_length_m: float = OBS_NORM_VEH_LENGTH_M,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ego-frame MPC start state and wheelbase, read from the ego block of the
    observation.

    Args:
        obs (torch.Tensor): (B,obs_dim) observations; the ego block starts
            at index 0.
        obs_norm_veh_length_m (float): length normalizer the env was built
            with [m].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x0 (B,5) =
            (0, 0, 0, speed, steering) and wheelbase (B,) [m].
    """
    speed = obs[:, EGO_SPEED_IDX] * MAX_SPEED_MPS
    steering = obs[:, EGO_STEERING_IDX] * STEERING_LIMIT_RAD
    length_meters = obs[:, EGO_LENGTH_IDX] * obs_norm_veh_length_m
    zero = torch.zeros_like(speed)
    x0 = torch.stack([zero, zero, zero, speed, steering], -1)
    return x0, wheelbase_from_length(length_meters)


def control_from_discrete(
        action_idx: torch.Tensor,
        steering_rad: torch.Tensor,
        dt_seconds: float = DT_SECONDS,
    ) -> torch.Tensor:
    """
    Teacher label from a discrete action: the acceleration, and drive.h's
    steering change as a rate. Executing the label reproduces the discrete
    step.

    Args:
        action_idx (torch.Tensor): (B,) discrete action index,
            accel_idx * 9 + steer_idx.
        steering_rad (torch.Tensor): (B,) current steering angle [rad].
        dt_seconds (float): step length [s].

    Returns:
        torch.Tensor: (B,2) control (accel [m/s^2], steering_rate [rad/s]).
    """
    table_kwargs = dict(dtype=steering_rad.dtype, device=steering_rad.device)
    accel_table = torch.tensor(ACCELERATION_VALUES, **table_kwargs)
    steering_table = torch.tensor(STEERING_VALUES, **table_kwargs)
    accel = accel_table[action_idx // len(STEERING_VALUES)]
    target_steering = steering_table[action_idx % len(STEERING_VALUES)]
    # The rate clip is drive.h's steering rate limit
    steering_rate = ((target_steering - steering_rad) / dt_seconds).clamp(
        -MAX_STEERING_RATE_RADPS, MAX_STEERING_RATE_RADPS
    )
    return torch.stack([accel, steering_rate], -1)


def env_action_from_control(
        u: torch.Tensor,
        steering_rad: torch.Tensor,
        dt_seconds: float = DT_SECONDS,
    ) -> torch.Tensor:
    """
    Continuous env action that makes PD execute control u.

    Args:
        u (torch.Tensor): (B,2) control
            (accel [m/s^2], steering_rate [rad/s]).
        steering_rad (torch.Tensor): (B,) current steering angle [rad].
        dt_seconds (float): step length [s].

    Returns:
        torch.Tensor: (B,2) action in [-1,1]^2
            (accel / 4, target steering / 0.667).
    """
    # The env takes a steering target; clipping it matches drive.h's limit
    target_steering = (steering_rad + u[:, STEERING_RATE] * dt_seconds).clamp(
        -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD
    )
    return torch.stack(
        [u[:, ACCEL] / MAX_ACCEL_MPS2, target_steering / STEERING_LIMIT_RAD],
        -1,
    )


def lqr_step_releasing_closures(
        *args,
        **kwargs,
    ):
    """
    mpc.pytorch's LQRStep with a GPU memory leak fixed. LQRStep defines a
    new autograd Function class per call, and those classes are never
    freed once a graph went through them; their closures pin that solve's
    cost (QuadCost), dynamics and trajectories (~1 MiB per 1024 samples,
    so ~40 MB per training iteration at batch 4096). After the
    differentiated step's backward, nothing reads its closures, so they
    are emptied. Gradients are unchanged (bit-identical in the test).
    The ctx also keeps current_x/current_u, which the final no-op step
    returns as its own outputs; output and grad_fn then form a cycle
    Python's gc cannot free (~0.14 MiB per step per 1024 samples), so
    the ctx drops them after backward too. Assumes one backward per
    graph (no retain_graph); solves without grad build no graph and are
    unaffected.

    Args:
        *args: LQRStep's positional arguments.
        **kwargs: LQRStep's keyword arguments.

    Returns:
        callable: LQRStepFn.apply of the patched function class.
    """
    apply = LQRStep(*args, **kwargs)
    function_class = apply.__self__
    original_forward = function_class.forward
    original_backward = function_class.backward

    def release_closures() -> None:
        """
        Drop the references the per-call class's functions close over.
        """
        for function in (original_forward, original_backward):
            for cell in function.__closure__ or ():
                cell.cell_contents = None

    def backward(ctx, *output_grads):
        """
        Original backward, then release the closures and the ctx's
        trajectories.

        Args:
            ctx: autograd context.
            *output_grads: gradients of LQRStepFn's outputs.

        Returns:
            tuple: gradients of LQRStepFn's inputs.
        """
        input_grads = original_backward(ctx, *output_grads)
        release_closures()
        ctx.current_x = ctx.current_u = None
        return input_grads

    function_class.backward = staticmethod(backward)
    return apply


# MPC.forward looks LQRStep up in mpc.mpc's namespace on every solve
mpc.LQRStep = lqr_step_releasing_closures


def mpc_forward_vectorized(
        self: mpc.MPC,
        x_init: torch.Tensor,
        cost: QuadCost,
        dx: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mpc.pytorch's MPC.forward with the best-iterate bookkeeping on tensors.

    Upstream keeps the best iterate as B one-column tensors and compares
    costs with a Python loop over the batch: B host syncs and ~16 kernels
    per sample. Here the best iterate stays a (T,B,.) tensor updated with
    torch.where, and the convergence tests are single reductions. Same
    results as upstream. Only QuadCost costs and LinDx or module dynamics
    are supported, and verbose logging is dropped.

    Args:
        self (mpc.MPC): the solver whose settings are used.
        x_init (torch.Tensor): (B,n_state) initial state.
        cost (QuadCost): C (T,B,n_tau,n_tau) and c (T,B,n_tau).
        dx (torch.nn.Module): dynamics module.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x (T,B,n_state),
            u (T,B,n_ctrl) and the best per-sample costs (B,).
    """
    assert isinstance(cost, QuadCost)
    n_batch = self.n_batch
    assert n_batch is not None
    assert x_init.ndimension() == 2 and x_init.size(0) == n_batch

    if self.u_init is None:
        u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x_init.data)
    else:
        u = self.u_init
        if u.ndimension() == 2:
            u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
    u = u.type_as(x_init.data)

    best_x = best_u = best_costs = best_du_norm = None
    n_not_improved = 0
    for _ in range(self.lqr_iter):
        u = Variable(util.detach_maybe(u), requires_grad=True)
        x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)
        if isinstance(dx, LinDx):
            F, f = dx.F, dx.f
        else:
            F, f = self.linearize_dynamics(
                x, util.detach_maybe(u), dx, diff=False)
        C, c = cost.C, cost.c

        x, u, _, costs, full_du_norm, _ = self.solve_lqr_subproblem(
            x_init, C, c, F, f, cost, dx, x, u)
        n_not_improved += 1

        if best_x is None:
            best_x, best_u = x, u
            best_costs, best_du_norm = costs, full_du_norm
        else:
            improved = costs <= best_costs + self.best_cost_eps
            keep = improved[None, :, None]
            best_x = torch.where(keep, x, best_x)
            best_u = torch.where(keep, u, best_u)
            best_costs = torch.where(improved, costs, best_costs)
            best_du_norm = torch.where(improved, full_du_norm, best_du_norm)
            if bool(improved.any()):
                n_not_improved = 0

        # One reduction and one host sync instead of one per sample
        if bool(full_du_norm.max() < self.eps) or (
                n_not_improved > self.not_improved_lim):
            break

    x, u = best_x, best_u
    if isinstance(dx, LinDx):
        F, f = dx.F, dx.f
    else:
        F, f = self.linearize_dynamics(x, u, dx, diff=True)
    C, c = cost.C, cost.c

    x, u = self.solve_lqr_subproblem(
        x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)

    if self.detach_unconverged and bool(best_du_norm.max() > self.eps):
        if self.exit_unconverged:
            assert False
        # Zero gradient for samples that did not reach a fixed point
        converged = best_du_norm < self.eps
        x_mask = converged[None, :, None].expand_as(x).type_as(x.data)
        u_mask = converged[None, :, None].expand_as(u).type_as(u.data)
        x = x * x_mask + x.clone().detach() * (1. - x_mask)
        u = u * u_mask + u.clone().detach() * (1. - u_mask)

    return x, u, best_costs


# Kept so tests can compare the patch against upstream
upstream_mpc_forward = mpc.MPC.forward
mpc.MPC.forward = mpc_forward_vectorized


def solve(
        x0: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        dyn: Bicycle,
        T: int,
        u_init: torch.Tensor | None = None,
        iters: int = 5,
        method: int = GradMethods.ANALYTIC,
        detach_unconverged: bool = False,
        u_lower: torch.Tensor = U_LO,
        u_upper: torch.Tensor = U_HI,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Box-constrained iLQR over horizon T with mpc.pytorch.

    Args:
        x0 (torch.Tensor): (B,5) initial state.
        Q (torch.Tensor): (T,B,7,7) quadratic cost on tau = [x; u].
        p (torch.Tensor): (T,B,7) linear cost on tau.
        dyn (Bicycle): dynamics module.
        T (int): horizon in steps.
        u_init (torch.Tensor | None): (T,B,2) warm-start controls, or None
            for zeros.
        iters (int): maximum iLQR iterations.
        method (int): mpc.pytorch GradMethods value used to linearize the
            dynamics.
        detach_unconverged (bool): if True, unconverged batch elements get
            zero gradient.
        u_lower (torch.Tensor): (2,) lower control bound; defaults to
            PD's box.
        u_upper (torch.Tensor): (2,) upper control bound; defaults to
            PD's box.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x (T,B,5) states and
            u (T,B,2) controls.
    """
    batch_size = x0.shape[0]
    lo = u_lower.to(x0).expand(T, batch_size, NU)
    hi = u_upper.to(x0).expand(T, batch_size, NU)
    x, u, _ = mpc.MPC(
        NX,
        NU,
        T,
        u_lower=lo,
        u_upper=hi,
        u_init=u_init,
        lqr_iter=iters,
        grad_method=method,
        exit_unconverged=False,
        detach_unconverged=detach_unconverged,
        verbose=-1,
        n_batch=batch_size,
        eps=SOLVER_EPS,
    )(x0, QuadCost(Q, p), dyn)
    return x, u


def target_cost(
        q_diag: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-step cost 0.5 * q * (tau - target)^2 as mpc.pytorch QuadCost terms.

    Args:
        q_diag (torch.Tensor): (T,B,7) diagonal weights on tau = [x; u].
        target (torch.Tensor): (T,B,7) target for tau.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Q (T,B,7,7) and p (T,B,7).
    """
    # mpc.pytorch minimizes 0.5 tau'Q tau + p'tau, so p = -Q * target puts
    # the minimum at target
    return torch.diag_embed(q_diag), -q_diag * target
