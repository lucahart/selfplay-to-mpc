"""WarpMPC formulation of the MPC student's problem (warpmpc_solver.py).

Needs the WarpMPC venv (JAX, Warp, CasADi) and a GPU; skipped otherwise.
Run with XLA_PYTHON_CLIENT_PREALLOCATE=false so JAX leaves torch room.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

pytest.importorskip("warpmpc")
ca = pytest.importorskip("casadi")
jnp = pytest.importorskip("jax.numpy")

from pufferlib.distill.bicycle import (  # noqa: E402
    MAX_SPEED_MPS,
    NU,
    NX,
    STEERING_LIMIT_RAD,
    Bicycle,
    ego_state_from_obs,
    solve,
    target_cost,
)
from pufferlib.distill.constraints import (  # noqa: E402
    CONSTRAINT_ELLIPSE,
    CONSTRAINT_PLANE,
    CONSTRAINT_TUBE,
    PLANE_DUMMY_OFFSET_M,
    PLANE_PARAM_COUNT,
    constraint_param_count,
    dummy_constraints,
)
from pufferlib.distill.obstacles import (  # noqa: E402
    OBSTACLE_PARAM_COUNT,
    dummy_obstacles,
    partner_obstacles,
)
from pufferlib.distill.warpmpc_solver import (  # noqa: E402
    FIRST_PARAM_COUNT,
    MIDDLE_PARAM_COUNT,
    MIN_PADDED_ROW_COUNT,
    PAD_WHEELBASE_M,
    TAU_DIM,
    CompiledBicycleSolver,
    bicycle_step_casadi,
    tube_terms,
    cached_solver,
    node_dim,
    node_slacks,
    pack_params,
    padded_row_count,
    param_count,
    rollout_warm_start,
    solve_cost_warpmpc,
    z_dim,
)

ROW_COUNT = 128
STUDENT_BOUND = np.array([4.0, 0.75], dtype=np.float32)
# Converged mpc.pytorch reference
REFERENCE_ILQR_ITERATIONS = 20
# |u0 difference| / box after 3 SQP x 50 ADMM steps (measured 4e-4)
CONTROL_TOLERANCE_OF_BOUND = 2e-3
X0_TOLERANCE = 1e-4

from conftest import make_small_env, needs_teacher  # noqa: E402
from pufferlib.distill.buffer import LabelBuffer  # noqa: E402
from pufferlib.distill.loop import (  # noqa: E402
    DistillConfig,
    DistillEvaluator,
    distill,
    distillation_loss,
)
from pufferlib.distill.policy import ContinuousPolicy  # noqa: E402
from pufferlib.distill.students import vehicle_rows  # noqa: E402
from pufferlib.ocean.benchmark.evaluators.base import (  # noqa: E402
    Evaluator,
)
from pufferlib.distill.overlay import (  # noqa: E402
    NODE_ELLIPSES_START,
    OVERLAY_ELLIPSE_COUNT_IDX,
    OVERLAY_HEADER,
    OVERLAY_NODE_FLOATS,
)
from pufferlib.distill.students import (  # noqa: E402
    COST_OUTPUTS_PER_STEP,
)
from test_students import (  # noqa: E402
    OBS_SEED,
    fresh_mpc_student,
    labels_for,
)
from test_teacher import rollout_obs  # noqa: E402
from pufferlib.distill.tube import (  # noqa: E402
    TUBE_ANCHOR_ROUTE,
    TUBE_NORMALS,
    reference_positions,
    shifted_reference,
    straight_reference,
    tube_planes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="WarpMPC's kernels need a GPU"
)


def random_problem(
        horizon_steps: int,
        row_count: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Student-like cost terms: zero p_x weight, speed targets near the
    current speed, lateral targets within half a meter.

    Args:
        horizon_steps (int): T.
        row_count (int): B.
        seed (int): generator seed.

    Returns:
        tuple: x0 (B,5), wheelbase (B,) [m], q_diag (T,B,7),
            target (T,B,7).
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (horizon_steps, row_count)
    x0 = torch.zeros(row_count, NX)
    x0[:, 3] = torch.rand(row_count, generator=generator) * 15.0
    x0[:, 4] = (torch.rand(row_count, generator=generator) - 0.5) * 0.4
    wheelbase = 2.5 + torch.rand(row_count, generator=generator)
    q_diag = torch.rand(*shape, TAU_DIM, generator=generator) * 2.0 + 0.5
    q_diag[..., 0] = 0.0
    target = torch.zeros(*shape, TAU_DIM)
    target[..., 1] = torch.rand(*shape, generator=generator) - 0.5
    speed_offset = (torch.rand(*shape, generator=generator) - 0.5) * 2.0
    target[..., 3] = x0[:, 3] + speed_offset
    return x0, wheelbase, q_diag, target


@pytest.fixture(scope="module", params=[2, 3, 5])
def compiled(request) -> CompiledBicycleSolver:
    """
    One compiled solver per horizon, shared by the tests of the module.

    Args:
        request: pytest request carrying the horizon.

    Returns:
        CompiledBicycleSolver: 3 SQP x 50 ADMM steps, float32.
    """
    return CompiledBicycleSolver(
        request.param, ROW_COUNT, STUDENT_BOUND, 3, 50
    )


def test_casadi_step_matches_torch_bicycle():
    generator = torch.Generator().manual_seed(0)
    row_count = 512
    x = torch.randn(row_count, NX, generator=generator, dtype=torch.float64)
    # Push some rows past the speed and steering clamps
    x[:, 3] *= MAX_SPEED_MPS / 2.0
    x[:, 4] *= STEERING_LIMIT_RAD
    u = torch.randn(row_count, NU, generator=generator, dtype=torch.float64)
    u *= torch.tensor([20.0, 3.0], dtype=torch.float64)
    wheelbase = 2.0 + torch.rand(
        row_count, generator=generator, dtype=torch.float64
    )
    expected = Bicycle(wheelbase)(x, u)

    x_sym, u_sym = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
    wheelbase_sym = ca.SX.sym("wheelbase")
    step = ca.Function(
        "step",
        [x_sym, u_sym, wheelbase_sym],
        [bicycle_step_casadi(x_sym, u_sym, wheelbase_sym)],
    ).map(row_count)
    actual = np.asarray(step(x.T.numpy(), u.T.numpy(), wheelbase.numpy()))
    np.testing.assert_allclose(actual.T, expected.numpy(), atol=1e-12)


def test_param_layout_matches_plan(compiled):
    plan = compiled.solver.plan
    horizon_steps = compiled.horizon_steps
    assert plan.n_parameters == param_count(horizon_steps)
    np.testing.assert_array_equal(
        plan.z_offsets[:horizon_steps],
        np.arange(horizon_steps) * TAU_DIM,
    )
    middle_offsets = (
        FIRST_PARAM_COUNT
        + np.arange(horizon_steps - 2) * MIDDLE_PARAM_COUNT
    )
    expected_offsets = [0, *middle_offsets, param_count(horizon_steps)
                        - 2 * TAU_DIM]
    np.testing.assert_array_equal(
        plan.param_offsets[:horizon_steps], expected_offsets
    )


def test_pack_params_places_every_block():
    horizon_steps = 4
    x0, wheelbase, q_diag, target = random_problem(horizon_steps, 3, seed=1)
    params = pack_params(x0, wheelbase, q_diag, target)
    assert params.shape == (3, param_count(horizon_steps))
    torch.testing.assert_close(params[:, :TAU_DIM], q_diag[0])
    torch.testing.assert_close(params[:, TAU_DIM:2 * TAU_DIM], target[0])
    torch.testing.assert_close(params[:, 2 * TAU_DIM:2 * TAU_DIM + NX], x0)
    torch.testing.assert_close(params[:, FIRST_PARAM_COUNT - 1], wheelbase)
    for step_idx in range(1, horizon_steps - 1):
        start = FIRST_PARAM_COUNT + (step_idx - 1) * MIDDLE_PARAM_COUNT
        block = params[:, start:start + MIDDLE_PARAM_COUNT]
        torch.testing.assert_close(block[:, :TAU_DIM], q_diag[step_idx])
        torch.testing.assert_close(block[:, TAU_DIM:-1], target[step_idx])
        torch.testing.assert_close(block[:, -1], wheelbase)
    torch.testing.assert_close(
        params[:, -2 * TAU_DIM:],
        torch.cat([q_diag[-1], target[-1]], dim=1),
    )


def test_rollout_warm_start_follows_dynamics():
    horizon_steps = 4
    x0, wheelbase, _, _ = random_problem(horizon_steps, 16, seed=2)
    u_init = torch.randn(horizon_steps, 16, NU)
    nodes = rollout_warm_start(x0, wheelbase, u_init).view(
        16, horizon_steps, TAU_DIM
    )
    torch.testing.assert_close(nodes[:, 0, :NX], x0)
    torch.testing.assert_close(nodes[:, :, NX:], u_init.transpose(0, 1))
    dynamics = Bicycle(wheelbase)
    for step_idx in range(horizon_steps - 1):
        torch.testing.assert_close(
            nodes[:, step_idx + 1, :NX],
            dynamics(nodes[:, step_idx, :NX], u_init[step_idx]),
        )


def test_solve_matches_converged_ilqr(compiled):
    horizon_steps = compiled.horizon_steps
    x0, wheelbase, q_diag, target = random_problem(
        horizon_steps, ROW_COUNT, seed=3
    )
    params = pack_params(x0, wheelbase, q_diag, target)
    z_init = rollout_warm_start(
        x0, wheelbase, torch.zeros(horizon_steps, ROW_COUNT, NU)
    )
    z = compiled.solve(jnp.asarray(z_init.numpy()),
                       jnp.asarray(params.numpy()))
    nodes = torch.tensor(np.asarray(z)).view(
        ROW_COUNT, horizon_steps, TAU_DIM
    )
    bound = torch.tensor(STUDENT_BOUND)
    Q, p = target_cost(q_diag, target)
    with torch.no_grad():
        _, reference_u = solve(
            x0, Q, p, Bicycle(wheelbase), horizon_steps,
            iters=REFERENCE_ILQR_ITERATIONS, u_lower=-bound, u_upper=bound,
        )
    first_control_error = (nodes[:, 0, NX:] - reference_u[0]).abs() / bound
    assert float(first_control_error.max()) < CONTROL_TOLERANCE_OF_BOUND
    assert float((nodes[:, 0, :NX] - x0).abs().max()) < X0_TOLERANCE
    assert bool((nodes[:, :, NX:].abs() <= bound * 1.001).all())


def test_rejects_horizon_without_terminal_node():
    with pytest.raises(ValueError, match="horizon_steps"):
        CompiledBicycleSolver(1, ROW_COUNT, STUDENT_BOUND, 3, 50)


def test_gradient_matches_ilqr_at_many_sqp_steps():
    # Near the optimum the line search accepts steps below 1; the gradient
    # must still be the full QP step's (it fell to 0.54 of iLQR's norm)
    import jax

    horizon_steps, sqp_iterations = 3, 5
    x0, wheelbase, q_diag, target = random_problem(
        horizon_steps, ROW_COUNT, seed=4
    )
    labels = torch.zeros(ROW_COUNT, NU)
    bound = torch.tensor(STUDENT_BOUND)
    solver = CompiledBicycleSolver(
        horizon_steps, ROW_COUNT, STUDENT_BOUND, sqp_iterations, 50
    )
    z_init = jnp.asarray(rollout_warm_start(
        x0, wheelbase, torch.zeros(horizon_steps, ROW_COUNT, NU)
    ).numpy())
    target_leaf = target.clone().requires_grad_(True)
    params = pack_params(x0, wheelbase, q_diag, target_leaf)

    def loss(params_in):
        """Summed squared first control, scaled by the box."""
        first_control = solver.solve(z_init, params_in)[:, NX:TAU_DIM]
        return jnp.sum((first_control / jnp.asarray(STUDENT_BOUND)) ** 2)

    params_grad = jax.grad(loss)(jnp.asarray(params.detach().numpy()))
    params.backward(torch.from_numpy(np.asarray(params_grad)))

    reference_target = target.clone().requires_grad_(True)
    Q, p = target_cost(q_diag, reference_target)
    _, reference_u = solve(
        x0, Q, p, Bicycle(wheelbase), horizon_steps,
        iters=REFERENCE_ILQR_ITERATIONS, u_lower=-bound, u_upper=bound,
    )
    (((reference_u[0] - labels) / bound) ** 2).sum().backward()
    warp_grad = target_leaf.grad.flatten().double()
    reference_grad = reference_target.grad.flatten().double()
    cosine = torch.nn.functional.cosine_similarity(
        warp_grad, reference_grad, dim=0
    )
    norm_ratio = warp_grad.norm() / reference_grad.norm()
    assert float(cosine) > 0.99
    assert abs(float(norm_ratio) - 1.0) < 0.02


def bridge_inputs(
        row_count: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    random_problem on CUDA, q_diag and target as gradient leaves.

    Args:
        row_count (int): B.
        seed (int): generator seed.

    Returns:
        tuple: x0 (B,5), wheelbase (B,), q_diag (3,B,7), target (3,B,7).
    """
    x0, wheelbase, q_diag, target = random_problem(3, row_count, seed)
    return (
        x0.cuda(),
        wheelbase.cuda(),
        q_diag.cuda().requires_grad_(True),
        target.cuda().requires_grad_(True),
    )


def test_bridge_gradient_equals_jax_gradient():
    x0, wheelbase, q_diag, target = bridge_inputs(ROW_COUNT, seed=5)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    controls = solve_cost_warpmpc(
        x0, wheelbase, q_diag, target, None, bound, 3, 25
    )
    ((controls[0] / bound) ** 2).sum().backward()

    import jax

    padded = padded_row_count(ROW_COUNT)
    solver = cached_solver(3, padded, tuple(STUDENT_BOUND.tolist()), 3, 25,
                           np.float32)
    pad_count = padded - ROW_COUNT
    params = pack_params(
        torch.cat([x0, torch.zeros(pad_count, NX, device="cuda")]),
        torch.cat([wheelbase,
                   torch.full((pad_count,), PAD_WHEELBASE_M, device="cuda")]),
        torch.cat([q_diag.detach(),
                   torch.ones(3, pad_count, TAU_DIM, device="cuda")], 1),
        torch.cat([target.detach(),
                   torch.zeros(3, pad_count, TAU_DIM, device="cuda")], 1),
    )
    z_init = rollout_warm_start(
        params[:, 2 * TAU_DIM:2 * TAU_DIM + NX],
        params[:, FIRST_PARAM_COUNT - 1],
        torch.zeros(3, padded, NU, device="cuda"),
    )
    z_init_jax = jnp.asarray(z_init.cpu().numpy())
    bound_jax = jnp.asarray(STUDENT_BOUND)

    def loss(params_in):
        """Summed squared first control of the real rows."""
        first_control = solver.solve(z_init_jax, params_in)[:ROW_COUNT,
                                                            NX:TAU_DIM]
        return jnp.sum((first_control / bound_jax) ** 2)

    params_grad = np.asarray(jax.grad(loss)(
        jnp.asarray(params.cpu().numpy())
    ))
    expected_target_grad = params_grad[:ROW_COUNT, TAU_DIM:2 * TAU_DIM]
    np.testing.assert_allclose(
        target.grad[0].cpu().numpy(), expected_target_grad,
        rtol=1e-4, atol=1e-6,
    )


def test_padding_leaves_rows_unchanged():
    # Rows are independent: solving 100 rows (padded to 256) gives the
    # same controls as the same rows inside a 300-row batch (padded to 512)
    x0, wheelbase, q_diag, target = bridge_inputs(300, seed=6)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        all_rows = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25
        )
        first_rows = solve_cost_warpmpc(
            x0[:100], wheelbase[:100], q_diag[:, :100], target[:, :100],
            None, bound, 3, 25,
        )
    assert first_rows.shape == (3, 100, NU)
    torch.testing.assert_close(first_rows, all_rows[:, :100],
                               rtol=1e-4, atol=1e-5)


def test_repeated_backward_does_not_leak_device_memory():
    import jax

    x0, wheelbase, q_diag, target = bridge_inputs(ROW_COUNT, seed=7)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    counts = []
    for iteration in range(30):
        controls = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25
        )
        controls[0].sum().backward()
        del controls
        if iteration in (9, 29):
            counts.append(
                (len(jax.live_arrays()), torch.cuda.memory_allocated())
            )
    assert counts[1] == counts[0]


def test_rejects_cpu_tensors():
    x0, wheelbase, q_diag, target = random_problem(3, 8, seed=8)
    with pytest.raises(ValueError, match="CUDA"):
        solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None,
            torch.tensor(STUDENT_BOUND), 3, 25,
        )


def test_padded_row_count_is_a_power_of_two():
    assert padded_row_count(1) == MIN_PADDED_ROW_COUNT
    assert padded_row_count(256) == 256
    assert padded_row_count(257) == 512
    assert padded_row_count(1024) == 1024


@needs_teacher
def test_warpmpc_student_matches_mpc_pytorch_student(teacher, device):
    # Same weights, two solvers: first controls and the head's gradient
    # agree (step 2 measured cosine 0.995 on (q, target))
    obs = rollout_obs(teacher, 20, OBS_SEED)
    labels = labels_for(teacher, obs)
    head_grads, predictions = [], []
    for solver in ("mpc_pytorch", "warpmpc"):
        student = fresh_mpc_student(
            teacher, device, solver=solver, train_lqr_iterations=20
        )
        predicted = student.predict(obs)
        distillation_loss(predicted, labels).backward()
        predictions.append(predicted.detach())
        head_grads.append(student.head.weight.grad.flatten().double())
    bound = bound_on(device)
    error = ((predictions[1] - predictions[0]).abs() / bound).amax(-1)
    assert float(error.quantile(0.99)) < 0.01
    cosine = torch.nn.functional.cosine_similarity(*head_grads, dim=0)
    assert float(cosine) > 0.99


@needs_teacher
def test_warpmpc_student_acts_with_empty_slots(teacher, device):
    student = fresh_mpc_student(teacher, device, solver="warpmpc")
    obs = rollout_obs(teacher, 3, OBS_SEED)[:50].clone()
    obs[::5] = 0.0
    driven = torch.ones(50, dtype=torch.bool, device=device)
    for _ in range(2):
        controls = student.act(obs, driven)
    assert bool((controls[::5] == 0).all())
    assert bool(torch.isfinite(controls).all())
    assert bool((controls.abs() <= bound_on(device) * 1.001).all())


def bound_on(
        device: str,
    ) -> torch.Tensor:
    """
    Args:
        device (str): torch device.

    Returns:
        torch.Tensor: (2,) the student's box on device.
    """
    return torch.tensor(STUDENT_BOUND, device=device)


def test_controls_stay_in_box_on_extreme_costs():
    # Costs at the head's limits (weights up to 10, lateral targets 5 m,
    # speed targets 10 m/s away): with OSQP's default rho the ADMM left
    # controls up to 2.3 times outside the box and training diverged
    generator = torch.Generator().manual_seed(9)
    row_count, horizon_steps = 1024, 3
    shape = (horizon_steps, row_count)
    x0 = torch.zeros(row_count, NX)
    x0[:, 3] = torch.rand(row_count, generator=generator) * 10.0
    wheelbase = 2.5 + torch.rand(row_count, generator=generator)
    q_diag = torch.rand(*shape, TAU_DIM, generator=generator) * 10.0
    q_diag[..., 0] = 0.0
    q_diag[..., NX:] = q_diag[..., NX:].clamp_min(0.1)
    target = torch.zeros(*shape, TAU_DIM)
    target[..., 1] = (torch.rand(*shape, generator=generator) - 0.5) * 10.0
    target[..., 2] = (torch.rand(*shape, generator=generator) - 0.5)
    speed_offset = (torch.rand(*shape, generator=generator) - 0.5) * 20.0
    target[..., 3] = x0[:, 3] + speed_offset
    target[..., 4] = (torch.rand(*shape, generator=generator) - 0.5) * 1.3
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        controls = solve_cost_warpmpc(
            x0.cuda(), wheelbase.cuda(), q_diag.cuda(), target.cuda(),
            None, bound, 3, 25,
        )
    assert float((controls.abs() / bound).max()) < 1.005


def test_failing_rows_of_the_t10_run_stay_finite_and_in_box():
    # Rows saved from a T = 10 DAgger run that diverged:
    # dump1 diverged to 26 times the box, dump2 (standstill, state weights
    # ~1e-14) returned NaN, dump3 starts from a warm start far outside the
    # box. Needs ADMM_SIGMA, the warm-start clamp and the step rejection
    import os
    path = os.path.join(os.path.dirname(__file__),
                        "data_warpmpc_failing_rows.pt")
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    for name, rows in torch.load(path).items():
        u_init = rows.get("u_init")
        with torch.no_grad():
            controls = solve_cost_warpmpc(
                rows["x0"].cuda(), rows["wheelbase"].cuda(),
                rows["q"].cuda(), rows["target"].cuda(),
                None if u_init is None else u_init.cuda(), bound, 5, 25,
            )
        assert bool(torch.isfinite(controls).all()), name
        assert float((controls.abs() / bound).max()) < 1.01, name


def straight_problem(
        horizon_steps: int,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Driving straight at 10 m/s along p_y = 0, on CUDA.

    Args:
        horizon_steps (int): T.
        row_count (int): B.

    Returns:
        tuple: x0 (B,5), wheelbase (B,) [m], q_diag (T,B,7),
            target (T,B,7).
    """
    x0 = torch.zeros(row_count, NX, device="cuda")
    x0[:, 3] = 10.0
    wheelbase = torch.full((row_count,), 2.7, device="cuda")
    q_diag = torch.ones(horizon_steps, row_count, TAU_DIM, device="cuda")
    q_diag[..., 0] = 0.0
    target = torch.zeros(horizon_steps, row_count, TAU_DIM, device="cuda")
    target[..., 3] = 10.0
    return x0, wheelbase, q_diag, target


def blocking_obstacle(
        horizon_steps: int,
        row_count: int,
    ) -> torch.Tensor:
    """
    One standing car slightly left of the lane, 9 m ahead; slot 2 is a
    dummy.

    Args:
        horizon_steps (int): T.
        row_count (int): B.

    Returns:
        torch.Tensor: (T,B,2,6) obstacles.
    """
    obstacles = dummy_obstacles(
        horizon_steps, row_count, 2, torch.float32, torch.device("cuda")
    )
    obstacles[:, :, 0] = torch.tensor(
        [9.0, 0.5, 1.0, 0.0, 4.0, 2.0], device="cuda"
    )
    return obstacles


def planned_positions(
        x0: torch.Tensor,
        wheelbase: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
    """
    Positions of the plan's nodes.

    Args:
        x0 (torch.Tensor): (B,5) start state.
        wheelbase (torch.Tensor): (B,) wheelbase [m].
        controls (torch.Tensor): (T,B,2) plan.

    Returns:
        torch.Tensor: (B,T,2) positions [m].
    """
    horizon_steps = controls.shape[0]
    nodes = rollout_warm_start(x0, wheelbase, controls).view(
        x0.shape[0], horizon_steps, TAU_DIM
    )
    return nodes[..., :2]


@pytest.mark.parametrize("constraint_type", [
    CONSTRAINT_ELLIPSE, CONSTRAINT_PLANE,
])
def test_param_layout_with_constraints_matches_plan(constraint_type):
    horizon_steps, constraint_count = 4, 3
    solver = CompiledBicycleSolver(
        horizon_steps, ROW_COUNT, STUDENT_BOUND, 1, 10,
        constraint_type=constraint_type, constraint_count=constraint_count,
    )
    plan = solver.solver.plan
    block = constraint_count * constraint_param_count(constraint_type)
    assert plan.n_parameters == param_count(
        horizon_steps, constraint_type, constraint_count
    )
    # Node 0 is tau only; later nodes carry the slacks
    np.testing.assert_array_equal(
        plan.z_offsets[:horizon_steps],
        [0, *(TAU_DIM + np.arange(horizon_steps - 1)
              * node_dim(constraint_count))],
    )
    assert plan.n_variables == z_dim(horizon_steps, constraint_count)
    middle_offsets = (
        FIRST_PARAM_COUNT
        + np.arange(horizon_steps - 2) * (MIDDLE_PARAM_COUNT + block)
    )
    expected_offsets = [
        0, *middle_offsets,
        param_count(horizon_steps, constraint_type, constraint_count)
        - 2 * TAU_DIM - block,
    ]
    np.testing.assert_array_equal(
        plan.param_offsets[:horizon_steps], expected_offsets
    )


def test_pack_params_places_constraints_last_from_node_one():
    horizon_steps, row_count = 4, 3
    x0, wheelbase, q_diag, target = random_problem(
        horizon_steps, row_count, seed=11
    )
    constraints = torch.randn(
        horizon_steps, row_count, 2, OBSTACLE_PARAM_COUNT
    )
    params = pack_params(x0, wheelbase, q_diag, target, constraints)
    assert params.shape == (
        row_count, param_count(horizon_steps, CONSTRAINT_ELLIPSE, 2)
    )
    block = 2 * OBSTACLE_PARAM_COUNT
    node_ends = np.cumsum(
        [FIRST_PARAM_COUNT]
        + [MIDDLE_PARAM_COUNT + block] * (horizon_steps - 2)
        + [2 * TAU_DIM + block]
    )
    torch.testing.assert_close(
        params[:, FIRST_PARAM_COUNT - 1], wheelbase
    )
    for step_idx in range(1, horizon_steps):
        end = node_ends[step_idx]
        torch.testing.assert_close(
            params[:, end - block:end],
            constraints[step_idx].reshape(row_count, -1),
        )


@pytest.mark.parametrize("constraint_type, constraint_count", [
    (CONSTRAINT_ELLIPSE, 4),
    (CONSTRAINT_PLANE, 3),
])
def test_dummy_constraints_leave_the_plan_unchanged(
        constraint_type, constraint_count,
    ):
    horizon_steps = 10
    x0, wheelbase, q_diag, target = bridge_inputs(ROW_COUNT, seed=12)
    q_diag = q_diag.detach()[-1:].expand(horizon_steps, -1, -1).contiguous()
    target = target.detach()[-1:].expand(horizon_steps, -1, -1).contiguous()
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    dummies = dummy_constraints(
        constraint_type, horizon_steps, ROW_COUNT, constraint_count,
        torch.float32, torch.device("cuda"),
    )
    with torch.no_grad():
        free = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 5, 25
        )
        with_dummies = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 5, 25,
            constraint_type, dummies,
        )
    torch.testing.assert_close(with_dummies, free, rtol=1e-3, atol=2e-3)


def test_plan_passes_a_blocking_car():
    horizon_steps = 10
    x0, wheelbase, q_diag, target = straight_problem(horizon_steps, 256)
    obstacles = blocking_obstacle(horizon_steps, 256)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        free = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 10, 25
        )
        avoiding = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 10, 25,
            CONSTRAINT_ELLIPSE, obstacles,
        )
    car = obstacles[0, 0, 0]

    def ellipse_value(controls):
        """Smallest ellipse value of the standing car along the plan."""
        offset = planned_positions(x0, wheelbase, controls) - car[:2]
        return ((offset[..., 0] / car[4]) ** 2
                + (offset[..., 1] / car[5]) ** 2).amin(-1)

    assert float(ellipse_value(free).max()) < 1.0
    assert float(ellipse_value(avoiding).min()) > 0.98
    # The car sits left of the lane: the plan passes it on the right
    lateral = planned_positions(x0, wheelbase, avoiding)[..., 1]
    assert float(lateral.amin(-1).max()) < 0.0


def test_obstacle_gradient_is_finite_when_active():
    horizon_steps = 10
    x0, wheelbase, q_diag, target = straight_problem(horizon_steps, 256)
    obstacles = blocking_obstacle(horizon_steps, 256).requires_grad_(True)
    q_diag.requires_grad_(True)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    controls = solve_cost_warpmpc(
        x0, wheelbase, q_diag, target, None, bound, 10, 25,
        CONSTRAINT_ELLIPSE, obstacles,
    )
    (controls[0] / bound).pow(2).sum().backward()
    assert bool(torch.isfinite(obstacles.grad).all())
    assert bool(torch.isfinite(q_diag.grad).all())
    # The car's semi-axes move the first control; the dummy slot does not
    assert float(obstacles.grad[:, :, 0, 4:].abs().sum()) > 0.0
    assert float(obstacles.grad[:, :, 1].abs().max()) < 1e-6


def planes_on_cuda(
        horizon_steps: int,
        row_count: int,
        plane_values: list[tuple[float, float, float]],
    ) -> torch.Tensor:
    """
    The same planes for every row and step.

    Args:
        horizon_steps (int): T.
        row_count (int): B.
        plane_values (list): per plane (a_px, a_py, b).

    Returns:
        torch.Tensor: (T,B,H,3) planes on CUDA.
    """
    planes = torch.tensor(plane_values, device="cuda")
    return planes.expand(
        horizon_steps, row_count, len(plane_values), PLANE_PARAM_COUNT
    ).clone()


def left_pulling_problem(
        horizon_steps: int,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    straight_problem with a lateral target 2 m to the left.

    Args:
        horizon_steps (int): T.
        row_count (int): B.

    Returns:
        tuple: x0 (B,5), wheelbase (B,) [m], q_diag (T,B,7),
            target (T,B,7).
    """
    x0, wheelbase, q_diag, target = straight_problem(
        horizon_steps, row_count
    )
    target[..., 1] = LEFT_TARGET_M
    return x0, wheelbase, q_diag, target


# Plane tests: 10 m/s, T = 10, lateral target 2 m left, left bound 0.3 m
LEFT_TARGET_M = 2.0
LEFT_BOUND_M = 0.3
PLANE_SQP_ITERATIONS = 10
PLANE_TOLERANCE_M = 0.02
# Converged enough for finite differences (analytic 1.093 vs numeric
# 1.108 per row, flat in the step from 5 mm to 5 cm)
CONVERGED_SQP_ITERATIONS = 20
CONVERGED_ADMM_ITERATIONS = 50


@pytest.mark.parametrize("lateral_weight, lateral_target_m", [
    (1.0, LEFT_TARGET_M),
    # The head's strongest pull (STATE_WEIGHT_MAX, TARGET_LATERAL_RANGE_M):
    # a slack price below the plane's multiplier lets the plan through
    (10.0, 5.0),
])
def test_reachable_plane_holds(lateral_weight, lateral_target_m):
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = left_pulling_problem(
        horizon_steps, row_count
    )
    q_diag[..., 1] = lateral_weight
    target[..., 1] = lateral_target_m
    planes = planes_on_cuda(
        horizon_steps, row_count, [(0.0, 1.0, LEFT_BOUND_M)]
    )
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        free = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            PLANE_SQP_ITERATIONS, 25,
        )
        bounded = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            PLANE_SQP_ITERATIONS, 25, CONSTRAINT_PLANE, planes,
        )
    free_lateral = planned_positions(x0, wheelbase, free)[..., 1]
    bounded_lateral = planned_positions(x0, wheelbase, bounded)[..., 1]
    assert float(free_lateral.amax()) > LEFT_BOUND_M + 0.2
    assert float(bounded_lateral.amax()) < LEFT_BOUND_M + PLANE_TOLERANCE_M


def test_unreachable_plane_is_absorbed_by_slack():
    # p_x <= 0 from node 1 on at 10 m/s: no plan satisfies it; the slack
    # keeps the QP feasible and the plan brakes as hard as the box allows
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = straight_problem(
        horizon_steps, row_count
    )
    planes = planes_on_cuda(horizon_steps, row_count, [(1.0, 0.0, 0.0)])
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        controls = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            PLANE_SQP_ITERATIONS, 25, CONSTRAINT_PLANE, planes,
        )
    assert bool(torch.isfinite(controls).all())
    assert float((controls.abs() / bound).max()) < 1.01
    assert float(controls[0, :, 0].max()) < -0.9 * float(bound[0])
    # The same solve directly, to read the slacks
    solver = cached_solver(
        horizon_steps, row_count, tuple(STUDENT_BOUND.tolist()),
        PLANE_SQP_ITERATIONS, 25, np.float32, CONSTRAINT_PLANE, 1,
    )
    params = pack_params(x0, wheelbase, q_diag, target, planes)
    z_init = rollout_warm_start(
        x0, wheelbase, torch.zeros_like(controls), 1
    )
    z = solver.solve(
        jnp.asarray(z_init.cpu().numpy()), jnp.asarray(params.cpu().numpy())
    )
    slacks = node_slacks(torch.tensor(np.asarray(z)), horizon_steps, 1)
    assert float(slacks.min()) > 0.5


def test_plane_gradient_matches_finite_differences():
    # Offset of an active plane against central differences on converged
    # solves (at 10 x 25 the differences are noisy); an inactive plane
    # gets no gradient
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = left_pulling_problem(
        horizon_steps, row_count
    )
    planes = planes_on_cuda(
        horizon_steps, row_count,
        [(0.0, 1.0, LEFT_BOUND_M), (0.0, -1.0, 500.0)],
    )
    bound = torch.tensor(STUDENT_BOUND, device="cuda")

    def first_rate_sum(plane_params):
        """Summed first steering rate of the plan."""
        controls = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            CONVERGED_SQP_ITERATIONS, CONVERGED_ADMM_ITERATIONS,
            CONSTRAINT_PLANE, plane_params,
        )
        return controls[0, :, 1].sum()

    leaf = planes.clone().requires_grad_(True)
    first_rate_sum(leaf).backward()
    analytic = float(leaf.grad[:, :, 0, 2].sum())
    step_m = 0.02
    shift = torch.zeros_like(planes)
    shift[:, :, 0, 2] = step_m
    with torch.no_grad():
        numeric = float(
            first_rate_sum(planes + shift) - first_rate_sum(planes - shift)
        ) / (2.0 * step_m)
    assert abs(numeric) > 1e-3
    assert abs(analytic - numeric) < 0.05 * abs(numeric)
    assert float(leaf.grad[:, :, 1].abs().max()) < 1e-6


# Tube tests: ego 2 m wide, margin 0.2 m, 10 m/s, T = 10
TUBE_EGO_WIDTH_M = 2.0
TUBE_EGO_LENGTH_M = 4.6
TUBE_MARGIN_M = 0.2
TUBE_BODY_OFFSET_M = 0.5 * TUBE_EGO_WIDTH_M + TUBE_MARGIN_M
# Half-width of the test roads; the plan's centre must stay 1.3 m inside
TUBE_ROAD_HALF_WIDTH_M = 2.5
TUBE_SEGMENT_HALF_LENGTH_M = 1.0
# Radius of the curved road's centreline (a left bend)
TUBE_BEND_RADIUS_M = 25.0
# The tangent planes of the bend's outer edge leave the arc by up to
# (segment distance)^2 / 2R between nodes; nodes sit near the tangent
TUBE_TOLERANCE_M = 0.05
TUBE_SQP_ITERATIONS = 10


@pytest.mark.parametrize("ellipse_count", [0, 2])
def test_tube_layout_has_two_plane_slacks_before_the_ellipses(
        ellipse_count,
    ):
    horizon_steps = 4
    solver = CompiledBicycleSolver(
        horizon_steps, ROW_COUNT, STUDENT_BOUND, 1, 10,
        constraint_type=CONSTRAINT_TUBE, constraint_count=ellipse_count,
    )
    plan = solver.solver.plan
    slack_count = 2 + ellipse_count
    assert plan.n_parameters == param_count(
        horizon_steps, CONSTRAINT_TUBE, ellipse_count
    )
    assert plan.n_parameters == param_count(horizon_steps) + (
        horizon_steps - 1
    ) * (6 + OBSTACLE_PARAM_COUNT * ellipse_count)
    assert plan.n_variables == z_dim(horizon_steps, slack_count)


def straight_road_segments(
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Road edges at p_y = +-TUBE_ROAD_HALF_WIDTH_M from 10 m behind to 40 m
    ahead, in 2 m segments, on CUDA.

    Args:
        row_count (int): B.

    Returns:
        tuple: centres (B,S,2), directions (B,S,2), half-lengths (B,S),
            valid (B,S).
    """
    centre_x = torch.arange(-9.0, 40.0, 2.0 * TUBE_SEGMENT_HALF_LENGTH_M)
    centres = torch.cat([
        torch.stack([centre_x, torch.full_like(centre_x, side_y)], -1)
        for side_y in (TUBE_ROAD_HALF_WIDTH_M, -TUBE_ROAD_HALF_WIDTH_M)
    ])
    directions = torch.zeros_like(centres)
    directions[:, 0] = 1.0
    return road_on_cuda(centres, directions, row_count)


def bend_segments(
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Edges of a left bend whose centreline is the circle of radius
    TUBE_BEND_RADIUS_M around (0, R) through the origin, heading +x
    there; chords of 2 m from 10 m behind to 60 degrees of arc, on CUDA.

    Args:
        row_count (int): B.

    Returns:
        tuple: centres (B,S,2), directions (B,S,2), half-lengths (B,S),
            valid (B,S).
    """
    centres, directions = [], []
    for radius_m in (
        TUBE_BEND_RADIUS_M - TUBE_ROAD_HALF_WIDTH_M,
        TUBE_BEND_RADIUS_M + TUBE_ROAD_HALF_WIDTH_M,
    ):
        chord_rad = 2.0 * TUBE_SEGMENT_HALF_LENGTH_M / radius_m
        arc_rad = torch.arange(-10.0 / radius_m, math.pi / 3.0, chord_rad)
        middle_rad = arc_rad + 0.5 * chord_rad
        # Angle 0 is the point below the circle's centre
        centres.append(torch.stack([
            radius_m * torch.sin(middle_rad),
            TUBE_BEND_RADIUS_M - radius_m * torch.cos(middle_rad),
        ], -1))
        directions.append(
            torch.stack([torch.cos(middle_rad), torch.sin(middle_rad)], -1)
        )
    return road_on_cuda(
        torch.cat(centres), torch.cat(directions), row_count
    )


def road_on_cuda(
        centres: torch.Tensor,
        directions: torch.Tensor,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The same segments for every row.

    Args:
        centres (torch.Tensor): (S,2) segment centres [m].
        directions (torch.Tensor): (S,2) unit directions.
        row_count (int): B.

    Returns:
        tuple: centres (B,S,2), directions (B,S,2), half-lengths (B,S) of
            TUBE_SEGMENT_HALF_LENGTH_M, valid (B,S), on CUDA.
    """
    segment_count = centres.shape[0]
    return (
        centres.to("cuda").expand(row_count, -1, -1),
        directions.to("cuda").expand(row_count, -1, -1),
        torch.full(
            (row_count, segment_count), TUBE_SEGMENT_HALF_LENGTH_M,
            device="cuda",
        ),
        torch.ones(row_count, segment_count, dtype=torch.bool, device="cuda"),
    )


def tube_for(
        target: torch.Tensor,
        segments: tuple[torch.Tensor, ...],
        normal_mode: str,
    ) -> torch.Tensor:
    """
    Tube parameters of a problem, as the student builds them.

    Args:
        target (torch.Tensor): (T,B,7) cost target.
        segments (tuple): road_edge_segments-like edges.
        normal_mode (str): one of tube.TUBE_NORMALS.

    Returns:
        torch.Tensor: (T,B,6) planes, flat.
    """
    positions, headings = reference_positions(target)
    row_count = target.shape[1]
    planes, guarded = tube_planes(
        positions,
        headings,
        *segments,
        torch.full((row_count,), TUBE_EGO_LENGTH_M, device=target.device),
        torch.full((row_count,), TUBE_EGO_WIDTH_M, device=target.device),
        TUBE_MARGIN_M,
        normal_mode,
    )
    assert not bool(guarded.any())
    return planes.flatten(2)


@pytest.mark.parametrize("normal_mode", TUBE_NORMALS)
def test_tube_keeps_the_plan_inside_a_straight_road(normal_mode):
    # The cost pulls 5 m left; the left edge is 2.5 m away
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = left_pulling_problem(
        horizon_steps, row_count
    )
    target[..., 1] = 5.0
    tube = tube_for(target, straight_road_segments(row_count), normal_mode)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        free = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            TUBE_SQP_ITERATIONS, 25,
        )
        bounded = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            TUBE_SQP_ITERATIONS, 25, CONSTRAINT_TUBE, tube,
        )
    inner_bound_m = TUBE_ROAD_HALF_WIDTH_M - TUBE_BODY_OFFSET_M
    free_lateral = planned_positions(x0, wheelbase, free)[..., 1]
    bounded_lateral = planned_positions(x0, wheelbase, bounded)[..., 1]
    assert float(free_lateral.amax()) > inner_bound_m + 0.2
    assert float(bounded_lateral.amax()) < inner_bound_m + TUBE_TOLERANCE_M
    # The left edge binds; the plan stays clear of the right one
    assert float(bounded_lateral.amax()) > inner_bound_m - 0.2
    assert float(bounded_lateral.amin()) > -inner_bound_m


@pytest.mark.parametrize("normal_mode", TUBE_NORMALS)
def test_tube_keeps_the_plan_inside_a_bend(normal_mode):
    # Heading targets follow the bend at 10 m/s; the lateral target pulls
    # 5 m to the right of the ego, out of the bend
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = straight_problem(
        horizon_steps, row_count
    )
    times_s = 0.1 * torch.arange(horizon_steps, device="cuda")
    target[..., 2] = (10.0 * times_s / TUBE_BEND_RADIUS_M)[:, None]
    target[..., 1] = -5.0
    tube = tube_for(target, bend_segments(row_count), normal_mode)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        free = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            TUBE_SQP_ITERATIONS, 25,
        )
        bounded = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound,
            TUBE_SQP_ITERATIONS, 25, CONSTRAINT_TUBE, tube,
        )
    bend_centre = torch.tensor([0.0, TUBE_BEND_RADIUS_M], device="cuda")
    outer_radius_m = (
        TUBE_BEND_RADIUS_M + TUBE_ROAD_HALF_WIDTH_M - TUBE_BODY_OFFSET_M
    )
    inner_radius_m = (
        TUBE_BEND_RADIUS_M - TUBE_ROAD_HALF_WIDTH_M + TUBE_BODY_OFFSET_M
    )
    free_radius = (
        planned_positions(x0, wheelbase, free) - bend_centre
    ).norm(dim=-1)
    bounded_radius = (
        planned_positions(x0, wheelbase, bounded) - bend_centre
    ).norm(dim=-1)
    assert float(free_radius.amax()) > outer_radius_m + 0.2
    assert float(bounded_radius.amax()) < outer_radius_m + TUBE_TOLERANCE_M
    assert float(bounded_radius.amin()) > inner_radius_m - TUBE_TOLERANCE_M
    # Every node from 1 on has a plane on both sides of the bend
    assert float(tube[1:, :, 2::3].amax()) < PLANE_DUMMY_OFFSET_M


@needs_teacher
def test_tube_student_builds_its_planes_and_trains(teacher, device):
    horizon_steps = 10
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=horizon_steps,
        constraint_type=CONSTRAINT_TUBE,
    )
    assert student.outputs_per_step == COST_OUTPUTS_PER_STEP
    obs = rollout_obs(teacher, 20, OBS_SEED)
    x0 = ego_state_from_obs(obs)[0]
    raw = student.head_output(obs)
    tube = student.constraints(obs, raw, x0)
    assert tube.shape == (horizon_steps, obs.shape[0], 6)
    assert not tube.requires_grad
    normals = tube.unflatten(-1, (2, 3))[..., :2]
    torch.testing.assert_close(
        normals.norm(dim=-1), torch.ones_like(normals[..., 0])
    )
    # Real road edges: most rows see an edge on at least one side
    has_edge = (tube[..., 2::3] < PLANE_DUMMY_OFFSET_M).any(-1)
    assert float(has_edge[1:].float().mean()) > 0.5
    predicted = student.predict(obs)
    distillation_loss(predicted, labels_for(teacher, obs)).backward()
    assert bool(torch.isfinite(student.head.weight.grad).all())


# Previous-plan tube: the car follows its plan's first control in the
# same bicycle the env steps, so the shifted plan's node 0 lands on the
# car up to the env's control clipping and fp32 world coordinates
SHIFT_NODE0_MEDIAN_TOLERANCE_M = 0.02
CLOSED_LOOP_STEPS = 6
TEACHER_WARMUP_STEPS = 40
# Median distance an agent moves per step once the teacher got it going
MIN_MEDIAN_STEP_M = 0.2


@needs_teacher
def test_previous_plan_student_needs_poses_before_every_act(
        teacher, device,
    ):
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=5,
        constraint_type=CONSTRAINT_TUBE,
    )
    assert student.needs_agent_poses
    obs = rollout_obs(teacher, 2, OBS_SEED)[:8]
    driven = torch.ones(8, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="set_agent_poses"):
        student.act(obs, driven)
    student.set_agent_poses(torch.zeros(8, 3, device=device))
    student.act(obs, driven)
    # One act uses the poses up
    with pytest.raises(ValueError, match="set_agent_poses"):
        student.act(obs, driven)


@needs_teacher
def test_previous_plan_reference_follows_the_car_in_closed_loop(
        teacher, device,
    ):
    # The evaluator's hook hands over poses and new episodes; after a
    # step, the shifted plan's node 0 sits on the car, and a done agent
    # falls back to the straight reference
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=10,
        constraint_type=CONSTRAINT_TUBE,
    )
    policy = ContinuousPolicy(student)
    assert policy.needs_agent_poses
    env = make_small_env(OBS_SEED)
    try:
        obs, _ = env.reset(seed=OBS_SEED)
        # Agents start at rest: the teacher gets them moving first
        teacher_policy = ContinuousPolicy(teacher)
        for _ in range(TEACHER_WARMUP_STEPS):
            distribution, _ = teacher_policy.forward_eval(
                torch.as_tensor(obs, device=device)
            )
            obs, _, _, _, _ = env.step(
                np.clip(distribution.loc.cpu().numpy(), -1.0, 1.0)
            )
        new_episode = np.ones(env.num_agents, dtype=bool)
        node0_errors, moved_m = [], []
        for step_idx in range(CLOSED_LOOP_STEPS):
            if step_idx == CLOSED_LOOP_STEPS - 1:
                # A done agent forgets its plan before the next act
                new_episode[0] = True
            Evaluator._observe_agents(None, policy, env, new_episode)
            obs_tensor = torch.as_tensor(obs, device=device)
            vehicles = vehicle_rows(obs_tensor)
            if step_idx == 0:
                # No plan yet: the first act allocates invalid references
                assert student.reference_valid is None
            else:
                kept = vehicles & student.reference_valid
                positions, headings = shifted_reference(
                    student.reference_states[:, kept],
                    student.reference_poses[kept],
                    student.agent_poses.to(student.plans)[kept],
                )
                node0_errors.append(positions[0].norm(dim=-1))
                moved_m.append((
                    student.agent_poses.to(student.plans)[kept, :2]
                    - student.reference_poses[kept, :2]
                ).norm(dim=-1))
                assert float(headings[0].abs().max()) < 0.05
            if step_idx == CLOSED_LOOP_STEPS - 1:
                assert not bool(student.reference_valid[0])
                x0 = ego_state_from_obs(obs_tensor)[0]
                reference = student.previous_plan_reference(
                    x0[vehicles], vehicles,
                    student.agent_poses.to(student.plans),
                )
                straight = straight_reference(x0[vehicles, 3], 10)
                torch.testing.assert_close(
                    reference[0][:, 0], straight[0][:, 0]
                )
            distribution, _ = policy.forward_eval(obs_tensor)
            assert bool((student.reference_valid == vehicles).all())
            action = np.clip(distribution.loc.cpu().numpy(), -1.0, 1.0)
            obs, _, terminals, truncations, _ = env.step(action)
            new_episode = (
                np.asarray(terminals).astype(bool)
                | np.asarray(truncations).astype(bool)
            )
    finally:
        env.close()
    errors = torch.cat(node0_errors)
    moved = torch.cat(moved_m)
    print(f"\n  moved median {float(moved.median()):.3f} m; node-0 error "
          f"median {float(errors.median()):.4f} m, "
          f"max {float(errors.max()):.4f} m")
    assert float(moved.median()) > MIN_MEDIAN_STEP_M
    assert float(errors.median()) < SHIFT_NODE0_MEDIAN_TOLERANCE_M
    # Every closed-loop solve counted its rows; the pop resets
    stats = student.pop_tube_stats()
    assert 0.0 <= stats["tube/guard_fraction"] <= 1.0
    assert student.pop_tube_stats() == {}


# Route-anchored tube: share of tube rows (node >= 1) with a route
# anchor once the teacher got the agents moving on their routes
MIN_ROUTE_FRACTION = 0.9


@needs_teacher
def test_route_tube_needs_windows_and_anchors_in_closed_loop(
        teacher, device,
    ):
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=10,
        constraint_type=CONSTRAINT_TUBE, tube_anchor=TUBE_ANCHOR_ROUTE,
    )
    policy = ContinuousPolicy(student)
    assert policy.needs_agent_poses and policy.needs_route_window
    env = make_small_env(OBS_SEED)
    try:
        obs, _ = env.reset(seed=OBS_SEED)
        obs_tensor = torch.as_tensor(obs, device=device)
        driven = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        student.set_agent_poses(
            torch.zeros(obs.shape[0], 3, device=device)
        )
        with pytest.raises(ValueError, match="set_route_window"):
            student.act(obs_tensor, driven)
        teacher_policy = ContinuousPolicy(teacher)
        for _ in range(TEACHER_WARMUP_STEPS):
            distribution, _ = teacher_policy.forward_eval(
                torch.as_tensor(obs, device=device)
            )
            obs, _, _, _, _ = env.step(
                np.clip(distribution.loc.cpu().numpy(), -1.0, 1.0)
            )
        student.pop_tube_stats()
        new_episode = np.ones(env.num_agents, dtype=bool)
        for _ in range(CLOSED_LOOP_STEPS):
            Evaluator._observe_agents(None, policy, env, new_episode)
            distribution, _ = policy.forward_eval(
                torch.as_tensor(obs, device=device)
            )
            action = distribution.loc.cpu().numpy()
            assert bool(np.isfinite(action).all())
            obs, _, terminals, truncations, _ = env.step(
                np.clip(action, -1.0, 1.0)
            )
            new_episode = (
                np.asarray(terminals).astype(bool)
                | np.asarray(truncations).astype(bool)
            )
    finally:
        env.close()
    stats = student.pop_tube_stats()
    print(f"\n  route fraction {stats['tube/route_fraction']:.3f}, "
          f"guard fraction {stats['tube/guard_fraction']:.3f}")
    assert stats["tube/route_fraction"] >= MIN_ROUTE_FRACTION
    # Open-loop solves (the forgetting probe's predict) are not counted
    student.predict(torch.as_tensor(obs, device=device))
    assert student.pop_tube_stats() == {}


@needs_teacher
def test_previous_plan_tube_trains_with_fresh_gradients_only(
        teacher, device,
    ):
    config = DistillConfig(
        seed=OBS_SEED, device=device, iterations=1,
        env_steps_per_iteration=4, gradient_steps_per_iteration=2,
        batch_size=64, buffer_size=10_000, eval_interval_labels=0,
        fresh_gradient=True,
    )
    for fresh_gradient in (False, True):
        student = fresh_mpc_student(
            teacher, device, solver="warpmpc", horizon_steps=5,
            constraint_type=CONSTRAINT_TUBE,
        )
        env = make_small_env(OBS_SEED)
        obs_dim = env.single_observation_space.shape[0]
        buffer = LabelBuffer(config.buffer_size, obs_dim, device)
        run_config = dataclasses.replace(
            config, fresh_gradient=fresh_gradient
        )
        try:
            if not fresh_gradient:
                with pytest.raises(ValueError, match="fresh_gradient"):
                    distill(env, teacher, student, buffer, run_config)
                continue
            result = distill(env, teacher, student, buffer, run_config)
        finally:
            env.close()
        assert student.weight_version > 0
        assert math.isfinite(result.log_rows[0]["loss"])
        assert 0.0 <= result.log_rows[0]["tube/guard_fraction"] <= 1.0
        assert bool(student.reference_valid.any())


@pytest.mark.slow
@needs_teacher
def test_evaluator_runs_the_previous_plan_tube_student(
        teacher, device, distill_args,
    ):
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=5,
        constraint_type=CONSTRAINT_TUBE,
    )
    evaluator = DistillEvaluator(distill_args, num_scenarios=2)
    metrics = evaluator.evaluate(student)
    for name in ("num_goals_reached", "collision_rate", "offroad_rate"):
        assert name in metrics and np.isfinite(metrics[name])


# Tube with partner ellipses: the car of blocking_obstacle on a road
# wide enough to pass it inside the planes
TUBE_WIDE_ROAD_HALF_WIDTH_M = 3.5


@pytest.mark.parametrize("normal_mode", TUBE_NORMALS)
def test_tube_ellipse_keeps_the_plan_off_a_car_and_inside_the_road(
        normal_mode,
    ):
    horizon_steps, row_count = 10, 256
    x0, wheelbase, q_diag, target = straight_problem(
        horizon_steps, row_count
    )
    centre_x = torch.arange(-9.0, 40.0, 2.0 * TUBE_SEGMENT_HALF_LENGTH_M)
    centres = torch.cat([
        torch.stack([centre_x, torch.full_like(centre_x, side_y)], -1)
        for side_y in (
            TUBE_WIDE_ROAD_HALF_WIDTH_M, -TUBE_WIDE_ROAD_HALF_WIDTH_M,
        )
    ])
    directions = torch.zeros_like(centres)
    directions[:, 0] = 1.0
    planes = tube_for(
        target, road_on_cuda(centres, directions, row_count), normal_mode
    )
    ellipses = blocking_obstacle(horizon_steps, row_count)
    tube = torch.cat([planes, ellipses.flatten(2)], dim=-1)
    bound = torch.tensor(STUDENT_BOUND, device="cuda")
    with torch.no_grad():
        planes_only = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 10, 25,
            CONSTRAINT_TUBE, planes,
        )
        with_car = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 10, 25,
            CONSTRAINT_TUBE, tube,
        )
    car = ellipses[0, 0, 0]

    def ellipse_value(controls):
        """Smallest ellipse value of the standing car along the plan."""
        offset = planned_positions(x0, wheelbase, controls) - car[:2]
        return ((offset[..., 0] / car[4]) ** 2
                + (offset[..., 1] / car[5]) ** 2).amin(-1)

    assert float(ellipse_value(planes_only).max()) < 1.0
    assert float(ellipse_value(with_car).min()) > 0.98
    lateral = planned_positions(x0, wheelbase, with_car)[..., 1]
    inner_bound_m = TUBE_WIDE_ROAD_HALF_WIDTH_M - TUBE_BODY_OFFSET_M
    # Passes the car on the right, inside the right plane
    assert float(lateral.amin(-1).max()) < 0.0
    assert float(lateral.amin()) > -inner_bound_m - TUBE_TOLERANCE_M


@needs_teacher
def test_tube_student_with_ellipses_trains_acts_and_draws_them(
        teacher, device,
    ):
    horizon_steps, ellipse_count = 10, 4
    student = fresh_mpc_student(
        teacher, device, solver="warpmpc", horizon_steps=horizon_steps,
        constraint_type=CONSTRAINT_TUBE, constraint_count=ellipse_count,
    )
    # The ellipses come from the observation: no extra head outputs
    assert student.outputs_per_step == COST_OUTPUTS_PER_STEP
    obs = rollout_obs(teacher, 20, OBS_SEED)
    x0 = ego_state_from_obs(obs)[0]
    tube = student.constraints(obs, student.head_output(obs), x0)
    assert tube.shape == (
        horizon_steps, obs.shape[0], 6 + OBSTACLE_PARAM_COUNT * ellipse_count
    )
    ellipses = tube[..., 6:].unflatten(-1, (ellipse_count, 6))
    torch.testing.assert_close(
        ellipses,
        partner_obstacles(
            obs, student.partner_start_idx, ellipse_count, horizon_steps
        ),
    )
    # Some rows see fewer than K partners: their last slots are dummies
    dummy = torch.tensor(
        [500.0, 0.0, 1.0, 0.0, 1.0, 1.0], device=device
    )
    is_dummy = (ellipses[0] == dummy).all(-1)
    assert bool(is_dummy.any()) and not bool(is_dummy.all())
    predicted = student.predict(obs)
    distillation_loss(predicted, labels_for(teacher, obs)).backward()
    assert bool(torch.isfinite(student.head.weight.grad).all())
    driven = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
    poses = torch.zeros(obs.shape[0], 3, device=device)
    student.set_agent_poses(poses)
    controls, chunk = student.act_and_grad(
        obs, labels_for(teacher, obs), driven, driven
    )
    assert bool(torch.isfinite(controls).all())
    assert bool(torch.isfinite(chunk.raw_grad).all())
    student.record_overlay = True
    student.set_agent_poses(poses[:16])
    student.reset_plans(driven)
    student.act(obs[:16], driven[:16])
    rows = student.last_overlay
    assert bool((rows[:, OVERLAY_ELLIPSE_COUNT_IDX] == ellipse_count).all())
    nodes = rows[:, OVERLAY_HEADER:].view(16, -1, OVERLAY_NODE_FLOATS)
    ellipses_end = NODE_ELLIPSES_START + 6 * ellipse_count
    torch.testing.assert_close(
        nodes[:, :horizon_steps, NODE_ELLIPSES_START:ellipses_end],
        ellipses[:, :16].transpose(0, 1).flatten(2),
    )


@needs_teacher
def test_tube_ellipses_must_fit_the_partner_slots(teacher, device):
    slots = teacher.policy.actor_backbone.obs_slots_partners_n
    with pytest.raises(ValueError, match="partner slots"):
        fresh_mpc_student(
            teacher, device, solver="warpmpc", horizon_steps=5,
            constraint_type=CONSTRAINT_TUBE, constraint_count=slots + 1,
        )


@pytest.mark.parametrize("ellipse_count", [0, 2])
def test_slack_weight_scale_multiplies_every_slack_price(
        ellipse_count: int,
    ) -> None:
    """slack_weight_scale scales the tube's plane and ellipse slack price,
    linear and quadratic terms alike."""
    x = ca.SX.sym("x", NX)
    slack = ca.SX.sym("s", 2 + ellipse_count)
    params = ca.SX.sym("p", 6 + 6 * ellipse_count)
    slack_value = np.linspace(0.5, 2.0, 2 + ellipse_count)
    costs = []
    for scale in (1.0, 10.0):
        cost, *_ = tube_terms(x, slack, params, scale)
        price = ca.Function("price", [slack], [cost])
        costs.append(float(price(slack_value)))
    assert costs[0] > 0
    assert math.isclose(costs[1], 10.0 * costs[0], rel_tol=1e-9)


SPLIT_FIRST_BOUND = (4.0, 0.75)
SPLIT_PLAN_BOUND = (4.0, 0.6)
SPLIT_HORIZON_STEPS = 10
SPLIT_ROW_COUNT = 64
SPLIT_BOX_TOLERANCE = 1.005


def hard_turn_problem() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Straight driving asked to reach full steering at once: the plan wants
    the largest steering rate on every node, on CUDA.

    Returns:
        tuple: x0 (B,5), wheelbase (B,) [m], q_diag (T,B,7),
            target (T,B,7).
    """
    x0, wheelbase, q_diag, target = straight_problem(
        SPLIT_HORIZON_STEPS, SPLIT_ROW_COUNT
    )
    q_diag[..., 1:4] = 0.0
    q_diag[..., 4] = 100.0
    q_diag[..., NX:] = 0.01
    target[..., 4] = STEERING_LIMIT_RAD
    return x0, wheelbase, q_diag, target


def test_plan_bound_holds_the_plan_and_frees_the_first_control():
    """With a 0.6 plan bound and a 0.75 first bound, the first steering
    rate goes past 0.6 and every later one stays within 0.6."""
    x0, wheelbase, q_diag, target = hard_turn_problem()
    first_bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    plan_bound = torch.tensor(SPLIT_PLAN_BOUND, device="cuda")
    with torch.no_grad():
        controls = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, first_bound, 5, 25,
            plan_control_bound=plan_bound,
        )
    first_rate = controls[0, :, 1].abs()
    later_rate = controls[1:, :, 1].abs()
    assert float(first_rate.min()) > SPLIT_PLAN_BOUND[1] * SPLIT_BOX_TOLERANCE
    assert float(first_rate.max()) <= (
        SPLIT_FIRST_BOUND[1] * SPLIT_BOX_TOLERANCE
    )
    assert float(later_rate.max()) <= SPLIT_PLAN_BOUND[1] * SPLIT_BOX_TOLERANCE


def test_plan_bound_equal_to_the_box_changes_nothing():
    """A plan bound equal to the control box solves the same problem as
    no plan bound."""
    x0, wheelbase, q_diag, target = hard_turn_problem()
    bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    with torch.no_grad():
        default = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 5, 25,
        )
        same = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 5, 25,
            plan_control_bound=bound.clone(),
        )
    assert torch.equal(default, same)


def test_explicit_slack_prices_equal_to_the_constants_change_nothing():
    """slack_prices equal to the scaled constants solve the same problem
    as no slack_prices (planes: linear 100, quadratic 10)."""
    x0, wheelbase, q_diag, target = straight_problem(10, SPLIT_ROW_COUNT)
    planes = dummy_constraints(
        CONSTRAINT_PLANE, 10, SPLIT_ROW_COUNT, 2, torch.float32,
        torch.device("cuda"),
    )
    bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    with torch.no_grad():
        default = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25,
            CONSTRAINT_PLANE, planes,
        )
        explicit = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25,
            CONSTRAINT_PLANE, planes, slack_prices=(100.0, 10.0),
        )
    assert torch.equal(default, explicit)


def test_scaling_cost_and_slack_prices_together_keeps_the_plan():
    """The QP's minimizer does not depend on a common scale of the cost
    and the slack prices: weights and price x 4 give the same converged
    plan (WarpMPC's ADMM is not scale-free, hence many iterations). The
    dummy planes stay inactive, so this checks the scaled cost with the
    slack variables present; slack_sum_check covers active rows."""
    x0, wheelbase, q_diag, target = hard_turn_problem()
    planes = dummy_constraints(
        CONSTRAINT_PLANE, SPLIT_HORIZON_STEPS, SPLIT_ROW_COUNT, 2,
        torch.float32, torch.device("cuda"),
    )
    bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    with torch.no_grad():
        small = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 5, 200,
            CONSTRAINT_PLANE, planes, slack_prices=(10.0, 0.0),
        )
        large = solve_cost_warpmpc(
            x0, wheelbase, 4.0 * q_diag, target, None, bound, 5, 200,
            CONSTRAINT_PLANE, planes, slack_prices=(40.0, 0.0),
        )
    assert float(((small - large).abs() / bound).max()) < 2e-2



def test_constant_node_slack_price_equals_the_global_price():
    """A per-node slack price that is the same on every node and row
    solves the same problem as that global linear price."""
    x0, wheelbase, q_diag, target = straight_problem(10, SPLIT_ROW_COUNT)
    planes = dummy_constraints(
        CONSTRAINT_TUBE, 10, SPLIT_ROW_COUNT, 0, torch.float32,
        torch.device("cuda"),
    )
    bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    price = torch.full((10, SPLIT_ROW_COUNT), 40.0, device="cuda")
    with torch.no_grad():
        global_price = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25,
            CONSTRAINT_TUBE, planes, slack_prices=(40.0, 0.0),
        )
        node_price = solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25,
            CONSTRAINT_TUBE, planes, node_slack_price=price,
        )
    assert torch.allclose(global_price, node_price, atol=1e-6)


def test_node_slack_price_needs_constraints():
    """A per-node slack price without constraint rows raises."""
    x0, wheelbase, q_diag, target = straight_problem(10, SPLIT_ROW_COUNT)
    bound = torch.tensor(SPLIT_FIRST_BOUND, device="cuda")
    price = torch.ones(10, SPLIT_ROW_COUNT, device="cuda")
    with pytest.raises(ValueError):
        solve_cost_warpmpc(
            x0, wheelbase, q_diag, target, None, bound, 3, 25,
            node_slack_price=price,
        )
