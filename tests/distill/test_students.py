"""MPC student: controls in its box, gradients through the solver, warm
starts, and that it can learn; plus the MLP's configurable output range."""

import os
import time

import numpy as np
import pytest
import torch

from pufferlib.distill.bicycle import (
    ACCEL,
    STEERING,
    STEERING_RATE,
    U_HI,
    Bicycle,
    ego_state_from_obs,
    env_action_from_control,
    solve,
    target_cost,
)
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.constraints import CONSTRAINT_TUBE
from pufferlib.distill.env import SMALL_TEACHER_CHECKPOINT
from pufferlib.distill.loop import (
    DistillConfig,
    distillation_loss,
    make_optimizer,
    train_student,
)
from pufferlib.distill.students import (
    COST_OUTPUTS_PER_STEP,
    GradChunk,
    MLPStudent,
    MLPStudentConfig,
    MPCStudent,
    MPCStudentConfig,
    SOLVER_WARPMPC,
)
from pufferlib.distill.teacher import Teacher

from conftest import make_small_env, needs_teacher
from test_teacher import rollout_obs

STUDENT_SEED = 61
OBS_SEED = 62
PD_RATE_BOUND = float(U_HI[STEERING_RATE])
WIDE_RATE_BOUND = 0.75
# Head bias that asks for a hard left turn: large lateral target and
# weight, cheap controls
LATERAL_TARGET_LOGIT = 5.0
HIGH_WEIGHT_LOGIT = 5.0
LOW_WEIGHT_LOGIT = -10.0
# A moderate turn cost that leaves ~12% of moving rows with an optimal
# rate strictly between PD's bound and the wide one
MODERATE_CONTROL_LOGIT = -2.0
MODERATE_LATERAL_LOGIT = 0.3
INTERIOR_MARGIN = 0.02
MIN_INTERIOR_ROWS = 20
MIN_MOVING_SPEED_MPS = 2.0
FD_SAMPLE_COUNT = 16
FD_EPS = 1e-5
FD_CONVERGED_ITERATIONS = 30
MIN_FD_COSINE = 0.95
# With 3 unconverged iLQR iterations, rounding can flip a line-search
# step: ~10% of entries differ by up to ~1e-3 of PD's box (0.0006 rad/s),
# far below the label error the loss sees (RMS ~0.3)
FP32_TOL = 1e-2
WARM_COLD_MEAN_TOL = 0.05
OVERFIT_STEPS = 300
OVERFIT_LEARNING_RATE = 1e-3
OVERFIT_LOSS_RATIO = 0.5
TIMING_BATCH_SIZE = 4096
TIMING_ACT_AGENTS = 1024
LEAK_BATCH_SIZE = 1024
LEAK_STEPS = 20
# Unpatched: ~0.8 MiB per step at this batch; closures released but ctx
# trajectories kept: ~0.14; fully patched: flat
MAX_GROWTH_PER_STEP_MIB = 0.01


def fresh_mpc_student(
        teacher: Teacher,
        device: str,
        **config_overrides,
    ) -> MPCStudent:
    """
    Seeded MPC student.

    Args:
        teacher (Teacher): backbone source.
        device (str): torch device.
        **config_overrides: MPCStudentConfig fields.

    Returns:
        MPCStudent: the student on device.
    """
    torch.manual_seed(STUDENT_SEED)
    return MPCStudent(teacher, MPCStudentConfig(**config_overrides)).to(
        device
    )


def moving_obs(
        teacher: Teacher,
        step_count: int = 40,
    ) -> torch.Tensor:
    """
    Rollout observations of agents that are moving (steering acts on the
    pose only when speed > 0).

    Args:
        teacher (Teacher): drives the rollout.
        step_count (int): env steps.

    Returns:
        torch.Tensor: (N,obs_dim) observations with speed above
            MIN_MOVING_SPEED_MPS.
    """
    obs = rollout_obs(teacher, step_count, OBS_SEED)
    speed = ego_state_from_obs(obs)[0][:, 3]
    return obs[speed > MIN_MOVING_SPEED_MPS]


def ask_for_turn(
        student: MPCStudent,
        control_weight_logit: float = LOW_WEIGHT_LOGIT,
        lateral_target_logit: float = LATERAL_TARGET_LOGIT,
    ) -> None:
    """
    Set the head to a constant cost that wants a lateral offset. The
    defaults (large offset, cheap controls) put the optimal rate on the
    box edge; costlier controls and a smaller offset land some rows
    inside it.

    Args:
        student (MPCStudent): modified in place.
        control_weight_logit (float): logit of both control weights.
        lateral_target_logit (float): logit of the lateral target.
    """
    bias = torch.zeros(student.horizon_steps, COST_OUTPUTS_PER_STEP)
    bias[:, :4] = LOW_WEIGHT_LOGIT
    bias[:, 0] = HIGH_WEIGHT_LOGIT  # lateral offset weight
    bias[:, 4:6] = control_weight_logit
    bias[:, 6] = lateral_target_logit
    with torch.no_grad():
        student.head.weight.zero_()
        student.head.bias.copy_(bias.view(-1))


def labels_for(
        teacher: Teacher,
        obs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Args:
        teacher (Teacher): labeler.
        obs (torch.Tensor): (B,obs_dim) observations.

    Returns:
        torch.Tensor: (B,2) teacher controls.
    """
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    return teacher.label(obs, steering)


@needs_teacher
def test_mpc_controls_in_box_and_shape(
        teacher: Teacher,
        device: str,
    ) -> None:
    """predict and act return finite (B,2) controls inside the box."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 20, OBS_SEED)
    driven = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
    for control in (student.predict(obs), student.act(obs, driven)):
        assert control.shape == (obs.shape[0], 2)
        assert bool(torch.isfinite(control).all())
        assert bool((control.abs() <= student.control_bound + 1e-6).all())


@needs_teacher
def test_mpc_box_is_wider_than_pd(
        teacher: Teacher,
        device: str,
    ) -> None:
    """The student reaches rates beyond 0.6; PD's default box clips."""
    student = fresh_mpc_student(teacher, device)
    ask_for_turn(student)
    obs = moving_obs(teacher)
    rate = student.predict(obs)[:, STEERING_RATE]
    print(f"\n  max planned rate {float(rate.abs().max()):.3f} rad/s")
    assert float(rate.abs().max()) > PD_RATE_BOUND + 0.05
    assert float(rate.abs().max()) <= WIDE_RATE_BOUND + 1e-6

    x0, wheelbase, q_diag, target = student.cost_terms(obs)
    Q, p = target_cost(q_diag, target)
    _, default_box_controls = solve(
        x0, Q, p, Bicycle(wheelbase), student.horizon_steps, iters=3
    )
    assert float(default_box_controls.abs().max(0).values.max(0).values[
        STEERING_RATE
    ]) <= PD_RATE_BOUND + 1e-6


@needs_teacher
def test_gradient_at_pd_bound_is_nonzero(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Rows whose optimal rate lies in (0.6, 0.75): with the wide box they
    are interior and get a gradient; PD's box pins them at 0.6 against a
    label on the opposite edge, and the gradient vanishes."""
    obs = moving_obs(teacher)

    def turning_student(rate_bound: float) -> MPCStudent:
        """
        Args:
            rate_bound (float): student's rate box.

        Returns:
            MPCStudent: student with a moderate constant turn cost.
        """
        student = fresh_mpc_student(
            teacher, device, steering_rate_bound_radps=rate_bound
        )
        ask_for_turn(student, MODERATE_CONTROL_LOGIT, MODERATE_LATERAL_LOGIT)
        return student

    with torch.no_grad():
        wide_rate = turning_student(WIDE_RATE_BOUND).predict(obs)[
            :, STEERING_RATE
        ]
    between = (wide_rate.abs() > PD_RATE_BOUND + INTERIOR_MARGIN) & (
        wide_rate.abs() < WIDE_RATE_BOUND - INTERIOR_MARGIN
    )
    assert int(between.sum()) >= MIN_INTERIOR_ROWS
    rows = obs[between]

    def head_gradient_norm(rate_bound: float) -> tuple[float, float]:
        """
        Args:
            rate_bound (float): student's rate box.

        Returns:
            tuple[float, float]: |d loss / d head.bias| and the largest
                |rate| on the selected rows.
        """
        student = turning_student(rate_bound)
        predicted = student.predict(rows)
        # Label on the opposite edge of PD's box; accel matches
        label = predicted.detach().clone()
        label[:, STEERING_RATE] = -torch.sign(
            predicted[:, STEERING_RATE].detach()
        ) * PD_RATE_BOUND
        distillation_loss(predicted, label).backward()
        rate = predicted[:, STEERING_RATE].detach().abs().max()
        return float(student.head.bias.grad.norm()), float(rate)

    pinned_gradient, pinned_rate = head_gradient_norm(PD_RATE_BOUND)
    wide_gradient, _ = head_gradient_norm(WIDE_RATE_BOUND)
    print(
        f"\n  {int(between.sum())} rows; box 0.60: max rate "
        f"{pinned_rate:.3f}, grad {pinned_gradient:.2e}; box 0.75: grad "
        f"{wide_gradient:.2e}"
    )
    assert pinned_rate <= PD_RATE_BOUND + 1e-6
    assert pinned_gradient < 1e-6
    assert wide_gradient > 1e-3


@needs_teacher
def test_initial_mpc_student_holds_lane_and_speed(
        teacher: Teacher,
        device: str,
    ) -> None:
    """At reset (speed 0, steering 0) a fresh head asks for ~nothing."""
    student = fresh_mpc_student(teacher, device)
    env = make_small_env(OBS_SEED)
    obs, _ = env.reset(seed=OBS_SEED)
    env.close()
    obs = torch.from_numpy(obs).to(device)
    normalized = student.predict(obs).detach() / U_HI.to(device)
    mean_abs = normalized.abs().mean(0)
    print(f"\n  mean |u0| / PD box at reset: {mean_abs.tolist()}")
    assert float(mean_abs.max()) < 0.1


@needs_teacher
def test_gradient_reaches_head_and_backbone(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Every parameter gets a finite gradient; head and backbone move."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 20, OBS_SEED)
    loss = distillation_loss(student.predict(obs), labels_for(teacher, obs))
    loss.backward()
    gradients = [p.grad for p in student.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all())
               for g in gradients)
    assert float(student.head.weight.grad.abs().sum()) > 0
    nonzero_backbone = [
        float(p.grad.abs().sum()) > 0 for p in student.backbone.parameters()
    ]
    assert np.mean(nonzero_backbone) > 0.5


@needs_teacher
def test_mpc_gradient_matches_finite_differences(
        teacher: Teacher,
        device: str,
    ) -> None:
    """d loss / d head.bias through the solver vs central differences
    (float64, converged solves); the analytic gradient is exact only for
    the linearization, so the check is on direction."""
    student = fresh_mpc_student(
        teacher, device, train_lqr_iterations=FD_CONVERGED_ITERATIONS
    ).double()
    obs = rollout_obs(teacher, 20, OBS_SEED)[:FD_SAMPLE_COUNT].double()
    labels = labels_for(teacher, obs.float()).double()
    with torch.no_grad():
        student.head.bias.normal_(std=0.5)

    student.zero_grad()
    distillation_loss(student.predict(obs), labels).backward()
    analytic = student.head.bias.grad.clone()

    finite_difference = torch.zeros_like(analytic)
    with torch.no_grad():
        for index in range(analytic.numel()):
            student.head.bias[index] += FD_EPS
            loss_plus = distillation_loss(student.predict(obs), labels)
            student.head.bias[index] -= 2 * FD_EPS
            loss_minus = distillation_loss(student.predict(obs), labels)
            student.head.bias[index] += FD_EPS
            finite_difference[index] = (loss_plus - loss_minus) / (
                2 * FD_EPS
            )
    cosine = torch.nn.functional.cosine_similarity(
        analytic, finite_difference, dim=0
    )
    relative = (analytic - finite_difference).norm() / (
        finite_difference.norm()
    )
    print(f"\n  cosine {float(cosine):.4f}, relative error "
          f"{float(relative):.3f}")
    assert float(cosine) > MIN_FD_COSINE


@needs_teacher
def test_fp32_close_to_fp64(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Training in fp32 gives the same controls as fp64, up to
    line-search flips far below the loss scale."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 20, OBS_SEED)
    fp32 = student.predict(obs).detach()
    fp64 = student.double().predict(obs.double()).detach()
    difference = ((fp32.double() - fp64) / U_HI.to(fp64)).abs()
    print(f"\n  max |fp32 - fp64| / PD box: {float(difference.max()):.2e}")
    assert float(difference.max()) < FP32_TOL


@needs_teacher
def test_act_warm_start_shift_and_reset(
        teacher: Teacher,
        device: str,
    ) -> None:
    """act keeps a shifted plan per row; reset zeroes only masked rows;
    a new batch size starts from zero plans."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 5, OBS_SEED)
    batch_size = obs.shape[0]
    driven = torch.ones(batch_size, dtype=torch.bool, device=device)
    first = student.act(obs, driven)
    plans = student.plans
    assert plans.shape == (student.horizon_steps, batch_size, 2)
    # The plan starts one step later; its last control is repeated
    assert torch.equal(plans[-1], plans[-2])
    assert not torch.equal(plans[0], first) or bool((first == 0).all())

    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[::3] = True
    kept = plans[:, ~mask].clone()
    student.reset_plans(mask)
    assert bool((student.plans[:, mask] == 0).all())
    assert torch.equal(student.plans[:, ~mask], kept)

    student.act(obs[:7], driven[:7])
    assert student.plans.shape == (student.horizon_steps, 7, 2)


@needs_teacher
def test_warm_started_act_close_to_cold_predict(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Along a rollout, the warm-started act matches a cold solve with
    the same iterations."""
    student = fresh_mpc_student(teacher, device)
    with torch.no_grad():
        student.head.bias.normal_(std=0.5)
    env = make_small_env(OBS_SEED)
    obs, _ = env.reset(seed=OBS_SEED)
    differences = []
    for _ in range(30):
        obs_tensor = torch.from_numpy(obs).to(device)
        driven = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        warm = student.act(obs_tensor, driven)
        with torch.no_grad():
            cold = student.predict(obs_tensor)
        differences.append(((warm - cold) / U_HI.to(device)).abs().cpu())
        steering = ego_state_from_obs(obs_tensor)[0][:, STEERING]
        control = teacher.label(obs_tensor, steering)
        obs, _, _, _, _ = env.step(
            env_action_from_control(control, steering).cpu().numpy()
        )
    env.close()
    differences = torch.cat(differences)
    mean = float(differences.mean())
    p99 = float(torch.quantile(differences.flatten(), 0.99))
    print(f"\n  |warm - cold| / PD box: mean {mean:.4f}, p99 {p99:.4f}")
    assert mean < WARM_COLD_MEAN_TOL


def shipped_gradients(
        student: MPCStudent,
        chunk: GradChunk,
        row_count: int,
    ) -> list[torch.Tensor]:
    """
    Parameter gradients the learner derives from a chunk.

    Args:
        student (MPCStudent): learner weights.
        chunk (GradChunk): shipped rows.
        row_count (int): rows in the update.

    Returns:
        list[torch.Tensor]: one gradient per parameter.
    """
    student.zero_grad()
    student.apply_cost_gradient(chunk, row_count)
    return [parameter.grad.clone() for parameter in student.parameters()]


def predict_gradients(
        student: MPCStudent,
        obs: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[torch.Tensor]:
    """
    Parameter gradients of the replay loss on a cold-started solve.

    Args:
        student (MPCStudent): model.
        obs (torch.Tensor): (B,obs_dim) observations.
        labels (torch.Tensor): (B,2) teacher controls.

    Returns:
        list[torch.Tensor]: one gradient per parameter.
    """
    student.zero_grad()
    distillation_loss(student.predict(obs), labels).backward()
    return [parameter.grad.clone() for parameter in student.parameters()]


@needs_teacher
def test_shipped_gradient_matches_predict_autograd(
        teacher: Teacher,
        device: str,
    ) -> None:
    """With unchanged weights, act_and_grad plus apply_cost_gradient
    gives the gradient of predict's replay loss on the live rows (first
    step of a rollout: plans are zero, so the solve is cold)."""
    student = fresh_mpc_student(teacher, device).double()
    with torch.no_grad():
        student.head.bias.normal_(std=0.5)
    obs = rollout_obs(teacher, 20, OBS_SEED).double()
    labels = labels_for(teacher, obs.float()).double()
    live = torch.zeros(obs.shape[0], dtype=torch.bool, device=device)
    live[::2] = True
    driven = ~live

    control, chunk = student.act_and_grad(obs, labels, live, driven)
    live_count = int(live.sum())
    assert chunk.obs.shape[0] == live_count
    assert chunk.raw_grad.shape == (
        live_count, student.horizon_steps * COST_OUTPUTS_PER_STEP
    )
    assert torch.equal(chunk.student_drove, driven[live])
    with torch.no_grad():
        assert torch.allclose(control, student.predict(obs))

    shipped = shipped_gradients(student, chunk, live_count)
    expected = predict_gradients(student, obs[live], labels[live])
    for shipped_grad, expected_grad in zip(shipped, expected):
        torch.testing.assert_close(
            shipped_grad, expected_grad, rtol=1e-6, atol=1e-9
        )


@needs_teacher
def test_chunk_keeps_gradient_of_collecting_weights(
        teacher: Teacher,
        device: str,
    ) -> None:
    """The chunk is fixed at collection: after a weight change the
    shipped gradient is still the old head-output gradient, and differs
    from the gradient of a fresh solve."""
    student = fresh_mpc_student(teacher, device).double()
    with torch.no_grad():
        student.head.bias.normal_(std=0.5)
    obs = rollout_obs(teacher, 20, OBS_SEED).double()
    labels = labels_for(teacher, obs.float()).double()
    live = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
    _, chunk = student.act_and_grad(obs, labels, live, live)
    stored = chunk.raw_grad.clone()
    with torch.no_grad():
        student.head.bias.add_(torch.randn_like(student.head.bias))

    assert torch.equal(chunk.raw_grad, stored)
    shipped = shipped_gradients(student, chunk, obs.shape[0])
    fresh = predict_gradients(student, obs, labels)
    assert not torch.allclose(shipped[-2], fresh[-2])


@needs_teacher
def test_non_finite_solver_gradient_is_rejected(
        teacher: Teacher,
        device: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
    """A NaN from the solve aborts instead of reaching the learner."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 5, OBS_SEED)
    labels = labels_for(teacher, obs)
    live = torch.ones(obs.shape[0], dtype=torch.bool, device=device)

    def nan_solve(x0, Q, p, *args, **kwargs):
        return None, p[..., :2] * float("nan")

    monkeypatch.setattr("pufferlib.distill.students.solve", nan_solve)
    with pytest.raises(ValueError, match="non-finite"):
        student.act_and_grad(obs, labels, live, live)


@needs_teacher
def test_mpc_student_overfits_small_buffer(
        teacher: Teacher,
        device: str,
    ) -> None:
    """The loss over 640 fixed labels halves within 300 steps."""
    obs = rollout_obs(teacher, 10, OBS_SEED)
    labels = labels_for(teacher, obs)
    buffer = LabelBuffer(obs.shape[0], obs.shape[1], device)
    buffer.add(obs, labels)
    student = fresh_mpc_student(teacher, device)
    config = DistillConfig(
        device=device,
        gradient_steps_per_iteration=OVERFIT_STEPS,
        batch_size=256,
        buffer_size=obs.shape[0],
        learning_rate=OVERFIT_LEARNING_RATE,
    )

    def full_loss() -> float:
        """
        Returns:
            float: loss over all stored rows.
        """
        with torch.no_grad():
            return float(distillation_loss(student.predict(obs), labels))

    initial = full_loss()
    train_student(student, make_optimizer(student, config), buffer, config)
    final = full_loss()
    print(f"\n  MPC overfit loss {initial:.4f} -> {final:.4f}")
    assert final < OVERFIT_LOSS_RATIO * initial


@pytest.mark.parametrize(
    "bad_settings",
    [
        dict(horizon_steps=0),
        dict(train_lqr_iterations=0),
        dict(act_lqr_iterations=-1),
        dict(steering_rate_bound_radps=0.5),
        dict(accel_bound_mps2=3.0),
        dict(solver="osqp"),
        dict(sqp_iterations=0),
        dict(admm_iterations=0),
    ],
)
def test_invalid_mpc_config_is_rejected(
        bad_settings: dict,
    ) -> None:
    """Nonpositive counts and a box narrower than PD's are rejected."""
    with pytest.raises(ValueError):
        MPCStudentConfig(**bad_settings)


@needs_teacher
def test_mlp_output_range_is_configurable(
        teacher: Teacher,
        device: str,
    ) -> None:
    """MLP default spans PD's box; the wide config reaches 0.75."""
    obs = rollout_obs(teacher, 5, OBS_SEED)
    for bound in (PD_RATE_BOUND, WIDE_RATE_BOUND):
        torch.manual_seed(STUDENT_SEED)
        config = MLPStudentConfig(steering_rate_bound_radps=bound)
        student = MLPStudent(teacher, config).to(device)
        with torch.no_grad():
            student.head.bias[STEERING_RATE] = 20.0  # saturate tanh
        rate = student.predict(obs)[:, STEERING_RATE]
        assert float(rate.max()) == pytest.approx(bound, rel=1e-4)
        assert float(student.control_bound[ACCEL]) == float(U_HI[ACCEL])
    with pytest.raises(ValueError):
        MLPStudentConfig(steering_rate_bound_radps=0.5)


@needs_teacher
def test_report_mpc_timing(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Reported, not asserted: training step and act cost."""
    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 64, OBS_SEED)
    batch = obs[torch.randint(obs.shape[0], (TIMING_BATCH_SIZE,))]
    labels = labels_for(teacher, batch)
    optimizer = torch.optim.Adam(student.parameters())

    def train_step() -> None:
        """
        One optimizer step on the batch.
        """
        optimizer.zero_grad()
        distillation_loss(student.predict(batch), labels).backward()
        optimizer.step()

    act_obs = batch[:TIMING_ACT_AGENTS]
    driven = torch.ones(TIMING_ACT_AGENTS, dtype=torch.bool, device=device)
    for name, step in (
        (f"train step, batch {TIMING_BATCH_SIZE}", train_step),
        (f"act, {TIMING_ACT_AGENTS} agents",
         lambda: student.act(act_obs, driven)),
    ):
        step()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"\n  {name}: {(time.perf_counter() - start) / 3 * 1000:.0f} ms")


@needs_teacher
def test_empty_slots_get_zero_controls_and_leave_others_unchanged(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Evaluators pad unused agent slots with zero observations: act
    returns 0 there and the same controls for the real rows; predict
    rejects them."""
    obs = rollout_obs(teacher, 5, OBS_SEED)[:64]
    padded = torch.cat([obs, torch.zeros_like(obs[:16])])
    driven = torch.ones(padded.shape[0], dtype=torch.bool, device=device)
    vehicles_only = fresh_mpc_student(teacher, device).act(
        obs, driven[:64]
    )
    with_padding = fresh_mpc_student(teacher, device).act(padded, driven)
    assert bool(torch.isfinite(with_padding).all())
    assert bool((with_padding[64:] == 0).all())
    assert torch.allclose(with_padding[:64], vehicles_only)
    with pytest.raises(ValueError):
        fresh_mpc_student(teacher, device).predict(padded)


@pytest.mark.slow
@needs_teacher
def test_evaluator_runs_mpc_student(
        teacher: Teacher,
        distill_args: dict,
        device: str,
    ) -> None:
    """validation_gigaflow (with padded slots) runs the MPC student."""
    from pufferlib.distill.loop import DistillEvaluator

    student = fresh_mpc_student(teacher, device)
    metrics = DistillEvaluator(distill_args, num_scenarios=2).evaluate(
        student
    )
    assert np.isfinite(metrics["collision_rate"])


@pytest.mark.slow
@needs_teacher
@pytest.mark.skipif(not torch.cuda.is_available(), reason="tube needs CUDA")
def test_evaluation_keeps_training_plan_state(
        teacher: Teacher,
        distill_args: dict,
        device: str,
    ) -> None:
    """A tube student whose warm starts have a training row count other
    than the evaluator's evaluates, then gets its training state back."""
    from pufferlib.distill.loop import DistillEvaluator

    student = fresh_mpc_student(
        teacher,
        device,
        solver=SOLVER_WARPMPC,
        constraint_type=CONSTRAINT_TUBE,
        constraint_count=4,
    )
    training_row_count = 7
    student.allocate_plan_state(
        training_row_count, torch.float32, torch.device(device)
    )
    training_plans = student.plans
    metrics = DistillEvaluator(distill_args, num_scenarios=2).evaluate(
        student
    )
    assert np.isfinite(metrics["collision_rate"])
    assert student.plans is training_plans
    assert student.reference_valid.shape == (training_row_count,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory")
@needs_teacher
def test_training_steps_do_not_leak_gpu_memory(
        teacher: Teacher,
        device: str,
    ) -> None:
    """mpc.pytorch pins each solve's cost and trajectories unless
    bicycle.lqr_step_releasing_closures is active (~1 MiB per step at
    batch 1024 without it); with it, memory stays flat."""
    import gc

    student = fresh_mpc_student(teacher, device)
    obs = rollout_obs(teacher, 20, OBS_SEED)[:LEAK_BATCH_SIZE]
    labels = labels_for(teacher, obs)

    def step() -> None:
        """
        One forward and backward through the solver.
        """
        distillation_loss(student.predict(obs), labels).backward()

    step()
    gc.collect()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(LEAK_STEPS):
        step()
    gc.collect()
    torch.cuda.synchronize()
    growth_mib = (torch.cuda.memory_allocated() - allocated_before) / 2**20
    print(f"\n  growth over {LEAK_STEPS} steps: {growth_mib:.3f} MiB")
    assert growth_mib / LEAK_STEPS < MAX_GROWTH_PER_STEP_MIB


@needs_teacher
@pytest.mark.parametrize("student_class, config_class", [
    (MLPStudent, MLPStudentConfig),
    (MPCStudent, MPCStudentConfig),
])
def test_init_from_scratch_keeps_architecture_not_weights(
        teacher: Teacher,
        student_class: type,
        config_class: type,
    ) -> None:
    """init_from_teacher=False: same parameter shapes as the teacher's
    backbone, none of its weight matrices, and finite outputs."""
    torch.manual_seed(0)
    student = student_class(
        teacher, config_class(init_from_teacher=False)
    ).to(teacher.device)
    teacher_parameters = dict(teacher.policy.actor_backbone.named_parameters())
    for name, parameter in student.backbone.named_parameters():
        teacher_parameter = teacher_parameters[name]
        assert parameter.shape == teacher_parameter.shape
        assert parameter.requires_grad
        if parameter.dim() == 2:
            assert not torch.equal(parameter, teacher_parameter)
    obs = rollout_obs(teacher, 5, OBS_SEED)
    assert bool(torch.isfinite(student.predict(obs)).all())


@needs_teacher
@pytest.mark.parametrize("sizes, residual, linear_count", [
    ((64, 512, 6), False, 6),
    ((128, 1024, 4), False, 4),
    ((64, 1024, 8), True, 9),
])
def test_resized_backbone_has_the_requested_shape(
        teacher: Teacher,
        sizes: tuple[int, int, int],
        residual: bool,
        linear_count: int,
    ) -> None:
    """encoder_size / hidden_size / num_layers / residual build the
    requested network (residual: input layer plus num_layers in blocks)
    and it predicts finite controls."""
    encoder_size, hidden_size, num_layers = sizes
    config = MLPStudentConfig(
        init_from_teacher=False,
        encoder_size=encoder_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        residual=residual,
    )
    student = MLPStudent(teacher, config).to(teacher.device)
    main_mlp = student.backbone.backbone
    linears = [m for m in main_mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == linear_count
    assert all(m.out_features == hidden_size for m in linears)
    assert student.backbone.ego_encoder[0].out_features == encoder_size
    assert student.head.in_features == hidden_size
    obs = rollout_obs(teacher, 5, OBS_SEED)
    assert bool(torch.isfinite(student.predict(obs)).all())


@needs_teacher
@pytest.mark.skipif(
    not os.path.exists(SMALL_TEACHER_CHECKPOINT),
    reason="4x512 policy checkpoint not present",
)
@pytest.mark.parametrize("student_class, config_class", [
    (MLPStudent, MLPStudentConfig),
    (MPCStudent, MPCStudentConfig),
])
def test_backbone_checkpoint_loads_the_4x512_policy(
        teacher: Teacher,
        student_class: type,
        config_class: type,
    ) -> None:
    """backbone_checkpoint with the old 4x512 layout copies that policy's
    actor backbone weights exactly into both students."""
    config = config_class(
        init_from_teacher=False,
        encoder_size=64,
        hidden_size=512,
        num_layers=4,
        keep_layer_norm=False,
        backbone_checkpoint=SMALL_TEACHER_CHECKPOINT,
    )
    student = student_class(teacher, config).to(teacher.device)
    policy_state = torch.load(
        SMALL_TEACHER_CHECKPOINT, map_location="cpu", weights_only=False
    )
    student_state = student.backbone.state_dict()
    for name, value in student_state.items():
        loaded = policy_state[f"actor_backbone.{name}"]
        assert torch.equal(value.cpu(), loaded)
    obs = rollout_obs(teacher, 5, OBS_SEED)
    assert bool(torch.isfinite(student.predict(obs)).all())


@needs_teacher
@pytest.mark.parametrize("student_class, config_class", [
    (MLPStudent, MLPStudentConfig),
    (MPCStudent, MPCStudentConfig),
])
def test_old_structure_backbone_drops_layer_norm(
        teacher: Teacher,
        student_class: type,
        config_class: type,
    ) -> None:
    """keep_layer_norm=False with 64 / 512 / 4 builds the 4x512 teacher's
    main MLP (no LayerNorm) under both students, and they predict finite
    controls."""
    config = config_class(
        init_from_teacher=False,
        encoder_size=64,
        hidden_size=512,
        num_layers=4,
        keep_layer_norm=False,
    )
    student = student_class(teacher, config).to(teacher.device)
    main_mlp = student.backbone.backbone
    linears = [m for m in main_mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4
    assert all(m.out_features == 512 for m in linears)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in main_mlp)
    assert student.backbone.ego_encoder[0].out_features == 64
    assert student.head.in_features == 512
    obs = rollout_obs(teacher, 5, OBS_SEED)
    assert bool(torch.isfinite(student.predict(obs)).all())


@needs_teacher
def test_plan_head_executes_the_first_control(
        teacher: Teacher,
    ) -> None:
    """plan_steps=T: the head has 2 T outputs and predict returns the
    first (accel, steering_rate) pair, squashed into the box."""
    config = MLPStudentConfig(init_from_teacher=False, plan_steps=10)
    student = MLPStudent(teacher, config).to(teacher.device)
    assert student.head.out_features == 2 * 10
    obs = rollout_obs(teacher, 5, OBS_SEED)
    hidden = student.backbone(obs.float(), student.ego_dim)
    expected = torch.tanh(student.head(hidden)[:, :2]) * student.control_scale
    assert torch.allclose(student.predict(obs), expected)


@pytest.mark.parametrize("fields", [
    {"hidden_size": 1024},
    {"residual": True},
    {"init_from_teacher": False, "num_layers": -1},
    {"keep_layer_norm": False},
    {"backbone_checkpoint": "policy.pt"},
    {"plan_steps": 0},
])
def test_mlp_config_rejects_invalid_sizes(
        fields: dict,
    ) -> None:
    """Resizing needs fresh weights; sizes and plan length must be valid."""
    with pytest.raises(ValueError):
        MLPStudentConfig(**fields)


def poison_float32_rows(
        student: MPCStudent,
        rows: list[int],
        both_dtypes: bool = False,
    ) -> None:
    """
    Make the head-output gradient non-finite on some rows: a hook on the
    cost weights of float32 solves (and of float64 ones too with
    both_dtypes) writes NaN into those rows' gradient, after the solver's
    backward, as a failing adjoint would.

    Args:
        student (MPCStudent): patched in place.
        rows (list[int]): rows to poison (in the solve's row order).
        both_dtypes (bool): also poison the float64 retry.
    """
    original = student.cost_from_head_output

    def poisoned(raw, x0):
        q_diag, target = original(raw, x0)
        poisoned_dtype = both_dtypes or raw.dtype == torch.float32
        if q_diag.requires_grad and poisoned_dtype:
            poison_rows = [row for row in rows if row < q_diag.shape[1]]

            def hook(grad):
                grad = grad.clone()
                grad[:, poison_rows] = float("nan")
                return grad

            q_diag.register_hook(hook)
        return q_diag, target

    student.cost_from_head_output = poisoned


@needs_teacher
def test_non_finite_rows_are_solved_again_in_float64(
        teacher: Teacher,
    ) -> None:
    """A non-finite float32 gradient on one row is replaced by that row's
    float64 gradient, close to the clean float32 one; other rows keep
    theirs, and the retry is counted."""
    obs = moving_obs(teacher)[:8]
    labels = labels_for(teacher, obs)
    counted = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
    clean_student = fresh_mpc_student(teacher, teacher.device)
    _, clean_grad, _ = clean_student.solve_with_gradient(
        obs, labels, counted, 3, None
    )
    student = fresh_mpc_student(teacher, teacher.device)
    poison_float32_rows(student, [2])
    _, grad, _ = student.solve_with_gradient(obs, labels, counted, 3, None)
    assert bool(torch.isfinite(grad).all())
    assert student.pop_solver_stats() == {"solver/fp64_retry_rows": 1}
    assert torch.equal(grad[:2], clean_grad[:2])
    assert torch.allclose(grad[2], clean_grad[2], rtol=1e-3, atol=1e-6)


@needs_teacher
def test_rows_failing_in_float64_too_are_dumped_and_raise(
        teacher: Teacher,
        tmp_path,
    ) -> None:
    """Fail-fast stays: a row that fails in float64 as well raises, after
    its inputs were saved for diagnosis."""
    obs = moving_obs(teacher)[:8]
    labels = labels_for(teacher, obs)
    counted = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
    student = fresh_mpc_student(teacher, teacher.device)
    student.failure_dump_dir = str(tmp_path)
    poison_float32_rows(student, [0], both_dtypes=True)
    with pytest.raises(ValueError, match="non-finite"):
        student.solve_with_gradient(obs, labels, counted, 3, None)
    dump = torch.load(tmp_path / "failing_solve.pt", weights_only=False)
    assert dump["obs"].shape[0] == 1
    assert "student_state" in dump
