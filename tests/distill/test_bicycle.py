"""The torch classic bicycle must match PufferDrive's C step.

Run: pytest -s tests/distill/test_bicycle.py   (-s prints the error reports)

Env comparisons read the C state before and after each step: (x, y, heading, length) from
get_global_agent_state() and (speed, steering) from the ego observation. Global positions are float32
with the map offset added, so positions are compared as displacements with a tolerance in ulps of the
coordinate. Agents whose step was not a plain move_dynamics step are excluded: a terminal on the
previous step means the agent is stopped/removed (no motion), and a truncation means the env reset.
"""

import math
import os
import time

import numpy as np
import pytest
import torch
from mpc import mpc

from pufferlib.distill import env as distill_env
from pufferlib.distill.env import TEACHER_CHECKPOINT
from pufferlib.distill.bicycle import (
    ACCEL,
    ACCELERATION_VALUES,
    DT_SECONDS,
    EGO_SPEED_IDX,
    EGO_STEERING_IDX,
    HEADING,
    MAX_ACCEL_MPS2,
    MAX_SPEED_MPS,
    MAX_STEERING_RATE_RADPS,
    NU,
    NX,
    PX,
    PY,
    REAR_AXLE_RATIO,
    SPEED,
    STEERING,
    STEERING_LIMIT_RAD,
    STEERING_RATE,
    STEERING_VALUES,
    Bicycle,
    GradMethods,
    U_HI,
    control_from_discrete,
    env_action_from_control,
    mpc_forward_vectorized,
    solve,
    target_cost,
    upstream_mpc_forward,
    wheelbase_from_length,
)

ENV_AGENT_COUNT = 256
RANDOM_ACTION_STEPS = 600
# The 4 x 1024 teacher reverses rarely: 600 steps gave 360 reversing
# transitions, below MIN_REGIME_COUNT
TEACHER_STEPS = 1280
EPISODE_STEPS = 1280
DIVERGENCE_REPORT_STEPS = (1, 10, 100, 200, 400, 800, 1279)
MAX_HOLD_STEPS = 30
OVERSTEER_ACTION = 1.5  # beyond the Box: drive.py does not clip, so this reaches the C steering clip
CRUISE_MIN_ACCEL_ACTION = 0.3
CRUISE_MAX_STEER_ACTION = 0.05
CRUISE_HOLD_FACTOR = 4

JACOBIAN_SAMPLE_COUNT = 10000
JACOBIAN_TOL = 1e-10
FINITE_DIFF_EPS = 1e-6
FINITE_DIFF_TOL = 1e-6
KINK_MARGIN = 1e-4

STEERING_TOL_RAD = 1e-5
SPEED_TOL_MPS = 1e-5
HEADING_TOL_RAD = 1e-5
POSITION_ABS_TOL_M = 1e-5
POSITION_ULP_COUNT = 2
ROLLOUT_HORIZONS = (10, 20)
ROLLOUT_TOL_PER_STEP_M = 1e-5
ROLLOUT_TOL_PER_STEP_RAD = 1e-5
MIN_REGIME_FRACTION = 0.01
MIN_VALID_FRACTION = 0.5
MIN_REGIME_COUNT = 500

SOLVE_BATCH = 64
SOLVE_HORIZON = 3
SOLVE_TARGET_SCALE = 5.0
BOUND_ATOL = 1e-6
MPC_BATCH_SIZE = 8
MPC_HORIZON = 10
MPC_ITERS = 10
MPC_U_TOL = 1e-6
MPC_GRAD_TOL = 1e-5
TIMING_AGENT_COUNTS = (1024, 10240)


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def float32_ulp(values):
    return np.spacing(np.abs(values).astype(np.float32)).astype(np.float64)


def report(name, errors):
    errors = np.abs(np.asarray(errors, dtype=np.float64))
    print(f"  {name:<28} max {errors.max():.3e}   p99 {np.percentile(errors, 99):.3e}   n {errors.size}")


# ---------------------------------------------------------------- 1. hand-computed cases


def step(state, control, length_meters=4.5):
    x = torch.tensor([state], dtype=torch.float64)
    u = torch.tensor([control], dtype=torch.float64)
    wheelbase = wheelbase_from_length(torch.tensor([length_meters], dtype=torch.float64))
    return Bicycle(wheelbase)(x, u)[0].tolist()


def test_straight_line():
    px, py, heading, speed, steering = step([1.0, 2.0, 0.0, 10.0, 0.0], [2.0, 0.0])
    assert speed == pytest.approx(10.2)
    assert (px, py, heading, steering) == pytest.approx((1.0 + 1.02, 2.0, 0.0, 0.0))


def test_reversing_heading_north():
    px, py, heading, speed, _ = step([0.0, 0.0, math.pi / 2, -5.0, 0.0], [0.0, 0.0])
    assert speed == -5.0
    assert (px, py, heading) == pytest.approx((0.0, -0.5, math.pi / 2), abs=1e-15)


def test_standstill_steering_moves_nothing_but_steering():
    px, py, heading, speed, steering = step([3.0, 4.0, 0.7, 0.0, 0.3], [0.0, 0.6])
    assert (px, py, heading, speed) == (3.0, 4.0, 0.7, 0.0)
    assert steering == pytest.approx(0.36)


def test_steering_limit_clips():
    assert step([0, 0, 0, 5.0, 0.65], [0.0, 0.6])[STEERING] == STEERING_LIMIT_RAD
    assert step([0, 0, 0, 5.0, -0.65], [0.0, -0.6])[STEERING] == -STEERING_LIMIT_RAD


def test_speed_clip_both_signs():
    assert step([0, 0, 0, 39.9, 0.0], [4.0, 0.0])[SPEED] == MAX_SPEED_MPS
    assert step([0, 0, 0, -39.9, 0.0], [-4.0, 0.0])[SPEED] == -MAX_SPEED_MPS


def test_turn_uses_new_steering_and_old_heading():
    # Independent form: tan(beta) = r tan(delta) gives cos(beta) tan(delta) = sin(beta) / r
    length, speed0, accel, steering0, rate, heading0 = 4.5, 10.0, 1.0, 0.25, 0.5, 0.4
    new_speed, new_steering = speed0 + accel * DT_SECONDS, steering0 + rate * DT_SECONDS
    beta = math.atan(REAR_AXLE_RATIO * math.tan(new_steering))
    wheelbase = 0.6 * length
    expected = [
        new_speed * math.cos(heading0 + beta) * DT_SECONDS,
        new_speed * math.sin(heading0 + beta) * DT_SECONDS,
        heading0 + new_speed * math.sin(beta) / (REAR_AXLE_RATIO * wheelbase) * DT_SECONDS,
        new_speed,
        new_steering,
    ]
    assert step([0.0, 0.0, heading0, speed0, steering0], [accel, rate], length) == pytest.approx(expected, abs=1e-13)


def test_yaw_rate_scales_with_inverse_length():
    short_yaw = step([0, 0, 0, 10.0, 0.3], [0.0, 0.0], 3.0)[HEADING]
    long_yaw = step([0, 0, 0, 10.0, 0.3], [0.0, 0.0], 12.0)[HEADING]
    assert short_yaw == pytest.approx(4.0 * long_yaw)


def test_per_agent_wheelbase_tiles_over_time_major_batches():
    generator = torch.Generator().manual_seed(0)
    agent_count, step_count = 7, 4
    wheelbase = wheelbase_from_length(torch.rand(agent_count, generator=generator, dtype=torch.float64) * 10 + 3)
    x = torch.rand(step_count, agent_count, NX, generator=generator, dtype=torch.float64)
    u = torch.rand(step_count, agent_count, NU, generator=generator, dtype=torch.float64)
    dynamics = Bicycle(wheelbase)
    flat = dynamics(x.view(-1, NX), u.view(-1, NU)).view(step_count, agent_count, NX)
    per_step = torch.stack([dynamics(x[t], u[t]) for t in range(step_count)])
    assert torch.equal(flat, per_step)


def test_labels_reproduce_discrete_steering_change():
    steering0 = torch.tensor([0.0, 0.0, 0.62, -0.3, 0.12], dtype=torch.float64)
    action_idx = torch.tensor([8, 4 * 9 + 0, 6 * 9 + 8, 3 * 9 + 4, 2 * 9 + 5])  # steer targets 0.667, -0.667, 0.667, 0, 0.167
    labels = control_from_discrete(action_idx, steering0)
    assert labels[:, ACCEL].tolist() == [ACCELERATION_VALUES[0], ACCELERATION_VALUES[4], ACCELERATION_VALUES[6], ACCELERATION_VALUES[3], ACCELERATION_VALUES[2]]
    # drive.h: steering += clip(target - steering, +-0.6 dt), then clip to the limit
    expected_rate = [0.6, -0.6, (0.667 - 0.62) / DT_SECONDS, 0.6, (0.167 - 0.12) / DT_SECONDS]
    assert labels[:, STEERING_RATE].tolist() == pytest.approx(expected_rate)
    action = env_action_from_control(labels, steering0)
    assert action.abs().max() <= 1.0
    expected_target = steering0 + torch.tensor(expected_rate, dtype=torch.float64) * DT_SECONDS
    assert (action[:, 1] * STEERING_LIMIT_RAD).tolist() == pytest.approx(expected_target.tolist())


# ---------------------------------------------------------------- 2. Jacobians


def random_states_and_controls(sample_count, generator):
    def uniform(low, high):
        return low + (high - low) * torch.rand(sample_count, generator=generator, dtype=torch.float64)

    x = torch.stack(
        [uniform(-100, 100), uniform(-100, 100), uniform(-math.pi, math.pi), uniform(-41, 41), uniform(-STEERING_LIMIT_RAD, STEERING_LIMIT_RAD)], -1
    )
    u = torch.stack([uniform(-MAX_ACCEL_MPS2, MAX_ACCEL_MPS2), uniform(-MAX_STEERING_RATE_RADPS, MAX_STEERING_RATE_RADPS)], -1)
    wheelbase = wheelbase_from_length(uniform(3, 15))
    raw_speed = x[:, SPEED] + u[:, ACCEL] * DT_SECONDS
    raw_steering = x[:, STEERING] + u[:, STEERING_RATE] * DT_SECONDS
    away_from_kinks = ((raw_speed.abs() - MAX_SPEED_MPS).abs() > KINK_MARGIN) & ((raw_steering.abs() - STEERING_LIMIT_RAD).abs() > KINK_MARGIN)
    speed_clipped = raw_speed.abs() > MAX_SPEED_MPS
    steering_clipped = raw_steering.abs() > STEERING_LIMIT_RAD
    keep = away_from_kinks
    return x[keep], u[keep], wheelbase[keep], speed_clipped[keep], steering_clipped[keep]


def autograd_jacobians(dynamics, x, u):
    """Rows are independent, so d(sum over batch of output j) gives row j for every sample."""
    x = x.clone().requires_grad_(True)
    u = u.clone().requires_grad_(True)
    next_x = dynamics(x, u)
    rows_A, rows_B = [], []
    for output_idx in range(NX):
        grad_x, grad_u = torch.autograd.grad(next_x[:, output_idx].sum(), [x, u], retain_graph=True)
        rows_A.append(grad_x)
        rows_B.append(grad_u)
    return torch.stack(rows_A, 1), torch.stack(rows_B, 1)


def test_jacobians_match_autograd_and_finite_differences():
    generator = torch.Generator().manual_seed(1)
    x, u, wheelbase, speed_clipped, steering_clipped = random_states_and_controls(JACOBIAN_SAMPLE_COUNT, generator)
    assert speed_clipped.float().mean() > MIN_REGIME_FRACTION, "sample never hits the speed clip"
    assert steering_clipped.float().mean() > MIN_REGIME_FRACTION, "sample never hits the steering clip"
    dynamics = Bicycle(wheelbase)
    A, B = dynamics.grad_input(x, u)
    A_autograd, B_autograd = autograd_jacobians(dynamics, x, u)
    print()
    report("A vs autograd", (A - A_autograd).flatten())
    report("B vs autograd", (B - B_autograd).flatten())
    assert (A - A_autograd).abs().max() < JACOBIAN_TOL
    assert (B - B_autograd).abs().max() < JACOBIAN_TOL

    tau = torch.cat([x, u], -1)
    analytic = torch.cat([A, B], -1)
    for input_idx in range(NX + NU):
        offset = torch.zeros_like(tau)
        offset[:, input_idx] = FINITE_DIFF_EPS
        plus, minus = tau + offset, tau - offset
        finite_diff = (dynamics(plus[:, :NX], plus[:, NX:]) - dynamics(minus[:, :NX], minus[:, NX:])) / (2 * FINITE_DIFF_EPS)
        column_error = (finite_diff - analytic[:, :, input_idx]).abs().max()
        assert column_error < FINITE_DIFF_TOL, f"input {input_idx}: finite-difference error {column_error:.2e}"


# ---------------------------------------------------------------- 2b. vectorized MPC.forward


def solve_with_gradients(forward, dtype, warm_start, seed):
    """Solve a random batch with the given MPC.forward; return x, u and grads of a loss."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(shape, low, high):
        return (low + (high - low) * torch.rand(shape, generator=generator, dtype=torch.float64)).to(dtype)

    x0 = torch.stack(
        [torch.zeros(SOLVE_BATCH, dtype=dtype), torch.zeros(SOLVE_BATCH, dtype=dtype), torch.zeros(SOLVE_BATCH, dtype=dtype),
         uniform(SOLVE_BATCH, 0, 20), uniform(SOLVE_BATCH, -0.5, 0.5)], -1
    )
    wheelbase = wheelbase_from_length(uniform(SOLVE_BATCH, 3, 15))
    q_diag = uniform((SOLVE_HORIZON, SOLVE_BATCH, NX + NU), 0.1, 10).requires_grad_(True)
    # Targets far beyond the control box so many controls sit on a bound
    target = uniform((SOLVE_HORIZON, SOLVE_BATCH, NX + NU), -SOLVE_TARGET_SCALE, SOLVE_TARGET_SCALE).requires_grad_(True)
    u_init = uniform((SOLVE_HORIZON, SOLVE_BATCH, NU), -0.3, 0.3) if warm_start else None
    Q, p = target_cost(q_diag, target)
    original_forward = mpc.MPC.forward
    mpc.MPC.forward = forward
    try:
        x, u = solve(x0, Q, p, Bicycle(wheelbase), SOLVE_HORIZON, u_init=u_init)
    finally:
        mpc.MPC.forward = original_forward
    (u[0].square().sum() + x[-1].square().sum()).backward()
    return x.detach(), u.detach(), q_diag.grad, target.grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("warm_start", [False, True])
def test_vectorized_mpc_forward_matches_upstream(dtype, warm_start):
    upstream = solve_with_gradients(upstream_mpc_forward, dtype, warm_start, seed=3)
    patched = solve_with_gradients(mpc_forward_vectorized, dtype, warm_start, seed=3)
    on_bound = (upstream[1][0].abs() >= U_HI.to(dtype) - BOUND_ATOL).float().mean()
    assert on_bound > MIN_REGIME_FRACTION, "no control sits on a bound"
    for name, expected, actual in zip(("x", "u", "dq", "dtarget"), upstream, patched):
        torch.testing.assert_close(actual, expected, msg=lambda message: f"{name}: {message}")


# ---------------------------------------------------------------- 3-6. against the C env


def make_env(action_type, seed, agent_count=ENV_AGENT_COUNT):
    return distill_env.make_env(action_type, seed, agent_count)


def read_state(env, obs):
    """Float64 (x, y, heading, speed, steering) and length per active agent."""
    global_state = env.get_global_agent_state()
    state = np.stack(
        [
            global_state["x"].astype(np.float64),
            global_state["y"].astype(np.float64),
            global_state["heading"].astype(np.float64),
            obs[:, EGO_SPEED_IDX].astype(np.float64) * MAX_SPEED_MPS,
            obs[:, EGO_STEERING_IDX].astype(np.float64) * STEERING_LIMIT_RAD,
        ],
        -1,
    )
    return state, global_state["length"].astype(np.float64)


def controls_from_env_action(action, steering0):
    """Mirror of drive.h's scaling: float32 products, then the rate limit expressed as a rate."""
    accel = (action[:, 0].astype(np.float32) * np.float32(MAX_ACCEL_MPS2)).astype(np.float64)
    target_steering = (action[:, 1].astype(np.float32) * np.float32(STEERING_LIMIT_RAD)).astype(np.float64)
    rate = np.clip((target_steering - steering0) / DT_SECONDS, -MAX_STEERING_RATE_RADPS, MAX_STEERING_RATE_RADPS)
    return np.stack([accel, rate], -1)


def run_and_record(env, seed, step_count, choose_action):
    """Steps the env and returns per-step arrays (step, agent, ...) of states, controls and validity."""
    obs, _ = env.reset(seed=seed)
    states, lengths, controls, valid = [], [], [], []
    previous_terminal = np.zeros(env.num_agents, dtype=bool)
    state, length = read_state(env, obs)
    for step_idx in range(step_count):
        action = choose_action(obs, state, step_idx)
        controls.append(controls_from_env_action(action, state[:, STEERING]))
        states.append(state)
        lengths.append(length)
        obs, _, terminals, truncations, _ = env.step(action)
        valid.append(~previous_terminal & ~truncations.astype(bool))
        previous_terminal = terminals.astype(bool)
        state, length = read_state(env, obs)
    states.append(state)
    lengths.append(length)
    return {
        "state": np.stack(states),
        "length": np.stack(lengths),
        "control": np.stack(controls),
        "valid": np.stack(valid),
    }


def rollout_model(start_state, controls, length):
    """Open-loop model rollout from start_state (N,5) with controls (T,N,2). Positions relative to start."""
    wheelbase = wheelbase_from_length(torch.from_numpy(length))
    dynamics = Bicycle(wheelbase)
    x = torch.from_numpy(start_state.copy())
    x[:, PX] = 0.0
    x[:, PY] = 0.0
    trajectory = []
    for control in controls:
        x = dynamics(x, torch.from_numpy(control))
        trajectory.append(x)
    return torch.stack(trajectory).numpy()


def pd_speed_sign_flips(before, predicted):
    """drive.h update_agent_speed signs the speed by velocity . new heading. When one step turns the heading
    more than 90 degrees past the direction of travel (sub-meter wheelbases at full lock), PD flips the
    sign of the speed. Not modeled: it is a PD artifact, not vehicle motion."""
    beta = np.arctan(REAR_AXLE_RATIO * np.tan(predicted[..., STEERING]))
    course_minus_new_heading = before[..., HEADING] + beta - predicted[..., HEADING]
    return (np.cos(course_minus_new_heading) < 0) & (predicted[..., SPEED] != 0)


def one_step_flips(record):
    """(step, agent) mask of transitions where PD flips the speed sign."""
    step_count, agent_count, _ = record["control"].shape
    before = record["state"][:-1].reshape(-1, NX)
    predicted = rollout_model(before, record["control"].reshape(1, -1, NU), record["length"][:-1].reshape(-1))[0]
    return pd_speed_sign_flips(before, predicted).reshape(step_count, agent_count)


def compare_one_step(record, label):
    valid = record["valid"]
    assert valid.mean() > MIN_VALID_FRACTION, f"only {valid.mean():.2f} of transitions are plain steps"
    before, after = record["state"][:-1][valid], record["state"][1:][valid]
    controls = record["control"][valid]
    predicted = rollout_model(before, controls[None], record["length"][:-1][valid])[0]
    flips = pd_speed_sign_flips(before, predicted)
    predicted[flips, SPEED] *= -1

    position_tol_x = POSITION_ULP_COUNT * np.maximum(float32_ulp(before[:, PX]), float32_ulp(after[:, PX])) + POSITION_ABS_TOL_M
    position_tol_y = POSITION_ULP_COUNT * np.maximum(float32_ulp(before[:, PY]), float32_ulp(after[:, PY])) + POSITION_ABS_TOL_M
    errors = {
        "dx": predicted[:, PX] - (after[:, PX] - before[:, PX]),
        "dy": predicted[:, PY] - (after[:, PY] - before[:, PY]),
        "heading": wrap_angle(predicted[:, HEADING] - after[:, HEADING]),
        "speed": predicted[:, SPEED] - after[:, SPEED],
        "steering": predicted[:, STEERING] - after[:, STEERING],
    }
    print(f"\n[{label}] one step, {valid.sum()} transitions ({(~valid).sum()} excluded, {flips.sum()} PD speed-sign flips)")
    for name, error in errors.items():
        report(name, error)
    for name, tol in (("dx", position_tol_x), ("dy", position_tol_y), ("heading", HEADING_TOL_RAD), ("speed", SPEED_TOL_MPS), ("steering", STEERING_TOL_RAD)):
        worst_idx = np.argmax(np.abs(errors[name]) - tol)
        assert np.all(np.abs(errors[name]) <= tol), f"{name}: error {errors[name][worst_idx]:.3e} at state {before[worst_idx]}, control {controls[worst_idx]}"
    return before, controls


def compare_rollouts(record, label):
    plain_steps = record["valid"] & ~one_step_flips(record)
    print(f"\n[{label}] open-loop rollouts (windows with a PD speed-sign flip excluded)")
    for horizon in ROLLOUT_HORIZONS:
        step_count = record["control"].shape[0]
        starts = range(0, step_count - horizon, horizon)
        position_errors, heading_errors = [[] for _ in range(horizon)], [[] for _ in range(horizon)]
        for start in starts:
            window_valid = plain_steps[start : start + horizon].all(0)
            if not window_valid.any():
                continue
            begin = record["state"][start][window_valid]
            truth = record["state"][start + 1 : start + horizon + 1][:, window_valid]
            predicted = rollout_model(begin, record["control"][start : start + horizon][:, window_valid], record["length"][start][window_valid])
            for t in range(horizon):
                displacement_error = np.hypot(predicted[t, :, PX] - (truth[t, :, PX] - begin[:, PX]), predicted[t, :, PY] - (truth[t, :, PY] - begin[:, PY]))
                # drive.h rounds sim_x/sim_y to float32 once per step, so the allowance grows by one ulp per step
                coordinate_magnitude = np.maximum.reduce([np.abs(begin[:, PX]), np.abs(begin[:, PY]), np.abs(truth[t, :, PX]), np.abs(truth[t, :, PY])])
                position_errors[t].append(displacement_error - (POSITION_ULP_COUNT + t + 1) * float32_ulp(coordinate_magnitude))
                heading_errors[t].append(np.abs(wrap_angle(predicted[t, :, HEADING] - truth[t, :, HEADING])))
        position_errors = [np.concatenate(e) for e in position_errors]
        heading_errors = [np.concatenate(e) for e in heading_errors]
        print(f"  T={horizon}: {position_errors[0].size} windows")
        for t in (0, horizon // 2 - 1, horizon - 1):
            print(f"    t={t + 1:>2}  position excess over (2 + t) ulp: max {position_errors[t].max():.3e}   heading max {heading_errors[t].max():.3e}")
        for t in range(horizon):
            assert position_errors[t].max() <= ROLLOUT_TOL_PER_STEP_M * (t + 1), f"T={horizon} t={t + 1}: position error {position_errors[t].max():.3e}"
            assert heading_errors[t].max() <= ROLLOUT_TOL_PER_STEP_RAD * (t + 1), f"T={horizon} t={t + 1}: heading error {heading_errors[t].max():.3e}"


def print_and_check_regimes(before, controls, label, require_steering_clip):
    raw_steering = before[:, STEERING] + controls[:, STEERING_RATE] * DT_SECONDS
    regimes = {
        "rate limit": np.abs(np.abs(controls[:, STEERING_RATE]) - MAX_STEERING_RATE_RADPS) < 1e-9,
        "steering clip": np.abs(raw_steering) > STEERING_LIMIT_RAD,
        "reversing": before[:, SPEED] < -0.1,
        "speed > 10 m/s": before[:, SPEED] > 10.0,
    }
    print(f"  regimes [{label}]: " + ", ".join(f"{name} {mask.sum()} ({mask.mean():.3f})" for name, mask in regimes.items()))
    required = ["rate limit", "reversing", "speed > 10 m/s"] + (["steering clip"] if require_steering_clip else [])
    for name in required:
        assert regimes[name].sum() >= MIN_REGIME_COUNT, f"{name} occurs in only {regimes[name].sum()} transitions"


def held_random_actions(agent_count, seed):
    """Each agent holds an action for a random number of steps: uniform, extremes, beyond-lock steering,
    or cruise (sustained throttle, small steering, long holds) so high speeds occur too."""
    rng = np.random.default_rng(seed)
    current = np.zeros((agent_count, 2), dtype=np.float32)
    hold_remaining = np.zeros(agent_count, dtype=np.int64)

    def choose_action(obs, state, step_idx):
        renew = hold_remaining <= 0
        renew_count = int(renew.sum())
        mode = rng.integers(0, 4, renew_count)[:, None]
        uniform = rng.uniform(-1, 1, (renew_count, 2))
        extreme = rng.choice([-1.0, 1.0], (renew_count, 2))
        overshoot = np.stack([rng.uniform(-1, 1, renew_count), rng.choice([-OVERSTEER_ACTION, OVERSTEER_ACTION], renew_count)], -1)
        cruise = np.stack([rng.uniform(CRUISE_MIN_ACCEL_ACTION, 1, renew_count), rng.uniform(-CRUISE_MAX_STEER_ACTION, CRUISE_MAX_STEER_ACTION, renew_count)], -1)
        current[renew] = np.select([mode == 0, mode == 1, mode == 2], [uniform, extreme, overshoot], cruise)
        hold_steps = rng.integers(1, MAX_HOLD_STEPS, renew_count)
        hold_remaining[renew] = np.where(mode[:, 0] == 3, CRUISE_HOLD_FACTOR * hold_steps, hold_steps)
        hold_remaining[:] -= 1
        return current.copy()

    return choose_action


@pytest.fixture(scope="module")
def random_action_record():
    env, _ = make_env("continuous", seed=7)
    record = run_and_record(env, 7, RANDOM_ACTION_STEPS, held_random_actions(env.num_agents, seed=7))
    env.close()
    return record


def test_one_step_matches_env_random_actions(random_action_record):
    before, controls = compare_one_step(random_action_record, "random actions")
    print_and_check_regimes(before, controls, "random actions", require_steering_clip=True)


def test_rollouts_match_env_random_actions(random_action_record):
    compare_rollouts(random_action_record, "random actions")


def load_teacher(args, discrete_env, device):
    from pufferlib.ocean import torch as ocean_torch
    from pufferlib.pufferl import clean_policy_state_dict

    policy = getattr(ocean_torch, args["policy_name"])(discrete_env, **args["policy"]).to(device)
    policy.load_state_dict(clean_policy_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=device)))
    return policy.eval()


def teacher_argmax(policy, obs, device):
    with torch.no_grad():
        logits, _ = policy.forward_eval(torch.from_numpy(obs).to(device))
    return logits[0].argmax(-1).cpu()


needs_teacher = pytest.mark.skipif(not os.path.exists(TEACHER_CHECKPOINT), reason="teacher checkpoint not found")


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def teacher_record(device):
    discrete_env, args = make_env("discrete", seed=11, agent_count=ENV_AGENT_COUNT)
    policy = load_teacher(args, discrete_env, device)
    discrete_env.close()
    env, _ = make_env("continuous", seed=11)

    action_indices = []

    def choose_action(obs, state, step_idx):
        steering0 = torch.from_numpy(state[:, STEERING])
        action_idx = teacher_argmax(policy, obs, device)
        action_indices.append(action_idx.numpy())
        return env_action_from_control(control_from_discrete(action_idx, steering0), steering0).numpy().astype(np.float32)

    record = run_and_record(env, 11, TEACHER_STEPS, choose_action)
    record["action_idx"] = np.stack(action_indices)
    env.close()
    return record


@needs_teacher
def test_one_step_matches_env_teacher(teacher_record):
    before, controls = compare_one_step(teacher_record, "teacher")
    print_and_check_regimes(before, controls, "teacher", require_steering_clip=False)


@needs_teacher
def test_rollouts_match_env_teacher(teacher_record):
    compare_rollouts(teacher_record, "teacher")


def discrete_rule_controls(action_idx, steering0):
    """drive.h's discrete step written out independently of bicycle.py: accel from the table, steering
    moves towards the table target by at most 0.6 rad/s * dt, then the limit clip. Returned as (a, rate)."""
    accel = np.asarray(ACCELERATION_VALUES)[action_idx // len(STEERING_VALUES)]
    target = np.asarray(STEERING_VALUES)[action_idx % len(STEERING_VALUES)]
    max_change = MAX_STEERING_RATE_RADPS * DT_SECONDS
    next_steering = np.clip(steering0 + np.clip(target - steering0, -max_change, max_change), -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD)
    return np.stack([accel, (next_steering - steering0) / DT_SECONDS], -1)


@needs_teacher
def test_decoded_labels_follow_discrete_rule(teacher_record):
    """The continuous env, driven by decoded labels, moves as drive.h's discrete rule says for the index."""
    rule_record = dict(teacher_record, control=discrete_rule_controls(teacher_record["action_idx"], teacher_record["state"][:-1, :, STEERING]))
    compare_one_step(rule_record, "discrete rule vs continuous env")


@needs_teacher
def test_decoded_actions_match_discrete_table(teacher_record):
    """Action level, as C decodes it (float32 action * float32 scale): accel is bit-exact; a steering target
    the rate limit does not cut is within 1 ulp of the table (target/0.667 does not round-trip exactly)."""
    action_idx = teacher_record["action_idx"]
    steering0 = teacher_record["state"][:-1, :, STEERING]
    action = env_action_from_control(
        torch.from_numpy(discrete_rule_controls(action_idx, steering0)).view(-1, NU), torch.from_numpy(steering0).view(-1)
    ).numpy().astype(np.float32)
    accel_table = np.asarray(ACCELERATION_VALUES, dtype=np.float32)[action_idx.reshape(-1) // len(STEERING_VALUES)]
    steering_table = np.asarray(STEERING_VALUES, dtype=np.float32)[action_idx.reshape(-1) % len(STEERING_VALUES)]
    decoded_accel = action[:, 0] * np.float32(MAX_ACCEL_MPS2)
    decoded_steering = action[:, 1] * np.float32(STEERING_LIMIT_RAD)
    reachable = np.abs(steering_table - steering0.reshape(-1)) < MAX_STEERING_RATE_RADPS * DT_SECONDS - KINK_MARGIN
    steering_ulps = np.abs(decoded_steering - steering_table)[reachable] / np.spacing(np.maximum(np.abs(steering_table[reachable]), np.float32(1e-3)))
    print(f"\n[decoded actions] accel max |error| {np.abs(decoded_accel - accel_table).max():.1e}, steering max {steering_ulps.max():.1f} ulp over {reachable.sum()} unclipped targets")
    assert np.array_equal(decoded_accel, accel_table)
    assert steering_ulps.max() <= 1.0


@needs_teacher
def test_report_discrete_vs_continuous_divergence(device):
    """Reported, not asserted: same seed, discrete env with the index and continuous env with the decoded
    label. 1-ulp steering differences grow chaotically; check_teacher.py checks equivalence at the metrics level."""
    seed = 13
    discrete_env, args = make_env("discrete", seed=seed)
    continuous_env, _ = make_env("continuous", seed=seed)
    policy = load_teacher(args, discrete_env, device)
    discrete_obs, _ = discrete_env.reset(seed=seed)
    continuous_obs, _ = continuous_env.reset(seed=seed)
    print("\n[discrete vs continuous env] max |difference| (x, y, heading, speed, steering)")
    for step_idx in range(EPISODE_STEPS):
        discrete_state, _ = read_state(discrete_env, discrete_obs)
        continuous_state, _ = read_state(continuous_env, continuous_obs)
        state_error = np.abs(discrete_state - continuous_state)
        state_error[:, HEADING] = np.abs(wrap_angle(discrete_state[:, HEADING] - continuous_state[:, HEADING]))
        if step_idx in DIVERGENCE_REPORT_STEPS:
            print(f"  step {step_idx:>4}: {np.array2string(state_error.max(0), precision=2)}  agents differing {(state_error.max(1) > 0).sum()}")
        action_idx = teacher_argmax(policy, discrete_obs, device)
        steering0 = torch.from_numpy(continuous_state[:, STEERING])
        continuous_action = env_action_from_control(control_from_discrete(action_idx, steering0), steering0).numpy().astype(np.float32)
        discrete_obs, _, _, discrete_truncations, _ = discrete_env.step(action_idx.numpy().astype(discrete_env.actions.dtype).reshape(discrete_env.actions.shape))
        continuous_obs, _, _, continuous_truncations, _ = continuous_env.step(continuous_action)
        if not np.array_equal(discrete_truncations, continuous_truncations):
            print(f"  step {step_idx:>4}: resets diverged")
            break
    discrete_env.close()
    continuous_env.close()


# ---------------------------------------------------------------- 7. mpc.pytorch


def random_mpc_problem(batch_size, horizon, generator, device="cpu"):
    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float64).to(device)

    x0 = torch.zeros(batch_size, NX, dtype=torch.float64, device=device)
    x0[:, SPEED] = rand(batch_size) * 15
    x0[:, STEERING] = (rand(batch_size) - 0.5) * 0.6
    q_diag = torch.cat([rand(horizon, batch_size, NX) * 5, 0.1 + rand(horizon, batch_size, NU)], -1)
    target = torch.zeros(horizon, batch_size, NX + NU, dtype=torch.float64, device=device)
    target[:, :, PY] = (rand(1, batch_size) - 0.5) * 6
    target[:, :, SPEED] = x0[:, SPEED] + (rand(1, batch_size) - 0.5) * 6
    wheelbase = wheelbase_from_length(3 + rand(batch_size) * 8)
    return x0, q_diag, target, wheelbase


def test_mpc_analytic_matches_autodiff():
    generator = torch.Generator().manual_seed(3)
    x0, q_diag, target, wheelbase = random_mpc_problem(MPC_BATCH_SIZE, MPC_HORIZON, generator)
    results = {}
    for method in (GradMethods.ANALYTIC, GradMethods.AUTO_DIFF):
        q = q_diag.clone().requires_grad_(True)
        r = target.clone().requires_grad_(True)
        Q, p = target_cost(q, r)
        _, u = solve(x0, Q, p, Bicycle(wheelbase), MPC_HORIZON, iters=MPC_ITERS, method=method)
        loss = (u[0] ** 2).sum()
        grad_q, grad_r = torch.autograd.grad(loss, [q, r])
        results[method] = (u.detach(), grad_q, grad_r)
    (u_analytic, grad_q_analytic, grad_r_analytic), (u_auto, grad_q_auto, grad_r_auto) = results.values()
    print()
    report("u analytic vs autodiff", (u_analytic - u_auto).flatten())
    report("dL/dq analytic vs autodiff", (grad_q_analytic - grad_q_auto).flatten())
    report("dL/dtarget analytic vs autodiff", (grad_r_analytic - grad_r_auto).flatten())
    assert (u_analytic - u_auto).abs().max() < MPC_U_TOL
    assert (grad_q_analytic - grad_q_auto).abs().max() < MPC_GRAD_TOL
    assert (grad_r_analytic - grad_r_auto).abs().max() < MPC_GRAD_TOL
    assert grad_r_analytic.abs().max() > 0, "no gradient reached the targets"


# ---------------------------------------------------------------- 8. timing (reported only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing is for the GPU")
def test_report_gpu_timing():
    generator = torch.Generator().manual_seed(4)
    print()
    for agent_count in TIMING_AGENT_COUNTS:
        x0, q_diag, target, wheelbase = random_mpc_problem(agent_count, MPC_HORIZON, generator, "cuda")
        x0, q_diag, target, wheelbase = x0.float(), q_diag.float(), target.float(), wheelbase.float()
        dynamics = Bicycle(wheelbase)
        u = torch.zeros(agent_count, NU, device="cuda")
        for _ in range(3):
            dynamics.grad_input(x0, u)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            dynamics(x0, u)
            dynamics.grad_input(x0, u)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - start) * 10
        for iters in (1, 2, 5):
            Q, p = target_cost(q_diag, target)
            solve(x0, Q, p, dynamics, MPC_HORIZON, iters=iters)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                solve(x0, Q, p, dynamics, MPC_HORIZON, iters=iters)
            torch.cuda.synchronize()
            solve_ms = (time.perf_counter() - start) / 3 * 1000
            print(f"  {agent_count:>6} agents  forward+grad_input {step_ms:6.3f} ms   solve T={MPC_HORIZON} iters={iters}: {solve_ms:8.1f} ms (fp32)")
