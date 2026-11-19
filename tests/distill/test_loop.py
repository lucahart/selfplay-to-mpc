"""The BC/DAgger loop: who drives, who gets labeled, and that the student
learns, reproducibly."""

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from pufferlib.distill.bicycle import (
    MAX_ACCEL_MPS2,
    STEERING,
    ego_state_from_obs,
    env_action_from_control,
)
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.env import NUPLAN_SCENARIO_STEPS, TEACHER_CHECKPOINT
from pufferlib.distill.loop import (
    STOP_FILE_NAME,
    BETA_DAGGER,
    DistillConfig,
    DistillEvaluator,
    EpisodeState,
    add_episode_sums,
    beta_at,
    distill,
    distillation_loss,
    load_checkpoint,
    make_optimizer,
    pop_episode_stats,
    ForgettingProbe,
    saturation_stats,
    stage_at,
    train_student,
)
from pufferlib.distill.students import MLPStudent, MPCStudent
from pufferlib.distill.teacher import Teacher

from conftest import (
    SPY_BRAKE_CONTROL,
    SPY_CRASH_CONTROL,
    RecordingBuffer,
    RecordingEnv,
    SpyStudent,
    make_small_env,
    needs_teacher,
)

LOOP_SEED = 51
SHORT_SCENARIO_STEPS = 30
RESAMPLE_STEPS = 20
# Spy brake accel as the env action the loop sends (exact in float32)
SPY_BRAKE_ACTION = SPY_BRAKE_CONTROL[0] / MAX_ACCEL_MPS2
BINOMIAL_SIGMAS = 4
# Most steering-rate labels sit on the box edge, which tanh only
# approaches, so overfitting 640 rows takes ~2000 steps
OVERFIT_STEPS = 2000
OVERFIT_BATCH_SIZE = 256
OVERFIT_LOSS_RATIO = 0.1
TIMING_AGENT_COUNT = 1024


def small_config(
        device: str,
        **overrides,
    ) -> DistillConfig:
    """
    Fast settings for loop tests.

    Args:
        device (str): torch device.
        **overrides: DistillConfig fields to change.

    Returns:
        DistillConfig: the config.
    """
    settings = dict(
        seed=LOOP_SEED,
        device=device,
        iterations=2,
        env_steps_per_iteration=50,
        gradient_steps_per_iteration=2,
        batch_size=64,
        buffer_size=100_000,
        eval_interval_labels=0,
    )
    settings.update(overrides)
    return DistillConfig(**settings)


def run_recorded(
        teacher: Teacher,
        student: torch.nn.Module,
        config: DistillConfig,
        extra_overrides: tuple[str, ...] = (),
    ) -> tuple[RecordingEnv, RecordingBuffer]:
    """
    Run distill on a small recording env with a recording buffer.

    Args:
        teacher (Teacher): labeler.
        student (torch.nn.Module): student on config.device.
        config (DistillConfig): run settings.
        extra_overrides (tuple[str, ...]): Hydra env overrides.

    Returns:
        tuple[RecordingEnv, RecordingBuffer]: the recordings.
    """
    env = RecordingEnv(make_small_env(config.seed, extra_overrides))
    obs_dim = env.env.single_observation_space.shape[0]
    buffer = RecordingBuffer(config.buffer_size, obs_dim, config.device)
    distill(env, teacher, student, buffer, config)
    env.env.close()
    return env, buffer


def spy_drove(
        env: RecordingEnv,
    ) -> np.ndarray:
    """
    Which agent the spy student drove at each step.

    Args:
        env (RecordingEnv): recording of a run with a SpyStudent braking.

    Returns:
        np.ndarray: (steps,B) bool.
    """
    return np.stack(env.actions)[:, :, 0] == np.float32(SPY_BRAKE_ACTION)


@needs_teacher
def test_bc_never_calls_student_act(
        teacher: Teacher,
        device: str,
    ) -> None:
    """beta = 1: the student never drives; actions are decoded labels."""
    spy = SpyStudent().to(device)
    env, _ = run_recorded(teacher, spy, small_config(device))
    assert spy.act_calls == 0
    for obs, action in zip(env.observations, env.actions):
        obs_tensor = torch.from_numpy(obs).to(device)
        steering = ego_state_from_obs(obs_tensor)[0][:, STEERING]
        expected = env_action_from_control(
            teacher.label(obs_tensor, steering), steering
        )
        assert np.array_equal(action, expected.cpu().numpy())


@needs_teacher
def test_only_live_agents_are_labeled(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Rows after a terminal are dropped until the agent's env resets."""
    spy = SpyStudent(SPY_CRASH_CONTROL).to(device)
    config = small_config(device, beta_constant=0.0, iterations=3)
    env, buffer = run_recorded(teacher, spy, config)
    terminals = np.stack(env.terminals)
    assert terminals.any(), "no infractions; the test would be vacuous"

    live = np.ones(env.num_agents, dtype=bool)
    new_episodes = env.new_episode_masks()
    for step_idx, added_obs in enumerate(buffer.added_obs):
        expected_obs = env.observations[step_idx][live]
        assert np.array_equal(added_obs.numpy(), expected_obs)
        live = ~terminals[step_idx] | new_episodes[step_idx]
    dropped = sum(
        env.num_agents - len(added) for added in buffer.added_obs
    )
    assert dropped > 0


@needs_teacher
def test_beta_fixed_per_episode_and_redrawn_on_new_episode(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Driver constant within an episode, redrawn only at new episodes."""
    spy = SpyStudent().to(device)
    config = small_config(
        device, beta_constant=0.5, iterations=3, env_steps_per_iteration=40
    )
    env, _ = run_recorded(
        teacher,
        spy,
        config,
        (f"env.scenario_length={SHORT_SCENARIO_STEPS}",),
    )
    drove = spy_drove(env)
    new_episodes = env.new_episode_masks()
    assert any(mask.any() for mask in new_episodes[:-1])
    draws = [drove[0]]
    for step_idx in range(1, len(drove)):
        restarted = new_episodes[step_idx - 1]
        unchanged = ~restarted
        assert np.array_equal(
            drove[step_idx][unchanged], drove[step_idx - 1][unchanged]
        )
        draws.append(drove[step_idx][restarted])
    all_draws = np.concatenate(draws)
    student_fraction = all_draws.mean()
    sigma = np.sqrt(0.25 / all_draws.size)
    assert abs(student_fraction - 0.5) < BINOMIAL_SIGMAS * sigma

    # reset_plans: once for all agents at the start, then per new episode
    assert bool(spy.reset_masks[0].all())
    expected_masks = [mask for mask in new_episodes if mask.any()]
    assert len(spy.reset_masks) - 1 == len(expected_masks)
    for recorded, expected in zip(spy.reset_masks[1:], expected_masks):
        assert np.array_equal(recorded.numpy(), expected)


@needs_teacher
def test_map_resample_counts_as_new_episode(
        teacher: Teacher,
        device: str,
    ) -> None:
    """At a map resample every agent is live, redrawn and reset."""
    spy = SpyStudent().to(device)
    config = small_config(
        device, beta_constant=0.5, iterations=1, env_steps_per_iteration=45
    )
    env, _ = run_recorded(
        teacher,
        spy,
        config,
        (f"env.resample_frequency={RESAMPLE_STEPS}",),
    )
    resample_steps = [
        step_idx for step_idx, tick in enumerate(env.ticks) if tick == 0
    ]
    assert resample_steps == [RESAMPLE_STEPS - 1, 2 * RESAMPLE_STEPS - 1]
    full_resets = [mask for mask in spy.reset_masks[1:] if bool(mask.all())]
    assert len(full_resets) >= len(resample_steps)


@needs_teacher
def test_beta_zero_student_drives_everyone(
        teacher: Teacher,
        device: str,
    ) -> None:
    """beta = 0: every action is the student's; labels are still added."""
    spy = SpyStudent().to(device)
    env, buffer = run_recorded(
        teacher, spy, small_config(device, beta_constant=0.0)
    )
    assert spy_drove(env).all()
    assert sum(len(added) for added in buffer.added_obs) > 0


@needs_teacher
def test_mlp_student_overfits_fixed_buffer(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Learning signal reaches the MLP through the normalized loss."""
    config = small_config(
        device,
        iterations=1,
        env_steps_per_iteration=10,
        gradient_steps_per_iteration=0,
    )
    torch.manual_seed(LOOP_SEED)
    student = MLPStudent(teacher).to(device)
    _, buffer = run_recorded(teacher, student, config)
    train_config = dataclasses.replace(
        config,
        gradient_steps_per_iteration=OVERFIT_STEPS,
        batch_size=OVERFIT_BATCH_SIZE,
    )
    all_obs = torch.cat(buffer.added_obs).to(device)
    all_labels = torch.cat(buffer.added_labels).to(device)

    def full_buffer_loss() -> float:
        """
        Returns:
            float: loss over every stored row (not a noisy minibatch).
        """
        with torch.no_grad():
            predicted = student.predict(all_obs)
            return float(distillation_loss(predicted, all_labels))

    initial_loss = full_buffer_loss()
    optimizer = make_optimizer(student, train_config)
    train_student(student, optimizer, buffer, train_config)
    final_loss = full_buffer_loss()
    print(f"\n  overfit loss {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < OVERFIT_LOSS_RATIO * initial_loss


def test_loss_normalization_weighs_controls_equally() -> None:
    """A full-scale error in accel and in steering rate cost the same."""
    label = torch.zeros(1, 2)
    accel_error = distillation_loss(torch.tensor([[8.0, 0.0]]), label)
    rate_error = distillation_loss(torch.tensor([[0.0, 1.2]]), label)
    assert torch.allclose(accel_error, rate_error)


def test_episode_stats_split_by_domain_and_driver() -> None:
    """Rows 0-2 nuPlan, 3 CARLA. Ends are counted per domain and driver
    with their mean length; a nuPlan end before NUPLAN_SCENARIO_STEPS is
    early; the goal-less share is over nuPlan labels; popping clears."""
    episodes = EpisodeState(
        teacher_drives=torch.tensor([True, False, False, False]),
        live=torch.ones(4, dtype=torch.bool),
        step_count=torch.tensor([NUPLAN_SCENARIO_STEPS, 50, 70, 900]),
        nuplan_rows=torch.tensor([True, True, True, False]),
    )
    episodes.sums["nuplan_labels"] = torch.tensor(8)
    episodes.sums["nuplan_goalless_labels"] = torch.tensor(2)
    add_episode_sums(episodes, torch.tensor([True, True, True, True]))
    stats = pop_episode_stats(episodes)
    assert stats["episodes/nuplan_teacher_ends"] == 1
    assert stats["episodes/nuplan_teacher_early_fraction"] == 0
    assert stats["episodes/nuplan_student_ends"] == 2
    assert stats["episodes/nuplan_student_mean_steps"] == 60
    assert stats["episodes/nuplan_student_early_fraction"] == 1
    assert stats["episodes/carla_student_mean_steps"] == 900
    assert stats["episodes/carla_teacher_ends"] == 0
    assert math.isnan(stats["episodes/carla_teacher_mean_steps"])
    assert stats["episodes/nuplan_goalless_label_fraction"] == 0.25
    assert episodes.sums == {}


def test_beta_schedules() -> None:
    """constant returns beta_constant; dagger steps down after warmup."""
    assert beta_at(DistillConfig(beta_constant=0.3), 10**9) == 0.3
    dagger = DistillConfig(
        beta_schedule=BETA_DAGGER,
        dagger_warmup_labels=100,
        dagger_stage_labels=10,
        dagger_stage_count=5,
    )
    labels = [0, 99, 100, 109, 110, 139, 140, 10**6]
    betas = [beta_at(dagger, count) for count in labels]
    assert betas == pytest.approx([1.0, 1.0, 0.8, 0.8, 0.6, 0.2, 0.0, 0.0])


def run_on_cpu(
        teacher: Teacher,
        config: DistillConfig,
    ) -> tuple[list, MLPStudent]:
    """
    One deterministic CPU run with a freshly seeded MLP student.

    Args:
        teacher (Teacher): labeler on the CPU.
        config (DistillConfig): run settings with device "cpu".

    Returns:
        tuple[list, MLPStudent]: labels added per step, and the student.
    """
    torch.manual_seed(config.seed)
    student = MLPStudent(teacher)
    _, buffer = run_recorded(teacher, student, config)
    return buffer.added_labels, student


@needs_teacher
def test_same_seed_same_run(
        distill_args: dict,
    ) -> None:
    """Two CPU runs with one seed: identical labels and parameters."""
    cpu_teacher = Teacher(TEACHER_CHECKPOINT, distill_args, "cpu")
    config = small_config("cpu", beta_constant=0.5, iterations=2)
    torch.use_deterministic_algorithms(True)
    try:
        labels_a, student_a = run_on_cpu(cpu_teacher, config)
        labels_b, student_b = run_on_cpu(cpu_teacher, config)
    finally:
        torch.use_deterministic_algorithms(False)
    assert all(torch.equal(a, b) for a, b in zip(labels_a, labels_b))
    for param_a, param_b in zip(
        student_a.parameters(), student_b.parameters()
    ):
        assert torch.equal(param_a, param_b)


@needs_teacher
def test_checkpoint_round_trip(
        tmp_path,
        teacher: Teacher,
        device: str,
    ) -> None:
    """Saved student, optimizer and labels_seen load back exactly."""
    config = small_config(device, output_dir=str(tmp_path))
    torch.manual_seed(LOOP_SEED)
    student = MLPStudent(teacher).to(device)
    env = make_small_env(config.seed)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    result = distill(env, teacher, student, buffer, config)
    env.close()

    torch.manual_seed(LOOP_SEED + 1)
    restored = MLPStudent(teacher).to(device)
    optimizer = make_optimizer(restored, config)
    labels_seen = load_checkpoint(
        str(tmp_path / "student.pt"), restored, optimizer
    )
    assert labels_seen == result.labels_seen
    obs, _ = buffer.sample(32)
    assert torch.equal(student.predict(obs), restored.predict(obs))
    saved_state = result.optimizer.state_dict()["state"]
    restored_state = optimizer.state_dict()["state"]
    for key, saved in saved_state.items():
        for name, value in saved.items():
            restored_value = restored_state[key][name]
            assert torch.equal(
                torch.as_tensor(value), torch.as_tensor(restored_value)
            )


@needs_teacher
def test_resume_continues_labels_and_evaluation_grid(
        tmp_path,
        teacher: Teacher,
        device: str,
    ) -> None:
    """resume_checkpoint restores labels_seen, skips the start-of-run
    evaluation and evaluates at the next multiples of the interval up to
    the absolute max_labels."""
    eval_interval = 1500
    first_dir = tmp_path / "first"
    first_config = small_config(
        device,
        iterations=1000,
        env_steps_per_iteration=10,
        eval_interval_labels=eval_interval,
        max_labels=2 * eval_interval,
        output_dir=str(first_dir),
    )
    env = make_small_env(first_config.seed)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(first_config.buffer_size, obs_dim, device)
    first = distill(env, teacher, SpyStudent().to(device), buffer,
                    first_config, RecordingEvaluator())
    env.close()

    resumed_dir = tmp_path / "resumed"
    resumed_config = dataclasses.replace(
        first_config,
        max_labels=4 * eval_interval,
        output_dir=str(resumed_dir),
        resume_checkpoint=str(first_dir / "student.pt"),
    )
    evaluator = RecordingEvaluator()
    env = make_small_env(resumed_config.seed)
    buffer = LabelBuffer(resumed_config.buffer_size, obs_dim, device)
    resumed = distill(env, teacher, SpyStudent().to(device), buffer,
                      resumed_config, evaluator)
    env.close()
    assert resumed.log_rows[0]["labels_seen"] > first.labels_seen
    assert resumed.labels_seen >= resumed_config.max_labels
    eval_labels = [
        json.loads(line)["labels_seen"]
        for line in (resumed_dir / "eval.jsonl").read_text().splitlines()
    ]
    assert evaluator.calls == len(eval_labels) == 2
    assert eval_labels[0] >= 3 * eval_interval > first.labels_seen


@needs_teacher
def test_stop_file_ends_the_run_after_the_iteration(
        tmp_path,
        teacher: Teacher,
        device: str,
    ) -> None:
    """A stop file in output_dir ends the run at the next iteration: no
    further labels or evaluations, and the final checkpoint is saved."""
    eval_interval = 1500
    config = small_config(
        device,
        iterations=1000,
        env_steps_per_iteration=10,
        eval_interval_labels=eval_interval,
        max_labels=6 * eval_interval,
        output_dir=str(tmp_path),
    )
    env = make_small_env(config.seed)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    evaluator = StoppingEvaluator(str(tmp_path / STOP_FILE_NAME))
    result = distill(env, teacher, SpyStudent().to(device), buffer,
                     config, evaluator)
    env.close()
    assert evaluator.calls == 2
    assert eval_interval <= result.labels_seen < 2 * eval_interval
    assert result.log_rows[-1]["labels_seen"] == result.labels_seen
    assert (tmp_path / "student.pt").is_file()


@pytest.mark.parametrize(
    "bad_settings",
    [
        dict(batch_size=200, buffer_size=100),
        dict(iterations=0),
        dict(env_steps_per_iteration=-1),
        dict(gradient_steps_per_iteration=-1),
        dict(beta_constant=1.5),
        dict(beta_constant=-0.1),
        dict(beta_schedule="linear"),
        dict(learning_rate=0.0),
        dict(learning_rate=float("nan")),
        dict(dagger_stage_count=0),
        dict(fresh_fraction=1.5),
        dict(fresh_fraction=-0.1),
        dict(chunks_per_update=0),
        dict(probe_set_size=-1),
    ],
)
def test_invalid_config_is_rejected(
        bad_settings: dict,
    ) -> None:
    """Config comes from the command line; reject what breaks the loop."""
    with pytest.raises(ValueError):
        DistillConfig(**bad_settings)


@needs_teacher
def test_report_iteration_timing(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Reported, not asserted: seconds per iteration at 1024 agents."""
    config = small_config(
        device,
        iterations=4,
        env_steps_per_iteration=32,
        gradient_steps_per_iteration=32,
        batch_size=4096,
        buffer_size=500_000,
    )
    torch.manual_seed(LOOP_SEED)
    student = MLPStudent(teacher).to(device)
    env = make_small_env(config.seed, agent_count=TIMING_AGENT_COUNT)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    result = distill(env, teacher, student, buffer, config)
    env.close()
    # The first iteration includes warmup
    rows = result.log_rows[1:]
    collect = np.mean([row["collect_seconds"] for row in rows])
    train = np.mean([row["train_seconds"] for row in rows])
    print(
        f"\n  {TIMING_AGENT_COUNT} agents, BC, MLP: collect {collect:.3f} s "
        f"({config.env_steps_per_iteration} steps), train {train:.3f} s "
        f"({config.gradient_steps_per_iteration} x {config.batch_size})"
    )


class RecordingEvaluator(DistillEvaluator):
    """Evaluator that records when it was called instead of evaluating."""

    def __init__(self) -> None:
        """
        No config needed; nothing is run.
        """
        self.calls = 0

    def evaluate(
            self,
            controller: torch.nn.Module,
            action_type: str = "continuous",
        ) -> dict:
        """
        Args:
            controller (torch.nn.Module): the student.
            action_type (str): unused.

        Returns:
            dict: a fixed metric so the loop can log it.
        """
        self.calls += 1
        return {"num_goals_reached": float(self.calls)}


class StoppingEvaluator(RecordingEvaluator):
    """RecordingEvaluator that writes the run's stop file on its second
    call (the first evaluation after training started)."""

    def __init__(
            self,
            stop_path: str,
        ) -> None:
        """
        Args:
            stop_path (str): the stop file to create.
        """
        super().__init__()
        self.stop_path = stop_path

    def evaluate(
            self,
            controller: torch.nn.Module,
            action_type: str = "continuous",
        ) -> dict:
        """
        Args:
            controller (torch.nn.Module): the student.
            action_type (str): unused.

        Returns:
            dict: RecordingEvaluator's metric.
        """
        metrics = super().evaluate(controller, action_type)
        if self.calls == 2:
            with open(self.stop_path, "w") as stop_file:
                stop_file.write("plateau\n")
        return metrics


@needs_teacher
def test_evaluation_schedule_and_label_budget(
        tmp_path,
        teacher: Teacher,
        device: str,
    ) -> None:
    """Evaluated at 0 labels and at each interval crossing; stops at the
    label budget; one checkpoint per evaluation."""
    eval_interval = 1500
    config = small_config(
        device,
        iterations=1000,
        env_steps_per_iteration=10,
        eval_interval_labels=eval_interval,
        max_labels=4 * eval_interval,
        output_dir=str(tmp_path),
    )
    evaluator = RecordingEvaluator()
    env = make_small_env(config.seed)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    result = distill(env, teacher, SpyStudent().to(device), buffer, config,
                     evaluator)
    env.close()
    labels_per_iteration = 10 * env.num_agents
    assert config.max_labels <= result.labels_seen
    assert result.labels_seen < config.max_labels + labels_per_iteration
    eval_rows = [
        json.loads(line)
        for line in (tmp_path / "eval.jsonl").read_text().splitlines()
    ]
    eval_labels = [row["labels_seen"] for row in eval_rows]
    assert eval_labels[0] == 0
    assert evaluator.calls == len(eval_labels) == 5
    for crossing, labels in enumerate(eval_labels[1:], start=1):
        assert crossing * eval_interval <= labels
        assert labels < crossing * eval_interval + labels_per_iteration
    for labels in eval_labels:
        assert (tmp_path / f"student_{labels}.pt").is_file()


def test_wrong_side_saturation_metric() -> None:
    """On the edge with the label: saturated only. On the edge opposite
    the label: also wrong-side. Zero labels never count as wrong-side."""
    bound = torch.tensor([4.0, 0.75])
    predicted = torch.tensor(
        [
            [4.0, 0.75],  # both on the edge, labels same side
            [-4.0, 0.75],  # both on the edge, labels opposite
            [0.0, 0.0],  # interior
            [4.0, -0.75],  # on the edge, labels zero
        ]
    )
    labels = torch.tensor(
        [[4.0, 0.6], [4.0, -0.6], [1.0, 0.1], [0.0, 0.0]]
    )
    on_bound, wrong_side = saturation_stats(predicted, labels, bound)
    assert float(on_bound) == pytest.approx(6 / 8)
    assert float(wrong_side) == pytest.approx(2 / 8)


def run_fresh(
        teacher: Teacher,
        config: DistillConfig,
    ) -> tuple[list, MPCStudent, RecordingBuffer]:
    """
    One fresh-gradient run with a freshly seeded MPC student.

    Args:
        teacher (Teacher): labeler on config.device.
        config (DistillConfig): run settings with fresh_gradient set.

    Returns:
        tuple: log rows, the student and the recording buffer.
    """
    torch.manual_seed(config.seed)
    student = MPCStudent(teacher).to(config.device)
    env = RecordingEnv(make_small_env(config.seed))
    obs_dim = env.env.single_observation_space.shape[0]
    buffer = RecordingBuffer(config.buffer_size, obs_dim, config.device)
    result = distill(env, teacher, student, buffer, config)
    env.env.close()
    return result.log_rows, student, buffer


@needs_teacher
def test_fresh_gradient_needs_mpc_student(
        teacher: Teacher,
        device: str,
    ) -> None:
    """The MLP student has no act_and_grad."""
    config = small_config(device, fresh_gradient=True)
    with pytest.raises(ValueError, match="MPCStudent"):
        run_recorded(teacher, MLPStudent(teacher).to(device), config)


@needs_teacher
@pytest.mark.parametrize(
    "fresh_fraction, fresh_per_iteration, replay_per_iteration",
    [(1.0, 10, 0), (0.0, 0, 10), (0.5, 5, 5)],
)
def test_fresh_slots_follow_fresh_fraction(
        teacher: Teacher,
        device: str,
        fresh_fraction: float,
        fresh_per_iteration: int,
        replay_per_iteration: int,
    ) -> None:
    """50 steps in groups of 5 give 10 update slots per iteration, split
    by fresh_fraction; every slot raises the weight version, and the
    shipped gradients are at most a few updates old. BC labels are still
    added for every live agent."""
    config = small_config(
        device,
        fresh_gradient=True,
        fresh_fraction=fresh_fraction,
        chunks_per_update=5,
    )
    rows, student, buffer = run_fresh(teacher, config)
    for row in rows:
        assert row["fresh_updates"] == fresh_per_iteration
        assert row["replay_updates"] == replay_per_iteration
    assert student.weight_version == 20
    assert len(buffer.added_labels) == 2 * config.env_steps_per_iteration
    if fresh_per_iteration:
        ages = [row["grad_age_updates"] for row in rows]
        assert all(0 <= age < 10 for age in ages)
        assert all(math.isfinite(row["loss"]) for row in rows)


@needs_teacher
@pytest.mark.parametrize("chunks_per_update", [1, 3])
def test_fresh_gradients_are_applied_before_the_next_step(
        teacher: Teacher,
        device: str,
        chunks_per_update: int,
    ) -> None:
    """Every update runs between two env steps, so each chunk was computed
    with the weights it updates: age 0 for any group size, and the weights
    change during collection."""
    config = small_config(
        device,
        fresh_gradient=True,
        chunks_per_update=chunks_per_update,
        iterations=1,
        env_steps_per_iteration=6,
    )
    rows, student, _ = run_fresh(teacher, config)
    assert rows[0]["grad_age_updates"] == 0
    assert rows[0]["fresh_updates"] == 6 // chunks_per_update
    assert student.weight_version == 6 // chunks_per_update


@needs_teacher
def test_fresh_gradient_runs_are_reproducible(
        distill_args: dict,
    ) -> None:
    """Two CPU runs with one seed give identical parameters."""
    cpu_teacher = Teacher(TEACHER_CHECKPOINT, distill_args, "cpu")
    config = small_config(
        "cpu",
        fresh_gradient=True,
        fresh_fraction=0.5,
        chunks_per_update=5,
        env_steps_per_iteration=10,
    )
    torch.use_deterministic_algorithms(True)
    try:
        _, student_a, _ = run_fresh(cpu_teacher, config)
        _, student_b, _ = run_fresh(cpu_teacher, config)
    finally:
        torch.use_deterministic_algorithms(False)
    for param_a, param_b in zip(
        student_a.parameters(), student_b.parameters()
    ):
        assert torch.equal(param_a, param_b)


@needs_teacher
@pytest.mark.parametrize("fresh_gradient", [False, True])
def test_report_mpc_iteration_timing(
        teacher: Teacher,
        device: str,
        fresh_gradient: bool,
    ) -> None:
    """Reported, not asserted: BC with the MPC student at 1024 agents,
    replay (solve in training) against fresh gradients (solve in
    collection). Both do 8 updates per 32 env steps."""
    config = small_config(
        device,
        iterations=4,
        env_steps_per_iteration=32,
        gradient_steps_per_iteration=8,
        batch_size=1024,
        buffer_size=500_000,
        fresh_gradient=fresh_gradient,
        chunks_per_update=4,
    )
    torch.manual_seed(LOOP_SEED)
    student = MPCStudent(teacher).to(device)
    env = make_small_env(config.seed, agent_count=TIMING_AGENT_COUNT)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    result = distill(env, teacher, student, buffer, config)
    env.close()
    rows = result.log_rows[1:]
    collect = np.mean([row["collect_seconds"] for row in rows])
    train = np.mean([row["train_seconds"] for row in rows])
    print(
        f"\n  {TIMING_AGENT_COUNT} agents, BC, MPC, "
        f"fresh_gradient={fresh_gradient}: collect {collect:.3f} s, "
        f"train {train:.3f} s ({collect + train:.3f} s per iteration)"
    )


def test_stage_at_matches_beta_schedule() -> None:
    """Stage 0 through the warmup, then one stage per dagger_stage_labels;
    the constant schedule never leaves stage 0."""
    dagger = DistillConfig(
        beta_schedule=BETA_DAGGER,
        dagger_warmup_labels=100,
        dagger_stage_labels=10,
    )
    labels = [0, 99, 100, 109, 110, 139, 140]
    assert [stage_at(dagger, count) for count in labels] == [
        0, 0, 1, 1, 2, 4, 5
    ]
    assert stage_at(DistillConfig(), 10**9) == 0


def test_probe_freezes_start_then_one_set_per_finished_stage() -> None:
    """The start set waits for enough rows; a skipped stage boundary still
    gets its set; sets never change afterwards."""
    config = DistillConfig(
        device="cpu",
        beta_schedule=BETA_DAGGER,
        dagger_warmup_labels=100,
        dagger_stage_labels=10,
        probe_set_size=20,
        probe_window_labels=40,
        buffer_size=1000,
        batch_size=10,
    )
    buffer = LabelBuffer(config.buffer_size, 4, "cpu")
    probe = ForgettingProbe(config, 0)

    def add_ids(first_id: int, row_count: int) -> None:
        ids = torch.arange(first_id, first_id + row_count).float()
        buffer.add(
            ids[:, None].expand(row_count, 4), torch.stack([ids, ids], -1)
        )

    add_ids(0, 15)
    probe.freeze_due_sets(buffer, 15)
    assert list(probe.sets) == []
    add_ids(15, 15)
    probe.freeze_due_sets(buffer, 30)
    assert list(probe.sets) == ["start"]
    start_labels = probe.sets["start"][1].clone()

    add_ids(30, 30)
    probe.freeze_due_sets(buffer, 105)  # stage 1: stage 0 finished
    assert list(probe.sets) == ["start", "stage_0"]
    # 20 rows spread over the newest 40 of 60 (ids 20..59)
    assert float(probe.sets["stage_0"][1][0, 0]) == 20.0
    probe.freeze_due_sets(buffer, 135)  # stage 4: stages 1 to 3 finished
    assert list(probe.sets) == [
        "start", "stage_0", "stage_1", "stage_2", "stage_3"
    ]
    assert torch.equal(probe.sets["start"][1], start_labels)


class ConstantStudent(torch.nn.Module):
    """Student that predicts a fixed control."""

    def predict(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            torch.Tensor: (B,2) zero controls.
        """
        return torch.zeros(obs.shape[0], 2)


def test_probe_loss_is_the_replay_loss_on_each_set() -> None:
    """Batched over PROBE_BATCH_SIZE, the logged value equals the loss on
    the whole set."""
    config = DistillConfig(device="cpu", probe_set_size=10, batch_size=10)
    probe = ForgettingProbe(config, 0)
    labels = torch.randn(10000, 2)
    probe.sets["start"] = (torch.zeros(10000, 3), labels)
    losses = probe.evaluate(ConstantStudent())
    expected = distillation_loss(torch.zeros(10000, 2), labels)
    assert losses["probe/start"] == pytest.approx(float(expected), rel=1e-5)


@needs_teacher
def test_probe_is_logged_every_iteration_and_leaves_training_alone(
        teacher: Teacher,
        device: str,
    ) -> None:
    """A short DAgger run logs probe/start from the first iteration and a
    stage set once the stage ends; the trained parameters are those of a
    run without the probe."""
    settings = dict(
        beta_schedule=BETA_DAGGER,
        dagger_warmup_labels=1,
        dagger_stage_labels=100,
        dagger_stage_count=3,
        iterations=3,
        env_steps_per_iteration=10,
    )
    students, logs = [], []
    for probe_set_size in (50, 0):
        config = small_config(
            device, probe_set_size=probe_set_size, **settings
        )
        torch.manual_seed(LOOP_SEED)
        student = MLPStudent(teacher).to(device)
        env = make_small_env(config.seed)
        obs_dim = env.single_observation_space.shape[0]
        buffer = LabelBuffer(config.buffer_size, obs_dim, device)
        logs.append(distill(env, teacher, student, buffer, config).log_rows)
        env.close()
        students.append(student)
    probed_rows, plain_rows = logs
    assert all("probe/start" in row for row in probed_rows)
    assert any("probe/stage_0" in row for row in probed_rows)
    assert all(math.isfinite(row["probe/start"]) for row in probed_rows)
    assert not any(key.startswith("probe/") for key in plain_rows[0])
    for param_a, param_b in zip(
        students[0].parameters(), students[1].parameters()
    ):
        assert torch.equal(param_a, param_b)
