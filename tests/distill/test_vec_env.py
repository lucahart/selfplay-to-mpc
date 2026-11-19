"""VecDrive: workers behave like the Drives they hold, ticks are per
agent, failures surface with their traceback, and the loop runs on it."""

import multiprocessing
import os

import numpy as np
import pytest
import torch

from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.env import NUPLAN_ENV_OVERRIDES, REPO_ROOT, make_env
from pufferlib.distill.loop import DistillConfig, distill
from pufferlib.distill.students import MPCStudent
from pufferlib.distill.teacher import Teacher
from pufferlib.distill.vec_env import VecDrive

from conftest import SMALL_AGENT_COUNT, needs_teacher

VEC_SEED = 91
STEP_COUNT = 12
RESAMPLE_STEPS = 5
ACTION_SEED = 92
CHECKED_IN_NUPLAN_DIR = os.path.join(
    REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "nuplan"
)
# The checked-in nuPlan bin has 16 policy vehicles
NUPLAN_BIN_VEHICLE_COUNT = 16
# Short nuPlan episodes, so the test sees a truncation
NUPLAN_TEST_SCENARIO_STEPS = 8
MIXED_STEP_COUNT = 12


def random_actions(
        generator: np.random.Generator,
        agent_count: int,
    ) -> np.ndarray:
    """
    Args:
        generator (np.random.Generator): seeded source.
        agent_count (int): rows.

    Returns:
        np.ndarray: (agent_count,2) float32 actions in the env's range.
    """
    return generator.uniform(-1, 1, (agent_count, 2)).astype(np.float32)


@pytest.mark.parametrize("worker_count", [1, 2])
def test_workers_reproduce_the_drives_they_hold(
        worker_count: int,
    ) -> None:
    """Row blocks of VecDrive equal separately run Drives seeded
    seed + worker index, step by step (observations, flags, ticks, world
    poses), with map resamples in the middle of the run."""
    overrides = (f"env.resample_frequency={RESAMPLE_STEPS}",)
    rows = SMALL_AGENT_COUNT
    vec = VecDrive(rows * worker_count, worker_count, VEC_SEED, overrides)
    plain = [
        make_env("continuous", VEC_SEED + index, rows, overrides)[0]
        for index in range(worker_count)
    ]
    try:
        vec_obs, _ = vec.reset(seed=VEC_SEED)
        plain_obs = [
            env.reset(seed=VEC_SEED + index)[0].copy()
            for index, env in enumerate(plain)
        ]
        assert np.array_equal(vec_obs, np.concatenate(plain_obs))
        generator = np.random.default_rng(ACTION_SEED)
        for _ in range(STEP_COUNT):
            actions = random_actions(generator, rows * worker_count)
            vec_obs, _, terminals, truncations, _ = vec.step(actions)
            for index, env in enumerate(plain):
                block = slice(index * rows, (index + 1) * rows)
                obs, _, terminal, truncation, _ = env.step(actions[block])
                assert np.array_equal(vec_obs[block], obs)
                assert np.array_equal(terminals[block], terminal.astype(bool))
                assert np.array_equal(
                    truncations[block], truncation.astype(bool)
                )
                assert (vec.tick[block] == env.tick).all()
                vec_state = vec.get_global_agent_state()
                plain_state = env.get_global_agent_state()
                for key in ("x", "y", "heading"):
                    assert np.array_equal(
                        vec_state[key][block], plain_state[key]
                    )
    finally:
        vec.close()
        for env in plain:
            env.close()
    assert multiprocessing.active_children() == []


def test_mixed_workers_run_nuplan_and_carla() -> None:
    """Worker 0 with NUPLAN_ENV_OVERRIDES (on the checked-in bin) equals
    a nuPlan replay Drive step by step and truncates at its short
    scenario length, with the same failed-replan flags; worker 1 (no
    overrides) stays on CARLA gigaflow, does not truncate and has no
    failed replans. Only one plain Drive runs here: Drives in one process share
    the C rand() stream, which replanning draws from."""
    nuplan_overrides = NUPLAN_ENV_OVERRIDES + (
        f"env.map_dir={CHECKED_IN_NUPLAN_DIR}",
        "env.num_maps=1",
        f"env.scenario_length={NUPLAN_TEST_SCENARIO_STEPS}",
    )
    rows = NUPLAN_BIN_VEHICLE_COUNT
    vec = VecDrive(
        rows * 2, 2, VEC_SEED, worker_env_overrides=(nuplan_overrides, ())
    )
    nuplan_env, _ = make_env("continuous", VEC_SEED, rows, nuplan_overrides)
    nuplan_rows = slice(0, rows)
    carla_rows = slice(rows, 2 * rows)
    try:
        vec_obs, _ = vec.reset(seed=VEC_SEED)
        obs, _ = nuplan_env.reset(seed=VEC_SEED)
        assert np.array_equal(vec_obs[nuplan_rows], obs)
        generator = np.random.default_rng(ACTION_SEED)
        truncated = np.zeros(rows * 2, dtype=bool)
        for _ in range(MIXED_STEP_COUNT):
            actions = random_actions(generator, rows * 2)
            vec_obs, _, _, truncations, _ = vec.step(actions)
            truncated |= truncations
            obs, _, _, _, _ = nuplan_env.step(actions[nuplan_rows])
            assert np.array_equal(vec_obs[nuplan_rows], obs)
            assert np.isfinite(vec_obs[carla_rows]).all()
            replan_failed = vec.get_replay_replan_failed()
            assert np.array_equal(
                replan_failed[nuplan_rows],
                nuplan_env.get_replay_replan_failed(),
            )
            assert not replan_failed[carla_rows].any()
        assert truncated[nuplan_rows].all()
        assert not truncated[carla_rows].any()
    finally:
        vec.close()
        nuplan_env.close()
    assert multiprocessing.active_children() == []


def test_worker_overrides_need_one_tuple_per_worker() -> None:
    """Two workers with three override tuples fail before any worker
    starts."""
    with pytest.raises(ValueError, match="worker override tuples"):
        VecDrive(
            SMALL_AGENT_COUNT * 2, 2, VEC_SEED,
            worker_env_overrides=((), (), ()),
        )
    assert multiprocessing.active_children() == []


def test_tick_is_per_agent_and_zero_right_after_a_resample() -> None:
    """Every agent's tick counts up and returns to 0 at the resample."""
    overrides = (f"env.resample_frequency={RESAMPLE_STEPS}",)
    vec = VecDrive(SMALL_AGENT_COUNT * 2, 2, VEC_SEED, overrides)
    try:
        vec.reset(seed=VEC_SEED)
        generator = np.random.default_rng(ACTION_SEED)
        ticks = []
        for _ in range(RESAMPLE_STEPS + 1):
            vec.step(random_actions(generator, vec.num_agents))
            ticks.append(vec.tick.copy())
        assert all(tick.shape == (vec.num_agents,) for tick in ticks)
        assert [int(tick[0]) for tick in ticks] == [1, 2, 3, 4, 0, 1]
        assert all((tick == tick[0]).all() for tick in ticks)
    finally:
        vec.close()


def test_invalid_use_is_rejected_before_any_worker_starts() -> None:
    """Agents must split evenly, and a bad override fails in the parent
    (it builds one env to learn the observation width)."""
    with pytest.raises(ValueError, match="multiple"):
        VecDrive(10, 3, VEC_SEED)
    with pytest.raises(Exception, match="no_such_key"):
        VecDrive(
            SMALL_AGENT_COUNT * 2, 2, VEC_SEED, ("env.no_such_key=1",)
        )
    assert multiprocessing.active_children() == []


def test_a_dead_worker_is_reported_and_everything_is_cleaned_up() -> None:
    """Killing a worker makes the next step raise instead of hanging, and
    close() leaves no process behind."""
    vec = VecDrive(SMALL_AGENT_COUNT * 2, 2, VEC_SEED)
    vec.reset(seed=VEC_SEED)
    vec.workers[1].kill()
    vec.workers[1].join()
    with pytest.raises(RuntimeError, match="env worker 1 died"):
        vec.step(np.zeros((vec.num_agents, 2), dtype=np.float32))
    assert multiprocessing.active_children() == []


@needs_teacher
def test_one_worker_loop_equals_the_plain_env_loop(
        distill_args: dict,
    ) -> None:
    """distill() on a one-worker VecDrive trains the same parameters as on
    the Drive it wraps (CPU, deterministic), fresh gradients included."""
    from pufferlib.distill.env import TEACHER_CHECKPOINT

    teacher = Teacher(TEACHER_CHECKPOINT, distill_args, "cpu")
    config = DistillConfig(
        seed=VEC_SEED,
        device="cpu",
        iterations=2,
        env_steps_per_iteration=6,
        batch_size=64,
        buffer_size=50_000,
        eval_interval_labels=0,
        fresh_gradient=True,
        chunks_per_update=2,
    )
    students = []
    torch.use_deterministic_algorithms(True)
    try:
        for use_vec in (False, True):
            torch.manual_seed(config.seed)
            student = MPCStudent(teacher)
            if use_vec:
                env = VecDrive(SMALL_AGENT_COUNT, 1, config.seed)
            else:
                env = make_env("continuous", config.seed, SMALL_AGENT_COUNT)[0]
            obs_dim = env.single_observation_space.shape[0]
            buffer = LabelBuffer(config.buffer_size, obs_dim, "cpu")
            distill(env, teacher, student, buffer, config)
            env.close()
            students.append(student)
    finally:
        torch.use_deterministic_algorithms(False)
    for param_a, param_b in zip(
        students[0].parameters(), students[1].parameters()
    ):
        assert torch.equal(param_a, param_b)
