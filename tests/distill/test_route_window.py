"""Route windows from the sim (drive.h write_route_window): every agent
gets its route ahead, 1 m apart in arc length, starting behind it, in
gigaflow and in nuPlan replay from the first step, and VecDrive hands over
the same windows as the Drives it holds."""

import numpy as np

from pufferlib.distill.env import load_distill_args, make_env
from pufferlib.distill.tube import ROUTE_WINDOW_SPACING_M
from pufferlib.distill.vec_env import VecDrive
from pufferlib.ocean.benchmark.manager import EvalManager
from pufferlib.ocean.drive.drive import ROUTE_WINDOW_POINT_COUNT, Drive

from conftest import SMALL_AGENT_COUNT
from test_nuplan_eval import (
    CHECKED_IN_NUPLAN_DIR,
    CHECKED_IN_POLICY_VEHICLE_COUNT,
    ENV_SEED,
    REPLAN_EVALUATOR,
    section_env_kwargs,
)

ROUTE_SEED = 17
STEP_COUNT = 30
# drive.h ROUTE_WINDOW_BACK_M: the agent's own arc length is this point
AGENT_POINT_IDX = 6
# Agents spawn on a lane; their route starts on it
SPAWN_LANE_DISTANCE_M = 0.1
# A chord is never longer than its arc; corners shorten it a little
MIN_CHORD_FRACTION = 0.5


def straight_ahead_actions(
        agent_count: int,
    ) -> np.ndarray:
    """
    Args:
        agent_count (int): rows.

    Returns:
        np.ndarray: (agent_count,2) gentle throttle, no steering.
    """
    actions = np.zeros((agent_count, 2), dtype=np.float32)
    actions[:, 0] = 0.3
    return actions


def test_every_spawned_agent_gets_its_route_window():
    env, _ = make_env("continuous", ROUTE_SEED, SMALL_AGENT_COUNT)
    try:
        env.reset(seed=ROUTE_SEED)
        window, valid = env.get_route_window()
        assert window.shape == (
            SMALL_AGENT_COUNT, ROUTE_WINDOW_POINT_COUNT, 2
        )
        assert bool(valid[:, AGENT_POINT_IDX:].any(axis=1).all())
        distance_m = np.linalg.norm(window[:, AGENT_POINT_IDX], axis=-1)
        assert float(distance_m.max()) < SPAWN_LANE_DISTANCE_M
    finally:
        env.close()


def test_window_points_are_one_meter_apart_along_the_route():
    env, _ = make_env("continuous", ROUTE_SEED, SMALL_AGENT_COUNT)
    try:
        env.reset(seed=ROUTE_SEED)
        for _ in range(STEP_COUNT):
            env.step(straight_ahead_actions(SMALL_AGENT_COUNT))
        window, valid = env.get_route_window()
        both_valid = valid[:, 1:] & valid[:, :-1]
        chords_m = np.linalg.norm(np.diff(window, axis=1), axis=-1)
        chords_m = chords_m[both_valid]
        assert float(chords_m.max()) <= ROUTE_WINDOW_SPACING_M + 1e-4
        assert float(chords_m.min()) >= (
            MIN_CHORD_FRACTION * ROUTE_WINDOW_SPACING_M
        )
        # Valid points form one run per agent
        starts = np.diff(valid.astype(np.int8), axis=1) == 1
        assert int(starts.sum(axis=1).max()) <= 1
    finally:
        env.close()


def test_vec_drive_hands_over_the_drives_windows():
    vec = VecDrive(SMALL_AGENT_COUNT, 1, ROUTE_SEED)
    plain, _ = make_env("continuous", ROUTE_SEED, SMALL_AGENT_COUNT)
    try:
        vec.reset(seed=ROUTE_SEED)
        plain.reset(seed=ROUTE_SEED)
        actions = straight_ahead_actions(SMALL_AGENT_COUNT)
        for _ in range(3):
            vec.step(actions)
            plain.step(actions)
        vec_window, vec_valid = vec.get_route_window()
        plain_window, plain_valid = plain.get_route_window()
        assert np.array_equal(vec_valid, plain_valid)
        assert np.array_equal(vec_window, plain_window)
    finally:
        vec.close()
        plain.close()


def test_replay_agents_get_their_logged_route_from_the_first_step():
    # Replay builds no path; the route line comes from the logged route,
    # at init (the first reset) as at every later one
    manager = EvalManager.from_config(load_distill_args())
    env_kwargs = section_env_kwargs(manager, REPLAN_EVALUATOR)
    env_kwargs.update(
        map_dir=CHECKED_IN_NUPLAN_DIR,
        num_maps=1,
        num_eval_scenarios=1,
        num_agents=CHECKED_IN_POLICY_VEHICLE_COUNT,
        action_type="continuous",
    )
    env = Drive(**env_kwargs)
    try:
        env.reset(seed=ENV_SEED)
        _, valid = env.get_route_window()
        assert bool(valid[:, AGENT_POINT_IDX].all())
    finally:
        env.close()
