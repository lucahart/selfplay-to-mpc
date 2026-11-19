"""The nuplan_multi eval section: replay self-play on nuPlan bins, with
the same eval settings as validation_gigaflow. Runs on the checked-in bin,
so it needs no download."""

import os

import numpy as np
import pytest

from pufferlib.distill.bicycle import ACCELERATION_VALUES, STEERING_VALUES
from pufferlib.distill.env import REPO_ROOT, load_distill_args
from pufferlib.ocean.benchmark.evaluators.base import Evaluator
from pufferlib.ocean.benchmark.manager import EvalManager
from pufferlib.ocean.drive.drive import Drive

NUPLAN_EVALUATOR = "nuplan_multi"
REPLAN_EVALUATOR = "nuplan_multi_replan"
CARLA_EVALUATOR = "validation_gigaflow"
CHECKED_IN_NUPLAN_DIR = os.path.join(
    REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "nuplan"
)
# The checked-in bin has 16 policy vehicles; one agent slot each
CHECKED_IN_POLICY_VEHICLE_COUNT = 16
# Mirrors drive.h (datatypes.h and the controller constants)
VEHICLE_TYPE = 1
CONTROLLER_STATIC = 0
CONTROLLER_POLICY = 1
CONTROLLER_REPLAY = 2
EXPERT_FLAG = 1
PARKED_FLAG = 2
ENV_SEED = 0


def find_evaluator(
        manager: EvalManager,
        name: str,
    ) -> Evaluator:
    """
    Args:
        manager (EvalManager): manager built from the composed config.
        name (str): [eval.<name>] section.

    Returns:
        Evaluator: the section's evaluator.
    """
    return next(
        evaluator for evaluator in manager.evaluators
        if evaluator.name == name
    )


def section_env_kwargs(
        manager: EvalManager,
        name: str,
    ) -> dict:
    """
    Drive kwargs as the evaluator builds them for its rollout.

    Args:
        manager (EvalManager): manager built from the composed config.
        name (str): [eval.<name>] section.

    Returns:
        dict: env kwargs with the section's overrides applied.
    """
    eval_args = manager._build_eval_args(
        find_evaluator(manager, name), env_name="puffer_drive",
        global_step=None,
    )
    return dict(eval_args["env"])


def test_nuplan_multi_section_composes() -> None:
    """The section is off during training, runs replay self-play on the
    fetched val bins, sweeps num_scenarios distinct scenarios and shares
    every reward weight with validation_gigaflow."""
    manager = EvalManager.from_config(load_distill_args())
    evaluator = find_evaluator(manager, NUPLAN_EVALUATOR)
    nuplan_env = section_env_kwargs(manager, NUPLAN_EVALUATOR)
    carla_env = section_env_kwargs(manager, CARLA_EVALUATOR)

    assert not evaluator.enabled
    assert nuplan_env["simulation_mode"] == "replay"
    assert nuplan_env["control_mode"] == "control_vehicles"
    assert nuplan_env["map_dir"] == "data/nuplan_mini_val"
    assert nuplan_env["eval_mode"] == 1
    num_scenarios = evaluator.config["eval"]["num_scenarios"]
    assert nuplan_env["num_eval_scenarios"] == num_scenarios
    reward_keys = [key for key in carla_env if key.startswith("reward_")]
    assert reward_keys
    for key in reward_keys:
        assert nuplan_env[key] == carla_env[key], key


def test_nuplan_multi_roles_on_checked_in_bin() -> None:
    """On the checked-in bin, the policy drives only unflagged vehicles,
    flagged experts replay their logs, parked agents stay static, and the
    episode truncates at scenario_length."""
    manager = EvalManager.from_config(load_distill_args())
    env_kwargs = section_env_kwargs(manager, NUPLAN_EVALUATOR)
    env_kwargs.update(
        map_dir=CHECKED_IN_NUPLAN_DIR,
        num_maps=1,
        num_eval_scenarios=1,
        num_agents=CHECKED_IN_POLICY_VEHICLE_COUNT,
    )
    env = Drive(**env_kwargs)
    try:
        env.reset(seed=ENV_SEED)
        agents = env.get_state()[0]["agents"]
        created = [agent for agent in agents if agent["log_valid"][0] == 1]
        for agent in created:
            if agent["controller"] == CONTROLLER_POLICY:
                assert agent["type"] == VEHICLE_TYPE
                assert agent["mark_as_expert"] == 0
            if agent["mark_as_expert"] == EXPERT_FLAG:
                assert agent["controller"] == CONTROLLER_REPLAY
            if agent["mark_as_expert"] == PARKED_FLAG:
                assert agent["controller"] == CONTROLLER_STATIC
        assert any(a["mark_as_expert"] == EXPERT_FLAG for a in created)
        assert any(a["mark_as_expert"] == PARKED_FLAG for a in created)

        steer_count = len(STEERING_VALUES)
        neutral_action = (
            (len(ACCELERATION_VALUES) // 2) * steer_count + steer_count // 2
        )
        actions = np.full(
            env.action_space.shape, neutral_action,
            dtype=env.action_space.dtype,
        )
        scenario_length = env_kwargs["scenario_length"]
        for step in range(1, scenario_length + 1):
            _, _, _, truncations, _ = env.step(actions)
            assert truncations.all() == (step == scenario_length), step
    finally:
        env.close()


def test_replay_episode_must_fit_the_log() -> None:
    """The checked-in bin's logs have 201 frames: init_step 0 with 200
    steps fits; one more step, or a later start, is rejected at env
    creation."""
    manager = EvalManager.from_config(load_distill_args())
    env_kwargs = section_env_kwargs(manager, NUPLAN_EVALUATOR)
    env_kwargs.update(
        map_dir=CHECKED_IN_NUPLAN_DIR,
        num_maps=1,
        num_eval_scenarios=1,
        num_agents=CHECKED_IN_POLICY_VEHICLE_COUNT,
    )
    assert env_kwargs["require_episode_within_log"]
    Drive(**dict(env_kwargs, init_step=0, scenario_length=200)).close()
    for init_step, scenario_length in ((0, 201), (1, 200)):
        with pytest.raises(ValueError, match="runs past its log of 201"):
            Drive(**dict(
                env_kwargs, init_step=init_step,
                scenario_length=scenario_length,
            ))


def test_nuplan_multi_replan_only_flips_replanning() -> None:
    """nuplan_multi_replan is nuplan_multi with replay_replan_goals on."""
    manager = EvalManager.from_config(load_distill_args())
    evaluator = find_evaluator(manager, REPLAN_EVALUATOR)
    replan_env = section_env_kwargs(manager, REPLAN_EVALUATOR)
    log_goal_env = section_env_kwargs(manager, NUPLAN_EVALUATOR)

    assert not evaluator.enabled
    assert evaluator.type_name == "multi_scenario"
    assert replan_env["replay_replan_goals"] == 1
    assert log_goal_env["replay_replan_goals"] == 0
    differing = {
        key for key in replan_env
        if replan_env[key] != log_goal_env.get(key)
    }
    assert differing == {"replay_replan_goals"}
