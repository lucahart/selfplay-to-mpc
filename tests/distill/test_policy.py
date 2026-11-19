"""ContinuousPolicy runs a teacher or student in evaluators exactly as the
distillation loop drives the env."""

import numpy as np
import pytest
import torch

import pufferlib.pytorch
from pufferlib.distill.bicycle import (
    STEERING,
    ego_state_from_obs,
    env_action_from_control,
)
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.loop import DistillConfig, DistillEvaluator, distill
from pufferlib.distill.policy import ACTION_SCALE, ContinuousPolicy
from pufferlib.distill.teacher import Teacher

from conftest import RecordingEnv, SpyStudent, make_small_env, needs_teacher
from test_teacher import rollout_obs

COMPARED_STEPS = 100
POLICY_SEED = 41
EVAL_SCENARIOS = 2


@needs_teacher
def test_teacher_policy_loc_is_decoded_label(
        teacher: Teacher,
    ) -> None:
    """loc is the decoded label, value 0, deterministic sample = loc."""
    obs = rollout_obs(teacher, 20, seed=POLICY_SEED)
    distribution, value = ContinuousPolicy(teacher).forward_eval(obs)
    steering = ego_state_from_obs(obs)[0][:, STEERING]
    expected = env_action_from_control(teacher.label(obs, steering), steering)
    assert torch.equal(distribution.loc, expected)
    assert torch.all(distribution.scale == ACTION_SCALE)
    assert torch.all(value == 0) and value.shape == (obs.shape[0], 1)
    action, _, _ = pufferlib.pytorch.sample_logits(
        distribution, deterministic=True
    )
    assert torch.equal(action.view_as(distribution.loc), distribution.loc)
    assert float(distribution.loc.abs().max()) <= 1.0


@needs_teacher
def test_policy_drives_env_like_the_loop(
        teacher: Teacher,
        device: str,
    ) -> None:
    """Same seed: ContinuousPolicy and BC collection send equal actions."""
    policy = ContinuousPolicy(teacher)
    env = make_small_env(POLICY_SEED)
    obs, _ = env.reset(seed=POLICY_SEED)
    policy_actions = []
    for _ in range(COMPARED_STEPS):
        obs_tensor = torch.from_numpy(obs).to(device)
        distribution, _ = policy.forward_eval(obs_tensor)
        action = distribution.loc.cpu().numpy()
        policy_actions.append(action)
        obs, _, _, _, _ = env.step(action)
    env.close()

    recording_env = RecordingEnv(make_small_env(POLICY_SEED))
    config = DistillConfig(
        seed=POLICY_SEED,
        device=device,
        iterations=1,
        env_steps_per_iteration=COMPARED_STEPS,
        gradient_steps_per_iteration=0,
        batch_size=16,
        buffer_size=100_000,
        eval_interval_labels=0,
    )
    obs_dim = recording_env.env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, device)
    distill(recording_env, teacher, SpyStudent().to(device), buffer, config)
    recording_env.env.close()
    assert np.array_equal(np.stack(policy_actions), np.stack(
        recording_env.actions
    ))


@pytest.mark.slow
@needs_teacher
def test_evaluator_runs_continuous_policy(
        teacher: Teacher,
        distill_args: dict,
    ) -> None:
    """validation_gigaflow runs ContinuousPolicy(teacher) to the end."""
    evaluator = DistillEvaluator(distill_args, num_scenarios=EVAL_SCENARIOS)
    metrics = evaluator.evaluate(teacher)
    for name in ("num_goals_reached", "collision_rate", "offroad_rate"):
        assert name in metrics and np.isfinite(metrics[name])
