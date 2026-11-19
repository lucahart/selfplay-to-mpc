"""An unconditioned Drive checkpoint loads into a policy with
env.reward_conditioning=true and acts exactly as before."""

import pathlib
import types

import pytest
import torch

from pufferlib import pufferl
from pufferlib.distill.env import load_distill_args, make_env
from pufferlib.ocean import torch as ocean_torch
from pufferlib.ocean.drive import binding

CONDITIONING_SEED = 5
PROBE_AGENT_COUNT = 64
# Separate actor and critic backbones, so both context encoders widen
POLICY_OVERRIDES = (
    "policy.shared_network=false",
    "policy.backbone_layer_norm=true",
    "policy.backbone_hidden_size=128",
)
CONDITIONING_OVERRIDE = "env.reward_conditioning=true"


def build_policy(
        conditioned: bool,
    ) -> tuple[ocean_torch.Drive, torch.Tensor, dict]:
    """
    Build a randomly initialized Drive policy and reset observations of
    its env.

    Args:
        conditioned (bool): build for env.reward_conditioning=true.

    Returns:
        tuple[ocean_torch.Drive, torch.Tensor, dict]: the policy, the
            (B,obs_dim) reset observations and the composed config.
    """
    overrides = POLICY_OVERRIDES
    if conditioned:
        overrides += (CONDITIONING_OVERRIDE,)
    env, args = make_env(
        "discrete", CONDITIONING_SEED, PROBE_AGENT_COUNT, overrides
    )
    obs, _ = env.reset(seed=CONDITIONING_SEED)
    torch.manual_seed(CONDITIONING_SEED)
    policy = ocean_torch.Drive(env, **args["policy"])
    env.close()
    return policy, torch.from_numpy(obs), args


def insert_reward_coefs(
        obs: torch.Tensor,
        reward_coefs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Unconditioned observations with reward coefficients written where
    drive.h puts them: right after the ego features.

    Args:
        obs (torch.Tensor): (B,obs_dim) unconditioned observations.
        reward_coefs (torch.Tensor): (B,NUM_REWARD_COEFS) values.

    Returns:
        torch.Tensor: (B,obs_dim+NUM_REWARD_COEFS) observations.
    """
    ego_end = binding.EGO_FEATURES
    return torch.cat([obs[:, :ego_end], reward_coefs, obs[:, ego_end:]], 1)


def test_reward_coefs_follow_the_ego_features() -> None:
    """The conditioned env's obs is the unconditioned one with the
    coefficients inserted after the ego features (same seed)."""
    _, plain_obs, _ = build_policy(conditioned=False)
    _, conditioned_obs, _ = build_policy(conditioned=True)
    ego_end = binding.EGO_FEATURES
    reward_coefs = conditioned_obs[
        :, ego_end:ego_end + binding.NUM_REWARD_COEFS
    ]
    torch.testing.assert_close(
        insert_reward_coefs(plain_obs, reward_coefs), conditioned_obs
    )


def test_widened_policy_outputs_match_at_any_coefs() -> None:
    """Zero-initialized coefficient inputs: same logits and values as the
    unconditioned policy, whatever the coefficients are."""
    plain_policy, plain_obs, _ = build_policy(conditioned=False)
    conditioned_policy, _, _ = build_policy(conditioned=True)
    conditioned_policy.load_state_dict(
        ocean_torch.add_reward_conditioning_inputs(plain_policy.state_dict())
    )
    generator = torch.Generator().manual_seed(CONDITIONING_SEED)
    reward_coefs = 4 * torch.rand(
        plain_obs.shape[0], binding.NUM_REWARD_COEFS, generator=generator
    ) - 2
    with torch.no_grad():
        plain_logits, plain_value = plain_policy(plain_obs)
        conditioned_logits, conditioned_value = conditioned_policy(
            insert_reward_coefs(plain_obs, reward_coefs)
        )
    for plain_head, conditioned_head in zip(plain_logits, conditioned_logits):
        torch.testing.assert_close(conditioned_head, plain_head)
    torch.testing.assert_close(conditioned_value, plain_value)


def test_load_policy_widens_only_with_the_flag(
        tmp_path: pathlib.Path,
    ) -> None:
    """load_policy loads an unconditioned checkpoint into a conditioned
    policy with load_model_add_reward_conditioning, and fails without
    it or when the checkpoint is already conditioned."""
    plain_policy, _, _ = build_policy(conditioned=False)
    checkpoint_path = tmp_path / "plain.pt"
    torch.save(plain_policy.state_dict(), checkpoint_path)
    env, _ = make_env(
        "discrete",
        CONDITIONING_SEED,
        PROBE_AGENT_COUNT,
        POLICY_OVERRIDES + (CONDITIONING_OVERRIDE,),
    )
    vecenv = types.SimpleNamespace(driver_env=env)
    args = load_distill_args(
        POLICY_OVERRIDES
        + (
            CONDITIONING_OVERRIDE,
            "train.device=cpu",
            f"load_model_path={checkpoint_path}",
        )
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        pufferl.load_policy(args, vecenv)

    args["load_model_add_reward_conditioning"] = True
    policy = pufferl.load_policy(args, vecenv)
    weight_key = "actor_backbone." + ocean_torch.CONTEXT_INPUT_WEIGHT_SUFFIX
    loaded_weight = policy.state_dict()[weight_key]
    assert torch.all(loaded_weight[:, :binding.NUM_REWARD_COEFS] == 0)

    # A conditioned checkpoint must not be widened a second time
    torch.save(policy.state_dict(), checkpoint_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        pufferl.load_policy(args, vecenv)
    env.close()


def test_flag_needs_reward_conditioning(
        tmp_path: pathlib.Path,
    ) -> None:
    """The flag without env.reward_conditioning is a config error."""
    plain_policy, _, _ = build_policy(conditioned=False)
    checkpoint_path = tmp_path / "plain.pt"
    torch.save(plain_policy.state_dict(), checkpoint_path)
    env, _ = make_env(
        "discrete", CONDITIONING_SEED, PROBE_AGENT_COUNT, POLICY_OVERRIDES
    )
    args = load_distill_args(
        POLICY_OVERRIDES
        + ("train.device=cpu", f"load_model_path={checkpoint_path}")
    )
    args["load_model_add_reward_conditioning"] = True
    with pytest.raises(ValueError, match="reward_conditioning"):
        pufferl.load_policy(args, types.SimpleNamespace(driver_env=env))
    env.close()


def test_state_dict_without_context_encoder_is_rejected() -> None:
    """No context encoder (no target features): nothing to widen."""
    with pytest.raises(ValueError, match="no context encoder"):
        ocean_torch.add_reward_conditioning_inputs(
            {"actor_backbone.ego_encoder.0.weight": torch.zeros(4, 10)}
        )
