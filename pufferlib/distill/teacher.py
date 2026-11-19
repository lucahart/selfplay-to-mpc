"""Frozen PPO teacher that labels states with continuous controls."""

import os

import torch

from pufferlib.distill.bicycle import control_from_discrete
from pufferlib.distill.env import make_env
from pufferlib.ocean import torch as ocean_torch
from pufferlib.pufferl import clean_policy_state_dict

# A small discrete env is only needed for the policy's spaces
PROBE_AGENT_COUNT = 64
PROBE_SEED = 0


class Teacher:
    """Discrete PPO policy (torch.Drive) loaded from a checkpoint; its
    argmax, decoded to (accel, steering_rate), is the label."""

    def __init__(
            self,
            checkpoint_path: str,
            args: dict,
            device: str,
        ) -> None:
        """
        Build torch.Drive against the discrete action space and load the
        checkpoint strictly, so a wrong or broken file fails here.

        Args:
            checkpoint_path (str): path to a model_*.pt state dict.
            args (dict): config from load_distill_args (policy section,
                and env.reward_conditioning for the observation layout).
            device (str): torch device for the policy.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"teacher checkpoint {checkpoint_path}")
        # Reward conditioning widens the observation the policy is built for
        reward_conditioning = str(bool(args["env"]["reward_conditioning"]))
        probe_env, _ = make_env(
            "discrete",
            PROBE_SEED,
            PROBE_AGENT_COUNT,
            (f"env.reward_conditioning={reward_conditioning.lower()}",),
        )
        policy_cls = getattr(ocean_torch, args["policy_name"])
        self.policy = policy_cls(probe_env, **args["policy"]).to(device)
        probe_env.close()
        state_dict = torch.load(checkpoint_path, map_location=device)
        # strict=True rejects a continuous-head or other-arch checkpoint
        self.policy.load_state_dict(clean_policy_state_dict(state_dict))
        self.policy.eval()
        self.policy.requires_grad_(False)
        self.device = device

    def action_indices(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        The teacher's deterministic discrete action.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations on self.device.

        Returns:
            torch.Tensor: (B,) action index, accel_idx * 9 + steer_idx.
        """
        with torch.no_grad():
            logits, _ = self.policy.forward_eval(obs.float())
        # Discrete heads come back as a tuple with one logits tensor
        return logits[0].argmax(-1)

    def label(
            self,
            obs: torch.Tensor,
            steering_rad: torch.Tensor,
        ) -> torch.Tensor:
        """
        Continuous label for each agent: what PD does with the argmax.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations on self.device.
            steering_rad (torch.Tensor): (B,) current steering angle [rad].

        Returns:
            torch.Tensor: (B,2) control (accel [m/s^2],
                steering_rate [rad/s]).
        """
        return control_from_discrete(self.action_indices(obs), steering_rad)
