"""Wraps a teacher or student as a policy the existing evaluators can run
in the continuous-action env."""

import numpy as np
import torch

from pufferlib.distill.bicycle import (
    ego_state_from_obs,
    env_action_from_control,
    STEERING,
)
from pufferlib.distill.students import MPCStudent, Student
from pufferlib.distill.teacher import Teacher
from pufferlib.distill.tube import poses_from_global_state

# Evaluators take the mean (deterministic=True); the scale only has to be
# a valid Normal
ACTION_SCALE = 1e-3


class ContinuousPolicy(torch.nn.Module):
    """Policy interface (forward_eval -> Normal, value) around a Teacher or
    a Student, emitting continuous env actions."""

    def __init__(
            self,
            controller: Teacher | Student,
        ) -> None:
        """
        Args:
            controller (Teacher | Student): what decides the controls.
        """
        super().__init__()
        if not isinstance(controller, (Teacher, Student)):
            raise TypeError("controller must be a Teacher or a Student")
        self.controller = controller

    def forward_eval(
            self,
            obs: torch.Tensor,
            state: dict | None = None,
        ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """
        Env action for every agent, as the mean of a narrow Normal.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.
            state (dict | None): recurrent state; unused.

        Returns:
            tuple[torch.distributions.Normal, torch.Tensor]: action
                distribution with loc (B,2) in [-1,1]^2, and value (B,1)
                zeros.
        """
        obs = obs.float()
        x0, _ = ego_state_from_obs(obs)
        steering = x0[:, STEERING]
        if isinstance(self.controller, Teacher):
            control = self.controller.label(obs, steering)
        else:
            driven_mask = torch.ones(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
            control = self.controller.act(obs, driven_mask)
        action = env_action_from_control(control, steering)
        value = torch.zeros(obs.shape[0], 1, device=obs.device)
        return torch.distributions.Normal(action, ACTION_SCALE), value

    @property
    def needs_agent_poses(self) -> bool:
        """
        Returns:
            bool: the evaluator must call observe_agents before every
                forward_eval (an MPC student with a previous-plan tube).
        """
        return (
            isinstance(self.controller, Student)
            and self.controller.needs_agent_poses
        )

    @property
    def needs_route_window(self) -> bool:
        """
        Returns:
            bool: observe_agents must also get the route windows (an MPC
                student with a route-anchored tube).
        """
        return (
            isinstance(self.controller, Student)
            and self.controller.needs_route_window
        )

    def observe_agents(
            self,
            global_state: dict,
            new_episode: np.ndarray,
            route_window: tuple[np.ndarray, np.ndarray] | None = None,
        ) -> None:
        """
        Hand the student the agents' world poses (and route windows) for
        the coming forward_eval, and forget the plans of agents whose
        episode ended.

        Args:
            global_state (dict): Drive.get_global_agent_state() of the env
                the observations come from: 'x', 'y', 'heading' (B,).
            new_episode (np.ndarray): (B,) bool, agents done since the last
                forward_eval (all at a reset).
            route_window (tuple | None): Drive.get_route_window() of the
                same env, points (B,R,2) and validity (B,R); needed when
                needs_route_window.
        """
        device = self.controller.control_bound.device
        poses = poses_from_global_state(global_state, device)
        self.controller.reset_plans(
            torch.as_tensor(new_episode, dtype=torch.bool, device=device)
        )
        self.controller.set_agent_poses(poses)
        if route_window is not None:
            window, window_valid = route_window
            self.controller.set_route_window(
                torch.as_tensor(window, device=device),
                torch.as_tensor(window_valid, device=device),
            )

    def render_overlay(
            self,
        ) -> np.ndarray | None:
        """
        What the MPC student planned with at the last forward_eval, for the
        video overlay (overlay.py). The first call switches the recording
        on, so the first frame of a render pass has no overlay.

        Returns:
            np.ndarray | None: (B,OVERLAY_FLOATS) float32 rows, or None for
                a teacher, an MLP student or before the first recorded step.
        """
        if not isinstance(self.controller, MPCStudent):
            return None
        self.controller.record_overlay = True
        if self.controller.last_overlay is None:
            return None
        return self.controller.last_overlay.cpu().numpy()

    def forward(
            self,
            obs: torch.Tensor,
            state: dict | None = None,
        ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """
        Same as forward_eval.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.
            state (dict | None): recurrent state; unused.

        Returns:
            tuple[torch.distributions.Normal, torch.Tensor]: see
                forward_eval.
        """
        return self.forward_eval(obs, state)
