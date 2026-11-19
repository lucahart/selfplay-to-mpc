"""Shared fixtures and test doubles for the distillation tests."""

import os

import numpy as np
import pytest
import torch

from pufferlib.distill.bicycle import NU, U_HI
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.env import (
    TEACHER_CHECKPOINT,
    load_distill_args,
    make_env,
)
from pufferlib.distill.students import Student
from pufferlib.distill.teacher import Teacher
from pufferlib.ocean.drive.drive import Drive

SMALL_AGENT_COUNT = 64
# The teacher's acceleration table never contains this value, so an
# executed action with it was chosen by the spy student
SPY_BRAKE_CONTROL = (-3.5, 0.0)
# Throttle and steering that leave the road within a few seconds
SPY_CRASH_CONTROL = (3.5, 0.6)

needs_teacher = pytest.mark.skipif(
    not os.path.exists(TEACHER_CHECKPOINT),
    reason="teacher checkpoint not found",
)


def pytest_configure(
        config: pytest.Config,
    ) -> None:
    """
    Register the slow marker.

    Args:
        config (pytest.Config): pytest configuration.
    """
    config.addinivalue_line(
        "markers", "slow: runs the real evaluator or a script"
    )


@pytest.fixture(scope="session")
def device() -> str:
    """
    Returns:
        str: "cuda" if available, else "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def distill_args() -> dict:
    """
    Returns:
        dict: config from load_distill_args.
    """
    return load_distill_args()


@pytest.fixture(scope="session")
def teacher(
        distill_args: dict,
        device: str,
    ) -> Teacher:
    """
    Args:
        distill_args (dict): config.
        device (str): torch device.

    Returns:
        Teacher: the s4 seed-2 teacher.
    """
    if not os.path.exists(TEACHER_CHECKPOINT):
        pytest.skip("teacher checkpoint not found")
    return Teacher(TEACHER_CHECKPOINT, distill_args, device)


def make_small_env(
        seed: int,
        extra_overrides: tuple[str, ...] = (),
        agent_count: int = SMALL_AGENT_COUNT,
    ) -> Drive:
    """
    Continuous-action env with few agents.

    Args:
        seed (int): env seed.
        extra_overrides (tuple[str, ...]): further Hydra env overrides.
        agent_count (int): agents in the env.

    Returns:
        Drive: the env.
    """
    env, _ = make_env("continuous", seed, agent_count, extra_overrides)
    return env


class RecordingBuffer(LabelBuffer):
    """LabelBuffer that also keeps a CPU copy of every add() call."""

    def __init__(
            self,
            capacity: int,
            obs_dim: int,
            device: str,
        ) -> None:
        """
        Args:
            capacity (int): buffer capacity.
            obs_dim (int): observation width.
            device (str): torch device.
        """
        super().__init__(capacity, obs_dim, device)
        self.added_obs = []
        self.added_labels = []

    def add(
            self,
            obs: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
        """
        Record, then add.

        Args:
            obs (torch.Tensor): (n,obs_dim) observations.
            labels (torch.Tensor): (n,2) labels.
        """
        self.added_obs.append(obs.detach().cpu().clone())
        self.added_labels.append(labels.detach().cpu().clone())
        super().add(obs, labels)


class SpyStudent(Student):
    """Student that drives with one constant control and records how the
    loop calls it."""

    def __init__(
            self,
            control: tuple[float, float] = SPY_BRAKE_CONTROL,
        ) -> None:
        """
        Args:
            control (tuple[float, float]): (accel, steering_rate) returned
                by act() for every agent.
        """
        super().__init__()
        self.register_buffer("control", torch.tensor(control))
        self.register_buffer("control_bound", U_HI.clone())
        # predict() needs a parameter for the optimizer to own
        self.bias = torch.nn.Parameter(torch.zeros(NU))
        self.act_calls = 0
        self.reset_masks = []

    def predict(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            torch.Tensor: (B,2) the learned bias for every row.
        """
        return self.bias.expand(obs.shape[0], NU)

    def act(
            self,
            obs: torch.Tensor,
            driven_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): (B,obs_dim) observations.
            driven_mask (torch.Tensor): (B,) bool, unused.

        Returns:
            torch.Tensor: (B,2) the constant control.
        """
        self.act_calls += 1
        return self.control.to(obs.device).expand(obs.shape[0], NU)

    def reset_plans(
            self,
            new_episode_mask: torch.Tensor,
        ) -> None:
        """
        Args:
            new_episode_mask (torch.Tensor): (B,) bool.
        """
        self.reset_masks.append(new_episode_mask.detach().cpu().clone())


class RecordingEnv:
    """Wraps a Drive env and records observations, actions and the flags
    of every step."""

    def __init__(
            self,
            env: Drive,
        ) -> None:
        """
        Args:
            env (Drive): wrapped env.
        """
        self.env = env
        self.observations = []  # obs before each step, then the last obs
        self.actions = []
        self.terminals = []
        self.truncations = []
        self.ticks = []

    @property
    def num_agents(self) -> int:
        """
        Returns:
            int: agents in the wrapped env.
        """
        return self.env.num_agents

    @property
    def tick(self) -> int:
        """
        Returns:
            int: the wrapped env's tick (0 right after a map resample).
        """
        return self.env.tick

    def get_global_agent_state(self) -> dict:
        """
        Returns:
            dict: the wrapped env's world poses (Drive's keys).
        """
        return self.env.get_global_agent_state()

    def reset(
            self,
            seed: int,
        ) -> tuple[np.ndarray, list]:
        """
        Args:
            seed (int): env seed.

        Returns:
            tuple[np.ndarray, list]: observations and infos.
        """
        obs, infos = self.env.reset(seed=seed)
        self.observations.append(obs.copy())
        return obs, infos

    def step(
            self,
            action: np.ndarray,
        ) -> tuple:
        """
        Args:
            action (np.ndarray): (B,2) continuous actions.

        Returns:
            tuple: the wrapped env's step result.
        """
        self.actions.append(np.array(action, copy=True))
        obs, rewards, terminals, truncations, infos = self.env.step(action)
        self.observations.append(obs.copy())
        self.terminals.append(np.asarray(terminals).astype(bool).copy())
        self.truncations.append(np.asarray(truncations).astype(bool).copy())
        self.ticks.append(self.env.tick)
        return obs, rewards, terminals, truncations, infos

    def new_episode_masks(self) -> list[np.ndarray]:
        """
        Returns:
            list[np.ndarray]: per step, (B,) bool, agents whose episode
                restarted after that step (env reset or map resample).
        """
        return [
            truncations | (tick == 0)
            for truncations, tick in zip(self.truncations, self.ticks)
        ]
