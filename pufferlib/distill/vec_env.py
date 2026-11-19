"""Several Drive envs in worker processes behind one Drive-like interface.

The workers hold only CPU work (env stepping); observations, actions,
rewards and flags live in shared memory, so a step costs one pipe message
per worker and no pickled arrays. Stepping is synchronous and the workers
are always visited in index order, so a seed and a worker count reproduce
the same trajectories."""

import traceback

import gymnasium
import numpy as np
import torch
import torch.multiprocessing as mp

from pufferlib.distill.env import make_env
from pufferlib.ocean.drive.drive import ROUTE_WINDOW_POINT_COUNT

START_METHOD = "spawn"
COMMAND_RESET = "reset"
COMMAND_STEP = "step"
COMMAND_CLOSE = "close"
REPLY_DONE = "done"
REPLY_ERROR = "error"
REPLY_POLL_SECONDS = 1.0
# A worker that does not answer for this long is treated as hung
REPLY_WAIT_LIMIT_SECONDS = 600.0
CLOSE_JOIN_SECONDS = 10.0
MAX_WORKER_COMMANDS = 10**9
ACTION_DIM = 2
# World pose per agent row: x, y, heading (Drive.get_global_agent_state)
POSE_KEYS = ("x", "y", "heading")


def worker_main(
        connection,
        index: int,
        seed: int,
        row_count: int,
        env_overrides: tuple[str, ...],
        first_row: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        truncations: torch.Tensor,
        poses: torch.Tensor,
        route_window: torch.Tensor,
        route_valid: torch.Tensor,
        replan_failed: torch.Tensor,
    ) -> None:
    """
    Process entry point: own one Drive of row_count agents and answer
    commands until closed. Any exception is sent to the parent.

    Args:
        connection: pipe end to the parent.
        index (int): worker index; reset seeds are offset by it.
        seed (int): env seed.
        row_count (int): agents of this worker's Drive.
        env_overrides (tuple[str, ...]): Hydra env overrides.
        first_row (int): first agent row of this worker in the shared
            arrays.
        observations (torch.Tensor): shared (B,obs_dim) fp32.
        actions (torch.Tensor): shared (B,2) fp32.
        rewards (torch.Tensor): shared (B,) fp32.
        terminals (torch.Tensor): shared (B,) bool.
        truncations (torch.Tensor): shared (B,) bool.
        poses (torch.Tensor): shared (B,3) fp32 world poses after the
            last reset or step.
        route_window (torch.Tensor): shared (B,R,2) fp32 route windows
            (Drive.get_route_window) after the last reset or step.
        route_valid (torch.Tensor): shared (B,R) bool, their validity.
        replan_failed (torch.Tensor): shared (B,) bool failed-replan flags
            (Drive.get_replay_replan_failed) after the last reset or step.
    """
    try:
        torch.set_num_threads(1)
        env, _ = make_env("continuous", seed, row_count, env_overrides)
        rows = slice(first_row, first_row + row_count)
        observation_view = observations[rows].numpy()
        action_view = actions[rows].numpy()
        reward_view = rewards[rows].numpy()
        terminal_view = terminals[rows].numpy()
        truncation_view = truncations[rows].numpy()
        pose_view = poses[rows].numpy()
        route_window_view = route_window[rows].numpy()
        route_valid_view = route_valid[rows].numpy()
        replan_failed_view = replan_failed[rows].numpy()
        connection.send((
            REPLY_DONE,
            env.single_observation_space.shape[0],
            [],
        ))
        for _ in range(MAX_WORKER_COMMANDS):
            command = connection.recv()
            if command[0] == COMMAND_CLOSE:
                break
            if command[0] == COMMAND_RESET:
                observation, infos = env.reset(seed=command[1] + index)
                terminal_view[:] = False
                truncation_view[:] = False
                reward_view[:] = 0.0
            else:
                observation, reward, terminal, truncation, infos = env.step(
                    action_view
                )
                reward_view[:] = reward
                terminal_view[:] = terminal
                truncation_view[:] = truncation
            observation_view[:] = observation
            global_state = env.get_global_agent_state()
            for pose_idx, key in enumerate(POSE_KEYS):
                pose_view[:, pose_idx] = global_state[key]
            window, valid = env.get_route_window()
            route_window_view[:] = window
            route_valid_view[:] = valid
            replan_failed_view[:] = env.get_replay_replan_failed()
            connection.send((REPLY_DONE, env.tick, list(infos)))
        env.close()
    except BaseException:
        connection.send((REPLY_ERROR, traceback.format_exc(), []))


class VecDrive:
    """Drive-like env over worker_count Drives of agent_count //
    worker_count agents each. Arrays returned by reset() and step() are
    views of shared memory: copy them before the next call."""

    def __init__(
            self,
            agent_count: int,
            worker_count: int,
            seed: int,
            env_overrides: tuple[str, ...] = (),
            worker_env_overrides: tuple[tuple[str, ...], ...] = (),
        ) -> None:
        """
        Start the workers and wait until every one built its env.

        Args:
            agent_count (int): total agents; divisible by worker_count.
            worker_count (int): number of Drive processes.
            seed (int): worker w is seeded seed + w.
            env_overrides (tuple[str, ...]): Hydra env overrides.
            worker_env_overrides (tuple[tuple[str, ...], ...]): per
                worker, overrides added after env_overrides (e.g. another
                simulation mode); () = every worker alike.
        """
        if worker_count <= 0 or agent_count % worker_count != 0:
            raise ValueError(
                f"agent_count {agent_count} must be a positive multiple of "
                f"worker_count {worker_count}"
            )
        if worker_env_overrides and len(worker_env_overrides) != worker_count:
            raise ValueError(
                f"{len(worker_env_overrides)} worker override tuples for "
                f"{worker_count} workers"
            )
        self.num_agents = agent_count
        self.worker_count = worker_count
        self.rows_per_worker = agent_count // worker_count
        self.workers = []
        self.connections = []
        self.closed = False
        context = mp.get_context(START_METHOD)
        self.actions = torch.zeros(agent_count, ACTION_DIM)
        self.rewards = torch.zeros(agent_count)
        self.terminals = torch.zeros(agent_count, dtype=torch.bool)
        self.truncations = torch.zeros(agent_count, dtype=torch.bool)
        self.poses = torch.zeros(agent_count, len(POSE_KEYS))
        self.route_window = torch.zeros(
            agent_count, ROUTE_WINDOW_POINT_COUNT, 2
        )
        self.route_valid = torch.zeros(
            agent_count, ROUTE_WINDOW_POINT_COUNT, dtype=torch.bool
        )
        self.replan_failed = torch.zeros(agent_count, dtype=torch.bool)
        self.worker_ticks = np.zeros(worker_count, dtype=np.int64)
        self.observations = self.build_observations(env_overrides)
        for tensor in (
            self.actions, self.rewards, self.terminals, self.truncations,
            self.poses, self.route_window, self.route_valid,
            self.replan_failed,
        ):
            tensor.share_memory_()
        self.observations.share_memory_()
        for index in range(worker_count):
            overrides = env_overrides + (
                worker_env_overrides[index] if worker_env_overrides else ()
            )
            parent_end, child_end = context.Pipe()
            worker = context.Process(
                target=worker_main,
                args=(
                    child_end,
                    index,
                    seed + index,
                    self.rows_per_worker,
                    overrides,
                    index * self.rows_per_worker,
                    self.observations,
                    self.actions,
                    self.rewards,
                    self.terminals,
                    self.truncations,
                    self.poses,
                    self.route_window,
                    self.route_valid,
                    self.replan_failed,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)
            self.connections.append(parent_end)
        replies = self.gather()
        # The shared observations are sized by a probe of env_overrides; a
        # worker mode with another layout would write past its rows
        widths = {reply[0] for reply in replies}
        if widths != {self.observations.shape[1]}:
            self.close()
            raise ValueError(
                f"worker observation widths {sorted(widths)} differ from "
                f"{self.observations.shape[1]}"
            )
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(replies[0][0],),
            dtype=np.float32,
        )

    def build_observations(
            self,
            env_overrides: tuple[str, ...],
        ) -> torch.Tensor:
        """
        Observation storage; its width comes from the env config, which a
        throwaway Drive of the smallest size reports.

        Args:
            env_overrides (tuple[str, ...]): Hydra env overrides.

        Returns:
            torch.Tensor: zeros (B,obs_dim) fp32.
        """
        probe_env, _ = make_env(
            "continuous", 0, self.rows_per_worker, env_overrides
        )
        width = probe_env.single_observation_space.shape[0]
        probe_env.close()
        return torch.zeros(self.num_agents, width)

    def gather(self) -> list:
        """
        One reply from every worker, in worker order; raises the worker's
        traceback if one failed and kills the rest.

        Returns:
            list: per worker, the payload after the reply tag.
        """
        replies = []
        polls = int(REPLY_WAIT_LIMIT_SECONDS / REPLY_POLL_SECONDS)
        for index, connection in enumerate(self.connections):
            for _ in range(polls):
                if connection.poll(REPLY_POLL_SECONDS):
                    break
                if not self.workers[index].is_alive():
                    self.close()
                    raise RuntimeError(f"env worker {index} died")
            else:
                self.close()
                raise TimeoutError(f"env worker {index} did not answer")
            reply = connection.recv()
            if reply[0] == REPLY_ERROR:
                self.close()
                raise RuntimeError(f"env worker {index} failed:\n{reply[1]}")
            replies.append(reply[1:])
        return replies

    @property
    def tick(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (B,) steps since each agent's env (re)sampled its
                maps; 0 right after a resample.
        """
        return np.repeat(self.worker_ticks, self.rows_per_worker)

    def get_global_agent_state(self) -> dict:
        """
        World poses of every agent row after the last reset or step, as
        Drive.get_global_agent_state names them (pose keys only).

        Returns:
            dict: 'x', 'y', 'heading' (B,) float32 copies.
        """
        poses = self.poses.numpy()
        return {
            key: poses[:, pose_idx].copy()
            for pose_idx, key in enumerate(POSE_KEYS)
        }

    def get_route_window(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Route windows of every agent row after the last reset or step, as
        Drive.get_route_window returns them.

        Returns:
            tuple: points (B,R,2) float32 copies [m] and validity (B,R)
                bool copies.
        """
        return (
            self.route_window.numpy().copy(),
            self.route_valid.numpy().copy(),
        )

    def get_replay_replan_failed(self) -> np.ndarray:
        """
        Failed-replan flags of every agent row after the last reset or
        step, as Drive.get_replay_replan_failed returns them.

        Returns:
            np.ndarray: (B,) bool copy.
        """
        return self.replan_failed.numpy().copy()

    def reset(
            self,
            seed: int = 0,
        ) -> tuple[np.ndarray, list]:
        """
        Reset every worker; worker w resets with seed + w, so worker 0
        equals a plain Drive reset with the same seed.

        Args:
            seed (int): reset seed of worker 0.

        Returns:
            tuple[np.ndarray, list]: (B,obs_dim) observations and infos.
        """
        self.send_all((COMMAND_RESET, seed))
        return self.collect_step_results()[0], []

    def step(
            self,
            actions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step every env once.

        Args:
            actions (np.ndarray): (B,2) actions.

        Returns:
            tuple: observations (B,obs_dim), rewards (B,), terminals (B,),
                truncations (B,) and the workers' infos.
        """
        self.actions.numpy()[:] = actions
        self.send_all((COMMAND_STEP,))
        return self.collect_step_results()

    def send_all(
            self,
            command: tuple,
        ) -> None:
        """
        Send a command to every worker; a worker that is gone ends the run.

        Args:
            command (tuple): command tag and arguments.
        """
        for index, connection in enumerate(self.connections):
            try:
                connection.send(command)
            except (BrokenPipeError, OSError):
                self.close()
                raise RuntimeError(f"env worker {index} died") from None

    def collect_step_results(self) -> tuple:
        """
        Wait for every worker's step or reset and expose the shared
        arrays.

        Returns:
            tuple: observations, rewards, terminals, truncations, infos.
        """
        infos = []
        for index, (tick, worker_infos) in enumerate(self.gather()):
            self.worker_ticks[index] = tick
            infos.extend(worker_infos)
        return (
            self.observations.numpy(),
            self.rewards.numpy(),
            self.terminals.numpy(),
            self.truncations.numpy(),
            infos,
        )

    def close(self) -> None:
        """
        Stop the workers; safe to call twice.
        """
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send((COMMAND_CLOSE,))
            except (BrokenPipeError, OSError):
                pass
        for worker in self.workers:
            worker.join(timeout=CLOSE_JOIN_SECONDS)
            if worker.is_alive():
                worker.terminate()
                worker.join()
