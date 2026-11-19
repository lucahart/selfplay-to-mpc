"""Online BC/DAgger loop: the teacher labels every live agent, a fraction
beta of agents follows the teacher for a whole episode and the rest
follow the student, and the student regresses on the labels."""

import dataclasses
import json
import math
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from pufferlib.distill.bicycle import (
    STEERING,
    U_HI,
    ego_state_from_obs,
    env_action_from_control,
)
from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.env import NUPLAN_SCENARIO_STEPS
from pufferlib.distill.policy import ContinuousPolicy
from pufferlib.distill.students import (
    GradChunk,
    MPCStudent,
    Student,
)
from pufferlib.distill.teacher import Teacher
from pufferlib.distill.tube import poses_from_global_state
from pufferlib.ocean.drive.drive import Drive

BETA_CONSTANT = "constant"
BETA_DAGGER = "dagger"
BETA_SCHEDULES = (BETA_CONSTANT, BETA_DAGGER)
# A prediction this close to a box edge counts as saturated
SATURATION_FRACTION_OF_BOUND = 0.99
CHECKPOINT_NAME = "student.pt"
# A file of this name in output_dir ends the run after the current
# iteration (save, close the workers, exit 0): a plateau stop that never
# kills a GPU process (killing MPS clients faulted the MPS server)
STOP_FILE_NAME = "stop.txt"
# Kept per evaluation: student_<labels_seen>.pt
EVAL_CHECKPOINT_TEMPLATE = "student_{labels_seen}.pt"
LOG_NAME = "log.jsonl"
# Rows per solve when the probe evaluates a set
PROBE_BATCH_SIZE = 4096
EVAL_LOG_NAME = "eval.jsonl"
DEFAULT_EVALUATOR = "validation_gigaflow"


@dataclasses.dataclass
class DistillConfig:
    """Settings of one distillation run. Validated on construction (they
    come from the command line)."""

    seed: int = 1
    device: str = "cuda"
    iterations: int = 100 # number of collect-then-train cycles
    env_steps_per_iteration: int = 32 # number of steps for each training update
    gradient_steps_per_iteration: int = 32 # randomly sample gradient steps
    batch_size: int = 4096 # number of samples per gradient update
    buffer_size: int = 1_000_000 # number of labels to choose from
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    # constant: beta = beta_constant (1.0 is BC). dagger: beta = 1 for
    # dagger_warmup_labels, then 1 - k / dagger_stage_count in stage k,
    # each stage lasting dagger_stage_labels
    beta_schedule: str = BETA_CONSTANT
    beta_constant: float = 1.0
    dagger_warmup_labels: int = 10_000_000
    dagger_stage_labels: int = 2_000_000
    dagger_stage_count: int = 5
    # 0 disables evaluation. With an evaluator, the student is evaluated
    # before training (0 labels) and whenever labels_seen crosses a
    # multiple of eval_interval_labels; each evaluation keeps a checkpoint
    eval_interval_labels: int = 2_000_000
    # Stop once this many labels were added (0: run all iterations)
    max_labels: int = 0
    # Fresh-gradient training (MPC student only): collection solves every
    # vehicle row once, whoever drives, and keeps the head-output gradient
    # of the loss; the update that uses it runs right after the step, so it
    # is never stale. Every chunks_per_update steps one update slot comes up:
    # a fraction fresh_fraction of the slots applies the shipped gradients,
    # the others do one replay step on the buffer (the chunks are dropped)
    fresh_gradient: bool = False
    fresh_fraction: float = 1.0
    chunks_per_update: int = 1
    # Env worker processes (vec_env.py): this many Drives of
    # agent_count / env_worker_count agents each behind one interface
    env_worker_count: int = 1
    # The first nuplan_worker_count of the env workers run nuPlan replay
    # self-play (env.NUPLAN_ENV_OVERRIDES), the rest CARLA gigaflow;
    # 0 = CARLA only
    nuplan_worker_count: int = 0
    # Forgetting probe (0 disables): frozen held-out sets of this many
    # labelled states, one from the start of the run and one per finished
    # DAgger stage (spread over the newest probe_window_labels rows); the
    # student's replay loss on each is logged every iteration
    probe_set_size: int = 0
    probe_window_labels: int = 65_536
    output_dir: str = ""
    # Continue from a save_checkpoint file: student, optimizer and
    # labels_seen are restored, so the DAgger schedule, max_labels and the
    # evaluation grid stay absolute; the start-of-run evaluation is skipped
    resume_checkpoint: str = ""

    def __post_init__(self) -> None:
        """
        Reject settings that would break the loop.
        """
        positive_ints = (
            "iterations",
            "env_steps_per_iteration",
            "batch_size",
            "buffer_size",
            "dagger_stage_labels",
            "dagger_stage_count",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative_ints = (
            "gradient_steps_per_iteration",
            "dagger_warmup_labels",
            "eval_interval_labels",
            "max_labels",
            "probe_set_size",
            "probe_window_labels",
        )
        for name in nonnegative_ints:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size must not exceed buffer_size")
        if not self.learning_rate > 0 or not self.max_grad_norm > 0:
            raise ValueError("learning_rate and max_grad_norm must be > 0")
        if self.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(f"beta_schedule must be one of {BETA_SCHEDULES}")
        if not 0.0 <= self.beta_constant <= 1.0:
            raise ValueError("beta_constant must be in [0, 1]")
        if not 0.0 <= self.fresh_fraction <= 1.0:
            raise ValueError("fresh_fraction must be in [0, 1]")
        if self.chunks_per_update <= 0 or self.env_worker_count <= 0:
            raise ValueError(
                "chunks_per_update and env_worker_count must be positive"
            )
        if not 0 <= self.nuplan_worker_count <= self.env_worker_count:
            raise ValueError(
                "nuplan_worker_count must be in 0..env_worker_count"
            )
        # One worker is a plain Drive, which has a single mode
        if self.nuplan_worker_count > 0 and self.env_worker_count < 2:
            raise ValueError("nuPlan workers need env_worker_count >= 2")


@dataclasses.dataclass
class EpisodeState:
    """Per-agent episode bookkeeping of the collection loop, and running
    sums of how episodes end since the last pop_episode_stats."""

    teacher_drives: torch.Tensor  # (B,) bool, fixed for the episode
    live: torch.Tensor  # (B,) bool, False once stopped/removed
    step_count: torch.Tensor  # (B,) int64, env steps into the episode
    nuplan_rows: torch.Tensor  # (B,) bool, rows of the nuPlan workers
    # 0-d tensors on the device, so counting adds no host sync per step
    sums: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


def add_episode_sums(
        episodes: EpisodeState,
        ended: torch.Tensor,
    ) -> None:
    """
    Count the episodes that end with this step by domain (nuPlan or
    CARLA rows) and driver: how many, their summed length, and for nuPlan
    how many ended before NUPLAN_SCENARIO_STEPS (termination_mode=1
    resets the scene once too many agents are inactive). Call after the
    step's step_count increment, before new drivers are drawn.

    Args:
        episodes (EpisodeState): driver, length and domain per row;
            sums updated in place.
        ended (torch.Tensor): (B,) bool, rows whose env truncated.
    """
    domains = (("nuplan", episodes.nuplan_rows),
               ("carla", ~episodes.nuplan_rows))
    drivers = (("teacher", episodes.teacher_drives),
               ("student", ~episodes.teacher_drives))
    early = episodes.step_count < NUPLAN_SCENARIO_STEPS
    for domain, domain_rows in domains:
        for driver, driver_rows in drivers:
            rows = ended & domain_rows & driver_rows
            prefix = f"{domain}_{driver}"
            counts = {
                f"{prefix}_ends": rows.sum(),
                f"{prefix}_steps": episodes.step_count[rows].sum(),
            }
            if domain == "nuplan":
                counts[f"{prefix}_early_ends"] = (rows & early).sum()
            for key, value in counts.items():
                episodes.sums[key] = episodes.sums.get(key, 0) + value


def pop_episode_stats(
        episodes: EpisodeState,
    ) -> dict:
    """
    Episode-end and goal-less label statistics since the last call, then
    clear the sums.

    Args:
        episodes (EpisodeState): sums to read; cleared.

    Returns:
        dict: per domain and driver, episodes ended and their mean length
            [steps] (NaN without ends); for nuPlan the share of ends
            before NUPLAN_SCENARIO_STEPS, and the share of nuPlan labels
            from agents whose replan failed (no goal ahead).
    """
    sums = {key: float(value) for key, value in episodes.sums.items()}
    episodes.sums = {}
    stats = {}
    for domain in ("nuplan", "carla"):
        for driver in ("teacher", "student"):
            prefix = f"{domain}_{driver}"
            ends = sums.get(f"{prefix}_ends", 0.0)
            stats[f"episodes/{prefix}_ends"] = ends
            stats[f"episodes/{prefix}_mean_steps"] = (
                sums.get(f"{prefix}_steps", 0.0) / ends if ends else math.nan
            )
            if domain == "nuplan":
                stats[f"episodes/{prefix}_early_fraction"] = (
                    sums.get(f"{prefix}_early_ends", 0.0) / ends
                    if ends else math.nan
                )
    nuplan_labels = sums.get("nuplan_labels", 0.0)
    stats["episodes/nuplan_goalless_label_fraction"] = (
        sums.get("nuplan_goalless_labels", 0.0) / nuplan_labels
        if nuplan_labels else math.nan
    )
    return stats


@dataclasses.dataclass
class DistillResult:
    """What a run leaves behind, for resuming and inspection."""

    labels_seen: int
    optimizer: torch.optim.Optimizer
    log_rows: list


def stage_at(
        config: DistillConfig,
        labels_seen: int,
    ) -> int:
    """
    DAgger stage: 0 during the constant schedule and the warmup, then 1,
    2, ... every dagger_stage_labels.

    Args:
        config (DistillConfig): run settings.
        labels_seen (int): labels added to the buffer so far.

    Returns:
        int: stage index.
    """
    if config.beta_schedule == BETA_CONSTANT:
        return 0
    if labels_seen < config.dagger_warmup_labels:
        return 0
    return 1 + (labels_seen - config.dagger_warmup_labels) // (
        config.dagger_stage_labels
    )


def beta_at(
        config: DistillConfig,
        labels_seen: int,
    ) -> float:
    """
    Probability that a new episode is driven by the teacher.

    Args:
        config (DistillConfig): run settings.
        labels_seen (int): labels added to the buffer so far.

    Returns:
        float: beta in [0, 1].
    """
    if config.beta_schedule == BETA_CONSTANT:
        return config.beta_constant
    stage = stage_at(config, labels_seen)
    return max(0.0, 1.0 - stage / config.dagger_stage_count)


def distillation_loss(
        predicted_control: torch.Tensor,
        label_control: torch.Tensor,
    ) -> torch.Tensor:
    """
    MSE in units of the control box, so accel and steering rate errors
    weigh the same.

    Args:
        predicted_control (torch.Tensor): (B,2) student controls.
        label_control (torch.Tensor): (B,2) teacher controls.

    Returns:
        torch.Tensor: scalar loss.
    """
    scale = U_HI.to(predicted_control)
    return torch.mean(((predicted_control - label_control) / scale) ** 2)


class ForgettingProbe:
    """Frozen held-out label sets and the student's replay loss on them.
    A rising loss on an old set while training continues means the
    student forgets that part of the state distribution."""

    def __init__(
            self,
            config: DistillConfig,
            labels_seen: int,
        ) -> None:
        """
        Args:
            config (DistillConfig): run settings.
            labels_seen (int): labels added before the run (stages
                already finished get no set).
        """
        self.config = config
        self.stage = stage_at(config, labels_seen)
        self.sets = {}

    def freeze_due_sets(
            self,
            buffer: LabelBuffer,
            labels_seen: int,
        ) -> None:
        """
        Freeze the "start" set once the buffer holds probe_set_size rows,
        and the set of every DAgger stage that finished since the last
        call from the newest probe_window_labels rows.

        Args:
            buffer (LabelBuffer): label source.
            labels_seen (int): labels added so far.
        """
        config = self.config
        if "start" not in self.sets and (
            buffer.filled_count >= config.probe_set_size
        ):
            self.sets["start"] = buffer.snapshot(config.probe_set_size)
        stage_now = stage_at(config, labels_seen)
        for finished in range(self.stage, stage_now):
            self.sets[f"stage_{finished}"] = buffer.snapshot(
                config.probe_set_size, config.probe_window_labels
            )
        self.stage = stage_now

    @torch.no_grad()
    def evaluate(
            self,
            student: Student,
        ) -> dict:
        """
        Replay-style loss (cold-started solve for the MPC student) on
        every frozen set.

        Args:
            student (Student): model being trained.

        Returns:
            dict: "probe/<set>" -> loss.
        """
        losses = {}
        for name, (obs, labels) in self.sets.items():
            total = 0.0
            for first in range(0, obs.shape[0], PROBE_BATCH_SIZE):
                obs_batch = obs[first:first + PROBE_BATCH_SIZE]
                label_batch = labels[first:first + PROBE_BATCH_SIZE]
                loss = distillation_loss(
                    student.predict(obs_batch), label_batch
                )
                total += float(loss) * obs_batch.shape[0]
            losses[f"probe/{name}"] = total / obs.shape[0]
        return losses


def make_optimizer(
        student: Student,
        config: DistillConfig,
    ) -> torch.optim.Optimizer:
    """
    Adam over the student's parameters.

    Args:
        student (Student): model being trained.
        config (DistillConfig): run settings.

    Returns:
        torch.optim.Optimizer: the optimizer.
    """
    return torch.optim.Adam(student.parameters(), lr=config.learning_rate)


def save_checkpoint(
        path: str,
        student: Student,
        optimizer: torch.optim.Optimizer,
        labels_seen: int,
        config: DistillConfig,
    ) -> None:
    """
    Save student, optimizer and progress.

    Args:
        path (str): output file.
        student (Student): trained model.
        optimizer (torch.optim.Optimizer): its optimizer.
        labels_seen (int): labels added so far.
        config (DistillConfig): run settings, stored for reference.
    """
    torch.save(
        {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "labels_seen": labels_seen,
            "config": dataclasses.asdict(config),
        },
        path,
    )


def load_checkpoint(
        path: str,
        student: Student,
        optimizer: torch.optim.Optimizer,
    ) -> int:
    """
    Restore student and optimizer in place.

    Args:
        path (str): checkpoint from save_checkpoint.
        student (Student): model with the same architecture.
        optimizer (torch.optim.Optimizer): optimizer over its parameters.

    Returns:
        int: labels_seen at save time.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(checkpoint["student"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["labels_seen"])


class DistillEvaluator:
    """Runs a pufferl evaluator (default validation_gigaflow) on a student
    through ContinuousPolicy."""

    def __init__(
            self,
            args: dict,
            evaluator_name: str = DEFAULT_EVALUATOR,
            num_scenarios: int | None = None,
        ) -> None:
        """
        Args:
            args (dict): config from load_distill_args.
            evaluator_name (str): [eval.<name>] section to run.
            num_scenarios (int | None): override of the section's count.
        """
        self.args = args
        self.evaluator_name = evaluator_name
        self.num_scenarios = num_scenarios

    def evaluate(
            self,
            controller: Teacher | Student,
            action_type: str = "continuous",
            render_results_dir: str | None = None,
            global_step: int | None = None,
        ) -> dict:
        """
        Run the evaluator once.

        Args:
            controller (Teacher | Student): what drives. A Teacher with
                action_type "discrete" runs its own policy unwrapped.
            action_type (str): env action type for the evaluation.
            render_results_dir (str | None): if set, also render videos;
                the evaluator writes them to
                <render_results_dir>/mp4/<evaluator_name>/.
            global_step (int | None): step stamped into video filenames.

        Returns:
            dict: the evaluator's metrics, plus a student's tube
                statistics over this evaluation (pop_tube_stats).
        """
        from pufferlib import pufferl

        args = dict(self.args)
        args["env"] = dict(args["env"], action_type=action_type)
        if render_results_dir is not None:
            args["render_results_dir"] = render_results_dir
        if action_type == "discrete":
            policy = controller.policy
        else:
            policy = ContinuousPolicy(controller)
        # Tube statistics of this evaluation only
        is_student = isinstance(controller, Student)
        if is_student:
            controller.pop_tube_stats()
            # The evaluator's agent rows differ from training's; training
            # resumes with its own warm starts afterwards
            training_plan_state = controller.take_plan_state()
        metrics = pufferl.eval(
            "puffer_drive",
            args=args,
            policy=policy,
            evaluator_name=self.evaluator_name,
            num_scenarios=self.num_scenarios,
            render=render_results_dir is not None,
            global_step=global_step,
        )
        if is_student:
            metrics.update(controller.pop_tube_stats())
            controller.restore_plan_state(training_plan_state)
        return metrics


def collect(
        env: Drive,
        obs: np.ndarray,
        teacher: Teacher,
        student: Student,
        buffer: LabelBuffer,
        episodes: EpisodeState,
        config: DistillConfig,
        labels_seen: int,
        generator: torch.Generator,
        chunk_sink: Callable[[GradChunk], None] | None = None,
    ) -> tuple[np.ndarray, int, dict]:
    """
    Run env_steps_per_iteration env steps: label every live agent, drive
    each agent by its episode's driver, and update the episode state.
    With `chunk_sink`, the MPC student solves every vehicle row in every
    step, whoever drives, and its gradient chunk is handed to the sink
    before the env steps, so the sink can update the weights the next
    step uses.

    Args:
        env (Drive): continuous-action env.
        obs (np.ndarray): (B,obs_dim) current observations.
        teacher (Teacher): labeler.
        student (Student): drives agents with teacher_drives False.
        buffer (LabelBuffer): receives (obs, label) of live agents.
        episodes (EpisodeState): updated in place.
        config (DistillConfig): run settings.
        labels_seen (int): labels added before this call.
        generator (torch.Generator): draws who drives new episodes.
        chunk_sink (Callable | None): called with each step's chunk; None
            keeps the replay collection.

    Returns:
        tuple[np.ndarray, int, dict]: next observations, updated
            labels_seen, and the latest env log (empty if none).
    """
    device = config.device
    env_log = {}
    for _ in range(config.env_steps_per_iteration):
        obs_tensor = torch.from_numpy(obs).to(device)
        if student.needs_agent_poses:
            student.set_agent_poses(
                poses_from_global_state(env.get_global_agent_state(), device)
            )
        if student.needs_route_window:
            window, window_valid = env.get_route_window()
            student.set_route_window(
                torch.from_numpy(window).to(device),
                torch.from_numpy(window_valid).to(device),
            )
        x0, _ = ego_state_from_obs(obs_tensor)
        steering = x0[:, STEERING]
        label = teacher.label(obs_tensor, steering)
        control = label
        if chunk_sink is not None:
            student_control, chunk = student.act_and_grad(
                obs_tensor, label, episodes.live, ~episodes.teacher_drives
            )
            if chunk is not None:
                chunk_sink(chunk)
            control = torch.where(
                episodes.teacher_drives[:, None], label, student_control
            )
        # With beta = 1 (BC) the student never runs during collection
        elif not bool(episodes.teacher_drives.all()):
            student_control = student.act(
                obs_tensor, ~episodes.teacher_drives
            )
            control = torch.where(
                episodes.teacher_drives[:, None], label, student_control
            )
        buffer.add(obs_tensor[episodes.live], label[episodes.live])
        labels_seen += int(episodes.live.sum())
        # Flags of the state the labels come from (after the last step)
        if bool(episodes.nuplan_rows.any()):
            replan_failed = torch.from_numpy(
                env.get_replay_replan_failed()
            ).to(device)
            nuplan_live = episodes.live & episodes.nuplan_rows
            sums = episodes.sums
            sums["nuplan_labels"] = (
                sums.get("nuplan_labels", 0) + nuplan_live.sum()
            )
            sums["nuplan_goalless_labels"] = (
                sums.get("nuplan_goalless_labels", 0)
                + (nuplan_live & replan_failed).sum()
            )

        action = env_action_from_control(control, steering)
        obs, _, terminals, truncations, infos = env.step(
            action.cpu().numpy()
        )

        # A stopped/removed agent is frozen until its env resets, so its
        # next observations carry no driving information
        terminal_mask = torch.from_numpy(
            np.asarray(terminals).astype(bool)
        ).to(device)
        truncation_mask = torch.from_numpy(
            np.asarray(truncations).astype(bool)
        ).to(device)
        # Env reset (truncation) or map resample (tick back to 0); a vector
        # env has one tick per agent
        new_episode = truncation_mask | torch.as_tensor(
            env.tick == 0, device=device
        )
        episodes.live = ~terminal_mask | new_episode
        episodes.step_count += 1
        add_episode_sums(episodes, truncation_mask)
        episodes.step_count[new_episode] = 0
        new_episode_count = int(new_episode.sum())
        if new_episode_count > 0:
            beta = beta_at(config, labels_seen)
            draws = torch.rand(
                new_episode_count, device=device, generator=generator
            )
            episodes.teacher_drives[new_episode] = draws < beta
            student.reset_plans(new_episode)
        for info in infos:
            if isinstance(info, dict) and "score" in info:
                env_log = info
    return obs, labels_seen, env_log


def saturation_stats(
        predicted_control: torch.Tensor,
        label_control: torch.Tensor,
        control_bound: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    How often predictions sit on the student's box edge, and how often on
    the edge opposite their label. The latter get zero gradient through
    an MPC solver, so a rising value means the student is stuck.

    Args:
        predicted_control (torch.Tensor): (B,2) student controls.
        label_control (torch.Tensor): (B,2) teacher controls.
        control_bound (torch.Tensor): (2,) the student's |u| bound.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: scalar fractions (over all
            entries) on the edge, and on the edge opposite a nonzero
            label.
    """
    bound = SATURATION_FRACTION_OF_BOUND * control_bound.to(predicted_control)
    on_bound = predicted_control.abs() >= bound
    opposite_sign = torch.sign(predicted_control) != torch.sign(label_control)
    wrong_side = on_bound & opposite_sign & (label_control != 0)
    return on_bound.float().mean(), wrong_side.float().mean()


def train_student(
        student: Student,
        optimizer: torch.optim.Optimizer,
        buffer: LabelBuffer,
        config: DistillConfig,
    ) -> dict:
    """
    gradient_steps_per_iteration minibatch steps on the buffer.

    Args:
        student (Student): model being trained.
        optimizer (torch.optim.Optimizer): its optimizer.
        buffer (LabelBuffer): source of (obs, label) samples.
        config (DistillConfig): run settings.

    Returns:
        dict: mean loss, saturation fraction and wrong-side saturation
            over the steps (NaN when no step ran).
    """
    student.train()
    losses, saturation_fractions, wrong_side_fractions = [], [], []
    for _ in range(config.gradient_steps_per_iteration):
        obs_batch, label_batch = buffer.sample(config.batch_size)
        predicted = student.predict(obs_batch)
        loss = distillation_loss(predicted, label_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            student.parameters(), config.max_grad_norm
        )
        optimizer.step()
        losses.append(loss.detach())
        saturation, wrong_side = saturation_stats(
            predicted.detach(), label_batch, student.control_bound
        )
        saturation_fractions.append(saturation)
        wrong_side_fractions.append(wrong_side)
    if not losses:
        return {
            "loss": math.nan,
            "saturation_fraction": math.nan,
            "wrong_side_saturation": math.nan,
        }
    return {
        "loss": float(torch.stack(losses).mean()),
        "saturation_fraction": float(torch.stack(saturation_fractions).mean()),
        "wrong_side_saturation": float(
            torch.stack(wrong_side_fractions).mean()
        ),
    }


def apply_fresh_update(
        student: MPCStudent,
        optimizer: torch.optim.Optimizer,
        group: list[GradChunk],
        config: DistillConfig,
    ) -> tuple[torch.Tensor, list[int]]:
    """
    One optimizer step from shipped gradients: re-run the network on each
    chunk's rows at the current weights, push the stored head-output
    gradient through it, clip, step. Raises the student's weight_version.

    Args:
        student (MPCStudent): model being trained.
        optimizer (torch.optim.Optimizer): its optimizer.
        group (list[GradChunk]): chunks of this update.
        config (DistillConfig): run settings.

    Returns:
        tuple: mean loss at collection (scalar tensor) and the age in
            updates of each chunk.
    """
    student.train()
    row_count = sum(chunk.obs.shape[0] for chunk in group)
    optimizer.zero_grad()
    ages = []
    for chunk in group:
        student.apply_cost_gradient(chunk, row_count)
        ages.append(student.weight_version - chunk.weight_version)
    torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
    optimizer.step()
    student.weight_version += 1
    loss = sum(chunk.loss_sum for chunk in group) / row_count
    return loss, ages


class FreshLearner:
    """Applies the gradients collection ships, as the chunks arrive. Every
    chunks_per_update chunks one update slot comes up: a fraction
    fresh_fraction of the slots applies the pending chunks, the others do
    one replay step on the buffer (the pending chunks are dropped). Every
    update raises the student's weight_version. A chunk is applied before
    the next env step, so its gradient is never stale."""

    def __init__(
            self,
            student: MPCStudent,
            optimizer: torch.optim.Optimizer,
            buffer: LabelBuffer,
            config: DistillConfig,
        ) -> None:
        """
        Args:
            student (MPCStudent): model being trained.
            optimizer (torch.optim.Optimizer): its optimizer.
            buffer (LabelBuffer): source of replay samples.
            config (DistillConfig): run settings.
        """
        self.student = student
        self.optimizer = optimizer
        self.buffer = buffer
        self.config = config
        self.replay_config = dataclasses.replace(
            config, gradient_steps_per_iteration=1
        )
        self.pending = []
        self.slot_count = 0
        self.fresh_losses, self.replay_losses = [], []
        self.ages, self.seconds = [], []

    def add(
            self,
            chunk: GradChunk,
        ) -> None:
        """
        Queue a chunk; run an update slot when chunks_per_update are
        pending.

        Args:
            chunk (GradChunk): chunk of the step that just ran.
        """
        self.pending.append(chunk)
        if len(self.pending) < self.config.chunks_per_update:
            return
        group, self.pending = self.pending, []
        self.slot_count += 1
        # Bresenham spreading of fresh slots over the run
        fresh_fraction = self.config.fresh_fraction
        fresh = math.floor(self.slot_count * fresh_fraction) > math.floor(
            (self.slot_count - 1) * fresh_fraction
        )
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        if fresh:
            loss, group_ages = apply_fresh_update(
                self.student, self.optimizer, group, self.config
            )
            self.fresh_losses.append(loss)
            self.ages.extend(group_ages)
        else:
            self.replay_losses.append(
                train_student(
                    self.student, self.optimizer, self.buffer,
                    self.replay_config,
                )["loss"]
            )
            self.student.weight_version += 1
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds.append(time.perf_counter() - start)

    def pop_stats(self) -> dict:
        """
        Statistics of the updates since the last call, then reset them.

        Returns:
            dict: loss at collection, replay loss, mean gradient age, update
                counts and the seconds spent updating.
        """
        stats = {
            "loss": float(torch.stack(self.fresh_losses).mean())
            if self.fresh_losses else math.nan,
            "replay_loss": float(np.mean(self.replay_losses))
            if self.replay_losses else math.nan,
            "grad_age_updates": float(np.mean(self.ages))
            if self.ages else math.nan,
            "fresh_updates": len(self.fresh_losses),
            "replay_updates": len(self.replay_losses),
            "update_seconds": sum(self.seconds),
        }
        self.fresh_losses, self.replay_losses = [], []
        self.ages, self.seconds = [], []
        return stats


def distill(
        env: Drive,
        teacher: Teacher,
        student: Student,
        buffer: LabelBuffer,
        config: DistillConfig,
        evaluator: DistillEvaluator | None = None,
    ) -> DistillResult:
    """
    The BC/DAgger loop: collect labels, train, log, evaluate.

    Args:
        env (Drive): continuous-action env built by make_env.
        teacher (Teacher): labeler, on config.device.
        student (Student): model being trained, on config.device.
        buffer (LabelBuffer): label storage, on config.device.
        config (DistillConfig): run settings.
        evaluator (DistillEvaluator | None): closed-loop evaluation every
            eval_interval_labels; None skips it.

    Returns:
        DistillResult: labels seen, optimizer and log rows.
    """
    device = config.device
    optimizer = make_optimizer(student, config)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    agent_count = env.num_agents
    obs, _ = env.reset(seed=config.seed)
    labels_seen = 0
    if config.resume_checkpoint:
        labels_seen = load_checkpoint(
            config.resume_checkpoint, student, optimizer
        )
    initial_draws = torch.rand(agent_count, device=device, generator=generator)
    # The first nuplan_worker_count workers' rows (train.py, VecDrive)
    nuplan_row_count = (
        agent_count // config.env_worker_count * config.nuplan_worker_count
    )
    episodes = EpisodeState(
        teacher_drives=initial_draws < beta_at(config, labels_seen),
        live=torch.ones(agent_count, dtype=torch.bool, device=device),
        step_count=torch.zeros(agent_count, dtype=torch.int64, device=device),
        nuplan_rows=(
            torch.arange(agent_count, device=device) < nuplan_row_count
        ),
    )
    student.reset_plans(episodes.live.clone())
    if config.fresh_gradient and not isinstance(student, MPCStudent):
        raise ValueError("fresh_gradient needs an MPCStudent")
    # Replay samples carry no previous plan to shift into a tube reference
    if student.needs_agent_poses and not config.fresh_gradient:
        raise ValueError(
            "the tube needs fresh_gradient=true"
        )
    learner = None
    if config.fresh_gradient:
        learner = FreshLearner(student, optimizer, buffer, config)
        # A solve that fails even in float64 leaves its inputs here
        if config.output_dir:
            student.failure_dump_dir = config.output_dir
    probe = None
    if config.probe_set_size > 0:
        probe = ForgettingProbe(config, labels_seen)
    log_rows = []
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
    evaluating = evaluator is not None and config.eval_interval_labels > 0
    # A resumed checkpoint was evaluated by the run that saved it
    if evaluating and not config.resume_checkpoint:
        run_evaluation(evaluator, student, optimizer, labels_seen, config)
    next_eval_labels = (
        labels_seen // config.eval_interval_labels + 1
    ) * config.eval_interval_labels if evaluating else 0

    stop_path = os.path.join(config.output_dir, STOP_FILE_NAME)
    for iteration in range(config.iterations):
        if config.max_labels and labels_seen >= config.max_labels:
            break
        if config.output_dir and os.path.exists(stop_path):
            print(f"stop file found at {labels_seen:,} labels")
            break
        collect_start = time.perf_counter()
        obs, labels_seen, env_log = collect(
            env,
            obs,
            teacher,
            student,
            buffer,
            episodes,
            config,
            labels_seen,
            generator,
            learner.add if learner is not None else None,
        )
        train_start = time.perf_counter()
        if learner is not None:
            # The updates ran inside collect(); report their time apart
            train_stats = learner.pop_stats()
            train_seconds = train_stats.pop("update_seconds")
            collect_seconds = train_start - collect_start - train_seconds
        else:
            train_stats = train_student(student, optimizer, buffer, config)
            train_seconds = time.perf_counter() - train_start
            collect_seconds = train_start - collect_start
        probe_losses = {}
        if probe is not None:
            probe.freeze_due_sets(buffer, labels_seen)
            probe_losses = probe.evaluate(student)

        row = {
            "iteration": iteration,
            "labels_seen": labels_seen,
            "beta": beta_at(config, labels_seen),
            "teacher_driven_fraction": float(
                episodes.teacher_drives.float().mean()
            ),
            "collect_seconds": collect_seconds,
            "train_seconds": train_seconds,
            **train_stats,
            **student.pop_tube_stats(),
            **student.pop_solver_stats(),
            **probe_losses,
            **pop_episode_stats(episodes),
            **{f"env/{key}": float(value) for key, value in env_log.items()},
        }
        log_rows.append(row)
        write_log_row(row, config)
        if evaluating and labels_seen >= next_eval_labels:
            run_evaluation(evaluator, student, optimizer, labels_seen, config)
            # One evaluation per crossing, even if an iteration skips several
            while next_eval_labels <= labels_seen:
                next_eval_labels += config.eval_interval_labels

    if config.output_dir:
        save_checkpoint(
            os.path.join(config.output_dir, CHECKPOINT_NAME),
            student,
            optimizer,
            labels_seen,
            config,
        )
    return DistillResult(labels_seen, optimizer, log_rows)


def run_evaluation(
        evaluator: DistillEvaluator,
        student: Student,
        optimizer: torch.optim.Optimizer,
        labels_seen: int,
        config: DistillConfig,
    ) -> dict:
    """
    Evaluate the student closed-loop, keep a checkpoint of it, and append
    the metrics to the run's evaluation log.

    Args:
        evaluator (DistillEvaluator): closed-loop evaluator.
        student (Student): model being trained.
        optimizer (torch.optim.Optimizer): its optimizer (checkpointed).
        labels_seen (int): labels added so far.
        config (DistillConfig): run settings (output_dir).

    Returns:
        dict: the evaluator's metrics.
    """
    eval_start = time.perf_counter()
    metrics = evaluator.evaluate(student)
    row = {
        "labels_seen": labels_seen,
        "seconds": time.perf_counter() - eval_start,
        **metrics,
    }
    print(
        f"eval at {labels_seen:,} labels: "
        f"goals {metrics.get('num_goals_reached', math.nan):.3f}  "
        f"collision {metrics.get('collision_rate', math.nan):.4f}  "
        f"offroad {metrics.get('offroad_rate', math.nan):.4f}"
    )
    if not config.output_dir:
        return metrics
    checkpoint_name = EVAL_CHECKPOINT_TEMPLATE.format(labels_seen=labels_seen)
    save_checkpoint(
        os.path.join(config.output_dir, checkpoint_name),
        student,
        optimizer,
        labels_seen,
        config,
    )
    with open(os.path.join(config.output_dir, EVAL_LOG_NAME), "a") as log:
        log.write(json.dumps(row) + "\n")
    return metrics


def write_log_row(
        row: dict,
        config: DistillConfig,
    ) -> None:
    """
    Print a one-line summary and append the row to the run's log file.

    Args:
        row (dict): metrics of one iteration.
        config (DistillConfig): run settings (output_dir).
    """
    print(
        f"iter {row['iteration']:>5}  labels {row['labels_seen']:>11,}  "
        f"beta {row['beta']:.2f}  loss {row['loss']:.4f}  "
        f"collect {row['collect_seconds']:.2f}s  "
        f"train {row['train_seconds']:.2f}s"
    )
    if not config.output_dir:
        return
    with open(os.path.join(config.output_dir, LOG_NAME), "a") as log_file:
        log_file.write(json.dumps(row) + "\n")
