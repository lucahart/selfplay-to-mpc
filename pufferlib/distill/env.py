"""Builds the PufferDrive env the teacher was trained on, for distillation.

Labels must come from the teacher's state distribution, so the env uses
puffer_drive.yaml plus the teacher's training overrides (resets and
termination depend on them).
"""

import os
from collections import defaultdict

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CONFIG_DIR = os.path.join(REPO_ROOT, "pufferlib", "config")
CONFIG_NAME = "puffer_drive"
# The paper's teachers (docs/setup.md): a 4 x 1024 (large) and a 4 x 512
# (small) self-play policy, trained on CARLA and fine-tuned on CARLA and
# nuPlan maps
TEACHER_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints/teacher_4x1024.pt")
SMALL_TEACHER_CHECKPOINT = os.path.join(
    REPO_ROOT, "checkpoints/teacher_4x512.pt"
)
# The network TEACHER_CHECKPOINT was trained with; students copy the
# teacher's backbone, so they get it too unless their sizes are set
TEACHER_POLICY_OVERRIDES = (
    "policy.ego_input_size=128",
    "policy.partner_input_size=128",
    "policy.lane_input_size=128",
    "policy.boundary_input_size=128",
    "policy.traffic_control_input_size=128",
    "policy.context_input_size=128",
    "policy.backbone_hidden_size=1024",
    "policy.backbone_num_layers=4",
    "policy.backbone_layer_norm=true",
    "policy.shared_network=false",
)
# The environment the teachers were trained in
TEACHER_ENV_OVERRIDES = (
    "env.dynamics_model=classic",
    "env.termination_mode=1",
    "env.reward_reverse=0.05",
    "env.scenario_length=1280",
    "env.reward_comfort=auto",
    "env.reward_collision=4",
    "env.traffic_control_obs_facing_only=true",
    "env.traffic_light_obs_unknown_as_off=true",
)
# nuPlan replay self-play of the teachers' nuPlan fine-tune, added after
# TEACHER_ENV_OVERRIDES, which already hold its light flags; the CARLA
# workers keep TEACHER_ENV_OVERRIDES alone
NUPLAN_TRAIN_SCENE_COUNT = 3637
# Steps of a nuPlan episode: the log's 201 frames from init_step 0
NUPLAN_SCENARIO_STEPS = 200
NUPLAN_ENV_OVERRIDES = (
    "env.simulation_mode=replay",
    "env.control_mode=control_vehicles",
    "env.init_mode=create_all_valid",
    "env.map_dir=data/nuplan_mini_train",
    f"env.num_maps={NUPLAN_TRAIN_SCENE_COUNT}",
    "env.max_agents_per_env=300",
    f"env.scenario_length={NUPLAN_SCENARIO_STEPS}",
    "env.init_step=0",
    "env.require_episode_within_log=true",
    "env.resample_frequency=1000",
    "env.use_neighbor_cache=0",
    "env.replay_replan_goals=1",
)
ACTION_TYPES = ("discrete", "continuous")


def load_distill_args(
        extra_overrides: tuple[str, ...] = (),
    ) -> dict:
    """
    Compose puffer_drive.yaml with the teacher's env and policy
    overrides, like pufferl.load_config but without reading sys.argv.

    Args:
        extra_overrides (tuple[str, ...]): further Hydra overrides,
            e.g. ("env.resample_frequency=20",).

    Returns:
        dict: plain nested config (env, policy, train, vec, eval, ...).
    """
    overrides = (
        list(TEACHER_ENV_OVERRIDES)
        + list(TEACHER_POLICY_OVERRIDES)
        + list(extra_overrides)
    )
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    args = defaultdict(dict, OmegaConf.to_container(cfg, resolve=True))
    args["train"]["use_rnn"] = args["rnn_name"] is not None
    # Relative map paths in the config assume the repo root as cwd
    args["env"]["map_dir"] = os.path.join(REPO_ROOT, args["env"]["map_dir"])
    return args


def make_env(
        action_type: str,
        seed: int,
        agent_count: int,
        extra_overrides: tuple[str, ...] = (),
    ) -> tuple[Drive, dict]:
    """
    Build a Drive env with the teacher's settings.

    Args:
        action_type (str): "discrete" (teacher's own) or "continuous".
        seed (int): env seed.
        agent_count (int): total agents across the env's maps.
        extra_overrides (tuple[str, ...]): further Hydra overrides.

    Returns:
        tuple[Drive, dict]: the env and the composed config.
    """
    if action_type not in ACTION_TYPES:
        raise ValueError(f"action_type must be one of {ACTION_TYPES}")
    if agent_count <= 0:
        raise ValueError(f"agent_count must be positive, got {agent_count}")
    args = load_distill_args(extra_overrides)
    env_kwargs = dict(
        args["env"],
        num_agents=agent_count,
        action_type=action_type,
        seed=seed,
    )
    # The torch model covers the classic step only, without phantom braking
    if env_kwargs["dynamics_model"] != "classic":
        raise ValueError("distillation needs dynamics_model=classic")
    if env_kwargs["phantom_braking_trigger_prob"] != 0.0:
        raise ValueError("phantom braking is not modeled; keep it off")
    return Drive(**env_kwargs), args
