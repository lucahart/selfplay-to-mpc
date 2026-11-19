"""Students distilled from the teacher: the MLP baseline and the DiffMPC
student. Both use the teacher's backbone, with its weights or from scratch,
and differ only in the head."""

import copy
import dataclasses
import importlib.util
import math
import os

import torch

import pufferlib.pytorch
from pufferlib.distill.bicycle import (
    EGO_LENGTH_IDX,
    NU,
    NX,
    OBS_NORM_VEH_LENGTH_M,
    PY,
    SPEED,
    STEERING_LIMIT_RAD,
    U_HI,
    Bicycle,
    ego_state_from_obs,
    solve,
    target_cost,
)
from pufferlib.distill.constraints import (
    CONSTRAINT_NONE,
    CONSTRAINT_TUBE,
    PLANE_PARAM_COUNT,
    TUBE_PLANE_COUNT,
    TUBE_PLANE_VALUE_COUNT,
    validate_constraints,
)
from pufferlib.distill.obstacles import (
    DUMMY_DISTANCE_M,
    EGO_WIDTH_IDX,
    OBS_NORM_VEH_WIDTH_M,
    OBSTACLE_PARAM_COUNT,
    partner_obstacles,
)
from pufferlib.distill.overlay import (
    OVERLAY_FLOATS,
    pack_overlay,
    planned_positions,
)
from pufferlib.distill.teacher import Teacher
from pufferlib.distill.tube import (
    DEFAULT_TUBE_MARGIN_M,
    POSE_DIM,
    TUBE_ANCHOR_PLAN,
    TUBE_ANCHOR_ROUTE,
    TUBE_ANCHORS,
    TUBE_NORMAL_PROJECTION,
    TUBE_SECTOR_FULL,
    count_fallback_planes,
    planned_states,
    road_edge_segments,
    route_anchors,
    route_planes,
    shifted_reference,
    straight_reference,
    tube_planes,
)

HEAD_INIT_STD = 0.01
PD_ACCEL_BOUND_MPS2 = float(U_HI[0])
PD_STEERING_RATE_BOUND_RADPS = float(U_HI[1])

# MPC cost: per step, weights for (y, heading, speed, steering, accel,
# rate) and targets for (y, heading, speed change, steering)
STATE_WEIGHT_COUNT = 4
CONTROL_WEIGHT_COUNT = 2
TARGET_COUNT = 4
COST_OUTPUTS_PER_STEP = STATE_WEIGHT_COUNT + CONTROL_WEIGHT_COUNT + (
    TARGET_COUNT
)
STATE_WEIGHT_MAX = 10.0
CONTROL_WEIGHT_MIN = 0.1
CONTROL_WEIGHT_MAX = 10.0
TARGET_LATERAL_RANGE_M = 5.0
# MPCStudent's per-agent rows: warm starts, the previous-plan tube
# reference, and the poses and route windows set for the coming act
PLAN_STATE_ATTRIBUTES = (
    "plans",
    "reference_states",
    "reference_poses",
    "reference_valid",
    "agent_poses",
    "route_window",
)
TARGET_HEADING_RANGE_RAD = 0.5
TARGET_SPEED_CHANGE_RANGE_MPS = 10.0
# Half-ranges of the state targets (lateral, heading, speed change,
# steering) that scale the fixed-sum weights (sum_mode_weights)
STATE_TERM_RANGES = (
    TARGET_LATERAL_RANGE_M,
    TARGET_HEADING_RANGE_RAD,
    TARGET_SPEED_CHANGE_RANGE_MPS,
    STEERING_LIMIT_RAD,
)
# Weight modes of the learned cost (MPCStudentConfig.weight_mode): each
# weight its own sigmoid (the original), or a fixed sum per node (design
# doc section 2.3: a softmax over the node's active terms, normalized by
# each term's range squared, with a floor on the control weights)
WEIGHT_MODE_SIGMOID = "sigmoid"
WEIGHT_MODE_SUM = "sum"
# Slack prices: "fixed" (warpmpc_solver's constants times
# slack_weight_scale) or "sum" (linear price = weight_sum on every node,
# quadratic = slack_quadratic_fraction * weight_sum). Linear only by
# default: curvature changed the converged u_0 by < 1% of the box
SLACK_PRICE_FIXED = "fixed"
SLACK_PRICE_SUM = "sum"
SLACK_PRICE_MODES = (SLACK_PRICE_FIXED, SLACK_PRICE_SUM)
DEFAULT_SLACK_QUADRATIC_FRACTION = 0.0
# Node weights kappa_k (weight_mode=sum, slack_price_mode=sum): "fixed"
# (1 on every node) or "learned" (one more head output z per node,
# kappa_k = 1 + (1 - kappa_min) tanh(z_k - mean z): a common shift of z
# changes nothing, and kappa stays in [kappa_min, 2 - kappa_min]; node k's
# weights and slack price are S * kappa_k). The bound keeps every node's
# scale where the ADMM converges: at S = 40, scales 10 to 100 gave a
# 25-ADMM u_0 p99 of 6-9% of the box, 310 gave 75%
NODE_WEIGHT_FIXED = "fixed"
NODE_WEIGHT_LEARNED = "learned"
NODE_WEIGHT_MODES = (NODE_WEIGHT_FIXED, NODE_WEIGHT_LEARNED)
DEFAULT_NODE_WEIGHT_MIN = 0.25
WEIGHT_MODES = (WEIGHT_MODE_SIGMOID, WEIGHT_MODE_SUM)
# Fixed-sum mode, per node: sum_j r_j^2 q_j over the active terms (S) and
# the floor on each control's share (rho)
DEFAULT_WEIGHT_SUM = 1.0
DEFAULT_CONTROL_SHARE_FLOOR = 0.01
# Inputs of a solve whose gradient stayed non-finite in float64
FAILING_SOLVE_NAME = "failing_solve.pt"
# A planned node violates a plane by more than this [m] or an ellipse by
# more than this (normalized distance below 1) in the violation stats
PLANE_VIOLATION_TOL_M = 0.01
ELLIPSE_VIOLATION_TOL = 0.01
# A rate counts as over (or at) PD's limit beyond this [rad/s]
RATE_LIMIT_TOL_RADPS = 1e-4

# Per-object encoders of DriveBackbone (torch.py), as <name>_encoder
ENCODER_NAMES = (
    "ego", "lane", "boundary", "partner", "traffic_control", "context",
)
# Size fields of MLPStudentConfig: 0 keeps the teacher's size
TEACHER_SIZE = 0
# Key prefix of the actor's encoders and backbone in a PPO policy state dict
POLICY_BACKBONE_PREFIX = "actor_backbone."
# A residual block holds two linear layers
LAYERS_PER_RESIDUAL_BLOCK = 2

# MPC solvers: mpc.pytorch's iLQR, or WarpMPC's batched SQP (needs JAX,
# Warp and CasADi, see docs/setup.md)
SOLVER_MPC_PYTORCH = "mpc_pytorch"
SOLVER_WARPMPC = "warpmpc"
SOLVERS = (SOLVER_MPC_PYTORCH, SOLVER_WARPMPC)
# Constraint types of the MPC student: none, or the tube
STUDENT_CONSTRAINT_TYPES = (CONSTRAINT_NONE, CONSTRAINT_TUBE)
# WarpMPC's stage layout needs a first and a terminal node
WARPMPC_MIN_HORIZON_STEPS = 2


def to_cpu(
        values: torch.Tensor | tuple | None,
    ) -> torch.Tensor | tuple | None:
    """
    Detached CPU copy of a tensor or a tuple of tensors, for dumps.

    Args:
        values (torch.Tensor | tuple | None): tensor, tuple or None.

    Returns:
        torch.Tensor | tuple | None: the same structure on the CPU.
    """
    if values is None:
        return None
    if isinstance(values, tuple):
        return tuple(value.detach().cpu() for value in values)
    return values.detach().cpu()


def vehicle_rows(
        obs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Rows that hold a vehicle. Evaluators pad unused agent slots with
    all-zero observations, whose vehicle length (and wheelbase) is 0.

    Args:
        obs (torch.Tensor): (B,obs_dim) observations.

    Returns:
        torch.Tensor: (B,) bool.
    """
    return obs[:, EGO_LENGTH_IDX] > 0


def sum_mode_weights(
        state_logits: torch.Tensor,
        control_logits: torch.Tensor,
        state_ranges: torch.Tensor,
        control_ranges: torch.Tensor,
        weight_sum: float,
        control_share_floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed-sum cost weights per node (design doc section 2.3). On node k
    the active terms A_k are the controls at k = 0, all six terms in the
    middle and the four states at k = T-1; their shares are
    pi_kj = (1 - n_k rho) softmax_{A_k}(z_k)_j + rho [j is a control],
    with n_k the number of controls in A_k, and q_kj = S pi_kj / r_j^2, so
    sum_{j in A_k} r_j^2 q_kj = S. Node-0 state weights are 0; the
    terminal controls get the floor, S rho / r_j^2.

    Args:
        state_logits (torch.Tensor): (T,B,4) logits of (p_y, psi, v,
            delta).
        control_logits (torch.Tensor): (T,B,2) logits of (accel, rate).
        state_ranges (torch.Tensor): (4,) r_j of the states.
        control_ranges (torch.Tensor): (2,) r_j of the controls.
        weight_sum (float): S.
        control_share_floor (float): rho.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (T,B,4) state and (T,B,2)
            control weights.
    """
    horizon_steps = state_logits.shape[0]
    state_count = state_logits.shape[-1]
    logits = torch.cat([state_logits, control_logits], -1)
    term_count = logits.shape[-1]
    is_control = torch.arange(term_count, device=logits.device) >= state_count
    node = torch.arange(horizon_steps, device=logits.device)[:, None]
    terminal = node == horizon_steps - 1
    # (T,6): controls only at k = 0, states only at T-1, all in between
    active = torch.where(
        node == 0, is_control, torch.where(terminal, ~is_control, True)
    )
    shares = torch.softmax(
        logits.masked_fill(~active[:, None, :], -torch.inf), -1
    )
    control_active = (active & is_control).to(logits.dtype)
    control_count = control_active.sum(-1)
    shares = (
        (1 - control_count * control_share_floor)[:, None, None] * shares
        + control_share_floor * control_active[:, None, :]
    )
    # The terminal controls are outside the sum and keep the floor
    terminal_controls = (terminal & is_control).to(logits.dtype)
    shares = shares + control_share_floor * terminal_controls[:, None, :]
    ranges = torch.cat([state_ranges, control_ranges])
    weights = weight_sum * shares / ranges.square()
    return weights[..., :state_count], weights[..., state_count:]


class Student(torch.nn.Module):
    """Interface the distillation loop and ContinuousPolicy rely on.
    Subclasses register a (2,) `control_bound` buffer: the largest |u| they
    can output, used for saturation metrics."""

    control_bound: torch.Tensor

    def predict(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Differentiable control prediction, used for the loss.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate).
        """
        raise NotImplementedError

    def act(
            self,
            obs: torch.Tensor,
            driven_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        Control used to drive, without gradients. Stateful students (MPC
        warm starts) keep one plan per agent row.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of all agents.
            driven_mask (torch.Tensor): (B,) bool, agents the student
                drives; other rows of the result are unused.

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate).
        """
        raise NotImplementedError

    @property
    def needs_agent_poses(self) -> bool:
        """
        Returns:
            bool: act needs set_agent_poses first; only an MPC student
                with a previous-plan tube does.
        """
        return False

    @property
    def needs_route_window(self) -> bool:
        """
        Returns:
            bool: act needs set_route_window first; only an MPC student
                with a route-anchored tube does.
        """
        return False

    def pop_tube_stats(self) -> dict:
        """
        Tube statistics since the last call, then reset; none without a
        tube.

        Returns:
            dict: metric name -> value.
        """
        return {}

    def pop_solver_stats(self) -> dict:
        """
        Solver statistics since the last call, then reset; none without a
        solver.

        Returns:
            dict: metric name -> value.
        """
        return {}

    def reset_plans(
            self,
            new_episode_mask: torch.Tensor,
        ) -> None:
        """
        Forget per-agent state (warm starts) of agents that start a new
        episode. Stateless students do nothing.

        Args:
            new_episode_mask (torch.Tensor): (B,) bool.
        """

    def take_plan_state(self) -> dict:
        """
        Hand over the per-agent state (warm starts, tube references) and
        start fresh, so an env with another row count (an evaluation) can
        run in between; restore_plan_state puts it back. Stateless
        students have none.

        Returns:
            dict: attribute name -> value.
        """
        return {}

    def restore_plan_state(
            self,
            plan_state: dict,
        ) -> None:
        """
        Put back the per-agent state from take_plan_state.

        Args:
            plan_state (dict): take_plan_state's result.
        """


def validate_bounds(
        accel_bound_mps2: float,
        steering_rate_bound_radps: float,
    ) -> None:
    """
    A student's control box must contain PD's box, or labels on PD's edge
    would be unreachable.

    Args:
        accel_bound_mps2 (float): student's |accel| bound [m/s^2].
        steering_rate_bound_radps (float): student's |rate| bound [rad/s].
    """
    if accel_bound_mps2 < PD_ACCEL_BOUND_MPS2:
        raise ValueError(f"accel bound must be >= {PD_ACCEL_BOUND_MPS2}")
    if steering_rate_bound_radps < PD_STEERING_RATE_BOUND_RADPS:
        raise ValueError(
            f"steering rate bound must be >= {PD_STEERING_RATE_BOUND_RADPS}"
        )


class ResidualBlock(torch.nn.Module):
    """Pre-LayerNorm residual MLP block (SimBa): x + W2 GELU(W1 LN(x))."""

    def __init__(
            self,
            width: int,
        ) -> None:
        """
        Two width x width linear layers behind a LayerNorm.

        Args:
            width (int): feature width in and out.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.inner = torch.nn.Linear(width, width)
        self.outer = torch.nn.Linear(width, width)
        self.activation = torch.nn.GELU()

    def forward(
            self,
            features: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (B,width) input.

        Returns:
            torch.Tensor: (B,width) input plus the block's update.
        """
        update = self.outer(self.activation(self.inner(self.norm(features))))
        return features + update


def resize_backbone(
        backbone: torch.nn.Module,
        encoder_size: int,
        hidden_size: int,
        num_layers: int,
        residual: bool,
        keep_layer_norm: bool,
    ) -> None:
    """
    Replace a DriveBackbone's encoders and main MLP, in place, by fresh
    ones of the given sizes; the observation layout stays the teacher's.
    Plain: num_layers x (GELU, Linear[, LayerNorm if the teacher has it
    and keep_layer_norm]),
    then GELU, as DriveBackbone builds it. Residual: an input Linear, num_layers / 2 ResidualBlocks, then
    LayerNorm and GELU.

    Args:
        backbone (torch.nn.Module): deep copy of the teacher's backbone.
        encoder_size (int): width of every per-object encoder.
        hidden_size (int): width of the main MLP.
        num_layers (int): linear layers of the main MLP (residual: in
            the blocks, so even).
        residual (bool): residual blocks instead of plain layers.
        keep_layer_norm (bool): plain only: keep the teacher's LayerNorm
            after each Linear; False builds the 4x512 teacher's layout.
    """
    encoders_out = 0
    for name in ENCODER_NAMES:
        encoder = getattr(backbone, f"{name}_encoder", None)
        if encoder is None:
            continue
        in_features = encoder[0].in_features
        setattr(
            backbone,
            f"{name}_encoder",
            backbone._create_encoder(in_features, encoder_size),
        )
        # forward pools with <name>_input_size as the feature width
        setattr(backbone, f"{name}_input_size", encoder_size)
        encoders_out += encoder_size
    activation_class = type(backbone.backbone[0])
    # A teacher built with backbone_layer_norm puts LayerNorm after each
    # Linear; plain students keep that unless keep_layer_norm is False
    layer_norm = keep_layer_norm and any(
        isinstance(module, torch.nn.LayerNorm) for module in backbone.backbone
    )
    if residual:
        if num_layers % LAYERS_PER_RESIDUAL_BLOCK != 0:
            raise ValueError("residual backbones need an even num_layers")
        block_count = num_layers // LAYERS_PER_RESIDUAL_BLOCK
        layers = [torch.nn.Linear(encoders_out, hidden_size)]
        layers += [ResidualBlock(hidden_size) for _ in range(block_count)]
        layers += [torch.nn.LayerNorm(hidden_size), activation_class()]
    else:
        layers = []
        in_size = encoders_out
        for _ in range(num_layers):
            layers += [
                activation_class(),
                torch.nn.Linear(in_size, hidden_size),
            ]
            if layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))
            in_size = hidden_size
        layers.append(activation_class())
    backbone.backbone = torch.nn.Sequential(*layers)
    backbone.out_dim = hidden_size


def student_backbone(
        teacher: Teacher,
        init_from_teacher: bool,
        sizes: tuple[int, int, int, bool, bool] | None = None,
        backbone_checkpoint: str = "",
    ) -> torch.nn.Module:
    """
    The teacher's encoders and backbone as the student's trainable
    network: with the teacher's weights, or initialized as PPO initializes
    the teacher (orthogonal weights, zero biases, unit LayerNorm),
    optionally resized, optionally then loaded from another policy.

    Args:
        teacher (Teacher): source of the architecture and weights.
        init_from_teacher (bool): keep the teacher's weights.
        sizes (tuple | None): (encoder_size, hidden_size, num_layers,
            residual, keep_layer_norm) for resize_backbone, fresh weights
            only; None keeps the teacher's architecture.
        backbone_checkpoint (str): fresh weights only: a PPO policy
            checkpoint whose actor_backbone weights replace the fresh
            ones (strict: its architecture must match the sized
            backbone); "" keeps the fresh weights.

    Returns:
        torch.nn.Module: the backbone (DriveBackbone), trainable.
    """
    backbone = copy.deepcopy(teacher.policy.actor_backbone)
    backbone.requires_grad_(True)
    if init_from_teacher:
        return backbone
    if sizes is not None:
        resize_backbone(backbone, *sizes)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Linear):
            pufferlib.pytorch.layer_init(module)
        elif isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()
    if backbone_checkpoint:
        policy_state = torch.load(
            backbone_checkpoint, map_location="cpu", weights_only=False
        )
        backbone_state = {
            name[len(POLICY_BACKBONE_PREFIX):]: value
            for name, value in policy_state.items()
            if name.startswith(POLICY_BACKBONE_PREFIX)
        }
        backbone.load_state_dict(backbone_state)
    return backbone


@dataclasses.dataclass
class BackboneConfig:
    """Initialization and size of a student's backbone, the same keys for
    the MLP and the MPC student."""

    # False: same architecture, fresh weights (no teacher features)
    init_from_teacher: bool = True
    # Backbone size with fresh weights; TEACHER_SIZE keeps the teacher's
    encoder_size: int = TEACHER_SIZE
    hidden_size: int = TEACHER_SIZE
    num_layers: int = TEACHER_SIZE
    residual: bool = False
    # False drops the LayerNorm a backbone_layer_norm teacher puts after
    # each main-MLP Linear (plain backbones, fresh weights only)
    keep_layer_norm: bool = True
    # PPO policy checkpoint whose actor backbone initializes the sized
    # backbone instead of fresh weights (e.g. SMALL_TEACHER_CHECKPOINT for
    # the 4x512 layout); "" for fresh weights
    backbone_checkpoint: str = ""

    def validate_backbone(self) -> None:
        """Reject negative sizes, and a resized or loaded backbone with the
        teacher's weights."""
        size_values = (self.encoder_size, self.hidden_size, self.num_layers)
        if min(size_values) < 0:
            raise ValueError("backbone sizes must be >= 0")
        resized = (
            self.residual
            or not self.keep_layer_norm
            or any(size != TEACHER_SIZE for size in size_values)
        )
        if resized and self.init_from_teacher:
            raise ValueError("a resized backbone needs init_from_teacher=false")
        if self.backbone_checkpoint and self.init_from_teacher:
            raise ValueError(
                "backbone_checkpoint needs init_from_teacher=false"
            )

    def backbone_sizes(
            self,
            teacher: Teacher,
        ) -> tuple[int, int, int, bool, bool] | None:
        """
        Sizes for student_backbone, the teacher's where left at
        TEACHER_SIZE.

        Args:
            teacher (Teacher): source of the default sizes.

        Returns:
            tuple | None: (encoder_size, hidden_size, num_layers,
                residual, keep_layer_norm), or None for the teacher's
                architecture.
        """
        if self.init_from_teacher:
            return None
        teacher_backbone = teacher.policy.actor_backbone
        teacher_layers = sum(
            isinstance(module, torch.nn.Linear)
            for module in teacher_backbone.backbone
        )
        return (
            self.encoder_size or teacher_backbone.ego_input_size,
            self.hidden_size or teacher_backbone.out_dim,
            self.num_layers or teacher_layers,
            self.residual,
            self.keep_layer_norm,
        )


@dataclasses.dataclass
class MLPStudentConfig(BackboneConfig):
    """Output range of the MLP head (tanh scaled to this box), the
    backbone (BackboneConfig) and the plan length of the head."""

    accel_bound_mps2: float = PD_ACCEL_BOUND_MPS2
    steering_rate_bound_radps: float = PD_STEERING_RATE_BOUND_RADPS
    # MLP-plan: the head outputs plan_steps controls and the first is
    # executed, the MPC student's output shape without its solver
    plan_steps: int = 1

    def __post_init__(self) -> None:
        """
        Reject a box narrower than PD's, invalid backbone sizes and a
        plan shorter than one step.
        """
        validate_bounds(self.accel_bound_mps2, self.steering_rate_bound_radps)
        self.validate_backbone()
        if self.plan_steps < 1:
            raise ValueError("plan_steps must be >= 1")


class MLPStudent(Student):
    """Teacher's backbone (its weights or fresh ones), trained, with a
    linear head to (a, omega) squashed into the control box."""

    def __init__(
            self,
            teacher: Teacher,
            config: MLPStudentConfig | None = None,
        ) -> None:
        """
        Copy the teacher's encoders and backbone; new linear head.

        Args:
            teacher (Teacher): source of the backbone architecture and,
                with config.init_from_teacher, its weights.
            config (MLPStudentConfig | None): output range and
                initialization; None is PD's box from the teacher.
        """
        super().__init__()
        config = config or MLPStudentConfig()
        self.backbone = student_backbone(
            teacher,
            config.init_from_teacher,
            config.backbone_sizes(teacher),
            config.backbone_checkpoint,
        )
        self.ego_dim = teacher.policy.ego_dim
        self.head = torch.nn.Linear(
            self.backbone.out_dim, NU * config.plan_steps
        )
        torch.nn.init.normal_(self.head.weight, std=HEAD_INIT_STD)
        torch.nn.init.zeros_(self.head.bias)
        # Buffer name kept so earlier checkpoints still load
        self.register_buffer(
            "control_scale",
            torch.tensor(
                [config.accel_bound_mps2, config.steering_rate_bound_radps]
            ),
        )

    @property
    def control_bound(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: (2,) the tanh output scale.
        """
        return self.control_scale

    def predict(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Backbone, linear head, tanh scaled to the control box; with
        plan_steps > 1 the first control of the plan.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate).
        """
        hidden = self.backbone(obs.float(), self.ego_dim)
        # The plan is (u_0, u_1, ...) row-major; labels exist for u_0 only
        first_control = self.head(hidden)[:, :NU]
        return torch.tanh(first_control) * self.control_scale

    def act(
            self,
            obs: torch.Tensor,
            driven_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        Same as predict, without gradients; stateless.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of all agents.
            driven_mask (torch.Tensor): (B,) bool; unused (cheap to run
                on all rows).

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate).
        """
        with torch.no_grad():
            return self.predict(obs)


@dataclasses.dataclass
class MPCStudentConfig(BackboneConfig):
    """Horizon, solver iterations, control box, constraints and backbone
    (BackboneConfig) of the MPC student."""

    horizon_steps: int = 3
    train_lqr_iterations: int = 3
    act_lqr_iterations: int = 3
    act_warm_start: bool = True
    # Wider than PD's box (4, 0.6) so predictions near PD's limits stay
    # interior and keep their gradient through the solver
    accel_bound_mps2: float = PD_ACCEL_BOUND_MPS2
    steering_rate_bound_radps: float = 0.75
    solver: str = SOLVER_MPC_PYTORCH
    # WarpMPC only, for training and acting alike; the *_lqr_iterations
    # are mpc.pytorch's
    sqp_iterations: int = 3
    admm_iterations: int = 25
    # WarpMPC only: constraint rows with slack on nodes k >= 1
    # (constraints.py): "tube" (fixed left and right planes on the
    # observed road edges, tube.py; its count is K constant-velocity
    # partner ellipses per node, 0 for none); "none" with 0 solves the
    # plain problem
    constraint_type: str = CONSTRAINT_NONE
    constraint_count: int = 0
    # Tube only: distance the body keeps inside the road edge [m]
    tube_margin_m: float = DEFAULT_TUBE_MARGIN_M
    # Tube only: where the planes are anchored (tube.py): "plan" (at the
    # reference positions, the agent's last plan shifted by its real pose
    # change) or "route" (on the agent's route, parallel to it; needs the
    # route windows before every act; nodes without a route anchor keep
    # the plan's planes)
    tube_anchor: str = TUBE_ANCHOR_PLAN
    # Route anchor only: an edge counts for a side only if it lies on that
    # side of the node's reference position as well as of the anchor
    # (tube.route_planes): drops island edges the plan has passed, which
    # would give planes violated from the start
    tube_edge_side_filter: bool = False
    # Route anchor only: edge segments count for a plane when they run
    # within this angle of the route tangent (90 keeps all), and a side
    # without a closer edge gets its plane this far from the anchor
    # (tube.route_planes)
    tube_edge_parallel_max_deg: float = 35.0
    tube_plane_max_offset_m: float = 8.0
    # Route anchor only: edge points count within this distance along the
    # route tangent from the anchor; 0 removes the limit
    tube_edge_along_window_m: float = 2.0
    # WarpMPC constraints: factor on every slack price (planes and
    # ellipses, linear and quadratic; warpmpc_solver.py)
    slack_weight_scale: float = 1.0
    # "sigmoid" (each weight its own sigmoid) or "sum" (fixed sum per
    # node, sum_mode_weights; weight_sum S and control_share_floor rho)
    weight_mode: str = WEIGHT_MODE_SIGMOID
    weight_sum: float = DEFAULT_WEIGHT_SUM
    control_share_floor: float = DEFAULT_CONTROL_SHARE_FLOOR
    # WarpMPC constraints with weight_mode=sum: "sum" prices every slack
    # linearly at the node's weight sum S (tracking : slack = 1, the same
    # at any S), plus slack_quadratic_fraction * S of curvature; "fixed"
    # keeps the solver's constants
    slack_price_mode: str = SLACK_PRICE_FIXED
    slack_quadratic_fraction: float = DEFAULT_SLACK_QUADRATIC_FRACTION
    # With slack_price_mode=sum: "learned" lets the head weigh the nodes
    # of the horizon (kappa_k), tracking : slack staying 1 on every node
    node_weight_mode: str = NODE_WEIGHT_FIXED
    node_weight_min: float = DEFAULT_NODE_WEIGHT_MIN

    def __post_init__(self) -> None:
        """
        Reject nonpositive counts, a box narrower than PD's, an unknown
        solver, WarpMPC without its package or with T < 2, invalid
        constraints or constraints without WarpMPC, a negative or
        non-finite tube margin, an unknown tube anchor, a nonpositive
        slack weight scale, and invalid backbone sizes.
        """
        self.validate_backbone()
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")
        validate_constraints(self.constraint_type, self.constraint_count)
        if self.constraint_type not in STUDENT_CONSTRAINT_TYPES:
            raise ValueError(
                f"constraint_type must be one of {STUDENT_CONSTRAINT_TYPES}"
            )
        if not math.isfinite(self.tube_margin_m) or self.tube_margin_m < 0:
            raise ValueError("tube_margin_m must be finite and >= 0")
        if self.tube_anchor not in TUBE_ANCHORS:
            raise ValueError(f"tube_anchor must be one of {TUBE_ANCHORS}")
        if not 0 < self.tube_edge_parallel_max_deg <= 90:
            raise ValueError("tube_edge_parallel_max_deg must be in (0, 90]")
        if not (math.isfinite(self.tube_edge_along_window_m)
                and self.tube_edge_along_window_m >= 0):
            raise ValueError(
                "tube_edge_along_window_m must be finite and >= 0"
            )
        if not (math.isfinite(self.tube_plane_max_offset_m)
                and self.tube_plane_max_offset_m > 0):
            raise ValueError("tube_plane_max_offset_m must be finite and > 0")
        if (self.tube_edge_side_filter
                and self.tube_anchor != TUBE_ANCHOR_ROUTE):
            raise ValueError("tube_edge_side_filter needs tube_anchor=route")
        if not (math.isfinite(self.slack_weight_scale)
                and self.slack_weight_scale > 0):
            raise ValueError("slack_weight_scale must be finite and > 0")
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
        if self.weight_mode == WEIGHT_MODE_SUM:
            # Node 0 and the terminal node have different active sets
            if self.horizon_steps < 2:
                raise ValueError("weight_mode=sum needs horizon_steps >= 2")
            if not (math.isfinite(self.weight_sum) and self.weight_sum > 0):
                raise ValueError("weight_sum must be finite and > 0")
            # Two controls share the floor: 2 rho must leave room
            if not 0 <= self.control_share_floor < 0.5:
                raise ValueError("control_share_floor must be in [0, 0.5)")
        if self.slack_price_mode not in SLACK_PRICE_MODES:
            raise ValueError(
                f"slack_price_mode must be one of {SLACK_PRICE_MODES}"
            )
        if self.slack_price_mode == SLACK_PRICE_SUM:
            # The price is the node's weight sum, so it needs one
            if self.weight_mode != WEIGHT_MODE_SUM:
                raise ValueError("slack_price_mode=sum needs weight_mode=sum")
            if self.constraint_type == CONSTRAINT_NONE:
                raise ValueError("slack_price_mode=sum needs constraints")
            if self.slack_weight_scale != 1.0:
                raise ValueError(
                    "slack_price_mode=sum sets the prices; it takes no "
                    "slack_weight_scale"
                )
            fraction = self.slack_quadratic_fraction
            if not (math.isfinite(fraction) and fraction >= 0):
                raise ValueError(
                    "slack_quadratic_fraction must be finite and >= 0"
                )
        if self.node_weight_mode not in NODE_WEIGHT_MODES:
            raise ValueError(
                f"node_weight_mode must be one of {NODE_WEIGHT_MODES}"
            )
        if self.node_weight_mode == NODE_WEIGHT_LEARNED:
            # kappa scales the node's weights and slack price together
            if self.slack_price_mode != SLACK_PRICE_SUM:
                raise ValueError(
                    "node_weight_mode=learned needs slack_price_mode=sum"
                )
            if not 0 < self.node_weight_min < 1:
                raise ValueError("node_weight_min must be in (0, 1)")
        if (self.constraint_type != CONSTRAINT_NONE
                and self.solver != SOLVER_WARPMPC):
            raise ValueError("constraints need solver=warpmpc")
        if self.solver == SOLVER_WARPMPC:
            if importlib.util.find_spec("warpmpc") is None:
                raise ValueError("solver=warpmpc needs the warpmpc package")
            if self.horizon_steps < WARPMPC_MIN_HORIZON_STEPS:
                raise ValueError(
                    "solver=warpmpc needs horizon_steps >= "
                    f"{WARPMPC_MIN_HORIZON_STEPS}"
                )
        counts = (
            "horizon_steps",
            "train_lqr_iterations",
            "act_lqr_iterations",
            "sqp_iterations",
            "admm_iterations",
        )
        for name in counts:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        validate_bounds(self.accel_bound_mps2, self.steering_rate_bound_radps)


@dataclasses.dataclass
class GradChunk:
    """Rows a collector solved with the MPC student, plus the gradient of
    their summed per-row loss with respect to the head output at the
    collecting weights. The learner re-runs only the network on `obs`."""

    obs: torch.Tensor  # (B,obs_dim) live vehicle rows
    label: torch.Tensor  # (B,2) teacher controls
    raw_grad: torch.Tensor  # (B,T*outputs_per_step) loss gradient
    student_drove: torch.Tensor  # (B,) bool
    loss_sum: torch.Tensor  # scalar, summed row loss at collection
    weight_version: int  # update count of the collecting weights


class MPCStudent(Student):
    """Teacher's backbone (its weights or fresh ones), trained, with a
    head that outputs a per-step quadratic cost; box-constrained iLQR on
    the classic bicycle turns it into controls, and the first control is
    executed."""

    def __init__(
            self,
            teacher: Teacher,
            config: MPCStudentConfig | None = None,
        ) -> None:
        """
        Copy the teacher's encoders and backbone; new cost head.

        Args:
            teacher (Teacher): source of the backbone architecture and,
                with config.init_from_teacher, its weights.
            config (MPCStudentConfig | None): solver settings and
                initialization; None uses the defaults.
        """
        super().__init__()
        self.config = config or MPCStudentConfig()
        self.backbone = student_backbone(
            teacher,
            self.config.init_from_teacher,
            self.config.backbone_sizes(teacher),
            self.config.backbone_checkpoint,
        )
        self.ego_dim = teacher.policy.ego_dim
        # The partner block follows the ego block and the reward and goal
        # context (Backbone.forward)
        self.partner_start_idx = self.ego_dim + self.backbone.context_dim
        # The tube's ellipses read the nearest partner slots
        uses_partners = self.config.constraint_type == CONSTRAINT_TUBE
        if (uses_partners and self.config.constraint_count
                > self.backbone.obs_slots_partners_n):
            raise ValueError("constraint_count exceeds the partner slots")
        self.horizon_steps = self.config.horizon_steps
        # Per step: the cost, then the node weight (learned node weights
        # only)
        self.learned_node_weights = (
            self.config.node_weight_mode == NODE_WEIGHT_LEARNED
        )
        self.outputs_per_step = (
            int(self.learned_node_weights) + COST_OUTPUTS_PER_STEP
        )
        self.head = torch.nn.Linear(
            self.backbone.out_dim,
            self.horizon_steps * self.outputs_per_step,
        )
        # Near-zero outputs: mid-range weights and targets equal to the
        # current state, i.e. hold lane, heading and speed
        torch.nn.init.normal_(self.head.weight, std=HEAD_INIT_STD)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "control_bound",
            torch.tensor(
                [
                    self.config.accel_bound_mps2,
                    self.config.steering_rate_bound_radps,
                ]
            ),
        )
        self.plans = None
        # Previous-plan tube reference, per agent row: the last solved
        # plan's states, the world pose it was made at, whether it exists
        # (cleared by reset_plans), and the poses of the coming act
        self.reference_states = None
        self.reference_poses = None
        self.reference_valid = None
        self.agent_poses = None
        # Route windows (points, validity) of the coming act, per agent row
        self.route_window = None
        # Tube rows (node >= 1, side) of closed-loop solves, how many the
        # projection guard dropped and how many had a route anchor, since
        # the last pop_tube_stats
        self.tube_guard_count = 0
        self.tube_row_count = 0
        self.tube_route_count = 0
        # Route rows with no edge in reach, whose plane sits at the
        # tube_plane_max_offset_m fallback
        self.tube_fallback_count = 0
        # Violated plane rows (node >= 1, side) and real-partner ellipse
        # rows of the closed-loop plans (act), and the summed violations
        self.plane_violation_count = 0
        self.plane_violation_sum_m = 0.0
        self.ellipse_row_count = 0
        self.ellipse_violation_count = 0
        self.ellipse_violation_sum = 0.0
        # The render pass sets record_overlay; act then keeps what it
        # planned with (overlay.py) for the video overlay
        self.record_overlay = False
        self.last_overlay = None
        self.weight_version = 0
        # Rows whose float32 solve gradient was non-finite and that the
        # float64 retry solved (solve_with_gradient), since the last
        # pop_solver_stats; where a failing solve's inputs are dumped
        # before the error (None: no dump)
        self.fp64_retry_rows = 0
        self.failure_dump_dir = None

    def head_output(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Network part of the cost: backbone and head.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            torch.Tensor: (B,T,outputs_per_step) raw head output.
        """
        hidden = self.backbone(
            obs.to(self.head.weight.dtype), self.ego_dim
        )
        return self.head(hidden).view(
            obs.shape[0], self.horizon_steps, self.outputs_per_step
        )

    def cost_from_head_output(
            self,
            raw: torch.Tensor,
            x0: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Elementwise map from the head output to the per-step cost.

        Args:
            raw (torch.Tensor): (B,T,outputs_per_step) head output; only
                the first COST_OUTPUTS_PER_STEP are read.
            x0 (torch.Tensor): (B,5) ego-frame start state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: q_diag (T,B,7) weights and
                target (T,B,7) on tau = (px, py, heading, speed,
                steering, accel, rate).
        """
        node_weights = self.node_weights(raw)
        raw = raw.transpose(0, 1)[..., :COST_OUTPUTS_PER_STEP]
        state_logits = raw[..., :STATE_WEIGHT_COUNT]
        control_logits = raw[
            ..., STATE_WEIGHT_COUNT:STATE_WEIGHT_COUNT + CONTROL_WEIGHT_COUNT
        ]
        target_raw = raw[..., STATE_WEIGHT_COUNT + CONTROL_WEIGHT_COUNT:]

        state_weights = STATE_WEIGHT_MAX * torch.sigmoid(state_logits)
        control_weights = CONTROL_WEIGHT_MIN + (
            CONTROL_WEIGHT_MAX - CONTROL_WEIGHT_MIN
        ) * torch.sigmoid(control_logits)
        if self.config.weight_mode == WEIGHT_MODE_SUM:
            state_weights, control_weights = sum_mode_weights(
                state_logits,
                control_logits,
                raw.new_tensor(STATE_TERM_RANGES),
                U_HI.to(raw),
                self.config.weight_sum,
                self.config.control_share_floor,
            )
            if node_weights is not None:
                state_weights = node_weights[..., None] * state_weights
                control_weights = node_weights[..., None] * control_weights
        zero = torch.zeros_like(raw[..., :1])
        # px gets weight 0: progress along the lane is free
        q_diag = torch.cat([zero, state_weights, control_weights], -1)

        lateral_target = TARGET_LATERAL_RANGE_M * torch.tanh(
            target_raw[..., 0]
        )
        heading_target = TARGET_HEADING_RANGE_RAD * torch.tanh(
            target_raw[..., 1]
        )
        # Speed target is relative to the current speed
        speed_target = x0[:, SPEED] + TARGET_SPEED_CHANGE_RANGE_MPS * (
            torch.tanh(target_raw[..., 2])
        )
        steering_target = STEERING_LIMIT_RAD * torch.tanh(target_raw[..., 3])
        zero_target = torch.zeros_like(lateral_target)
        # Order follows tau (px, py, heading, speed, steering, accel,
        # rate); px and both controls target 0
        target = torch.stack(
            [
                zero_target,
                lateral_target,
                heading_target,
                speed_target,
                steering_target,
                zero_target,
                zero_target,
            ],
            -1,
        )
        return q_diag, target

    def cost_terms(
            self,
            obs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ego-frame start state and the per-step cost from the head.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.

        Returns:
            tuple: x0 (B,5), wheelbase (B,) [m], q_diag (T,B,7) weights
                and target (T,B,7) on tau = (px, py, heading, speed,
                steering, accel, rate).
        """
        x0, wheelbase = ego_state_from_obs(
            obs.to(self.head.weight.dtype)
        )
        q_diag, target = self.cost_from_head_output(
            self.head_output(obs), x0
        )
        return x0, wheelbase, q_diag, target

    def constraints(
            self,
            obs: torch.Tensor,
            raw: torch.Tensor,
            x0: torch.Tensor,
            reference: tuple[torch.Tensor, torch.Tensor] | None = None,
            route: tuple[torch.Tensor, torch.Tensor] | None = None,
            closed_loop: bool = False,
        ) -> torch.Tensor | None:
        """
        Tube constraint parameters over the horizon: planes from the
        observed road edges around the reference positions, and ellipses
        around the nearest partners.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.
            raw (torch.Tensor): (B,T,outputs_per_step) head output.
            x0 (torch.Tensor): (B,5) ego-frame start state.
            reference (tuple | None): previous-plan tube reference,
                positions (T,B,2) and headings (T,B), from act; None
                outside the closed loop (predict), which places the planes
                on the straight reference of a state without a plan.
            route (tuple | None): route windows (B,R,2) and validity (B,R)
                of a route-anchored tube, from act; None anchors the
                planes on the reference positions.
            closed_loop (bool): a solve of act or act_and_grad; only these
                count in the tube statistics (pop_tube_stats), not
                predict's (the probe's).

        Returns:
            torch.Tensor | None: (T,B,V) tube parameters, without
                gradient; None without constraints.
        """
        constraint_type = self.config.constraint_type
        if constraint_type == CONSTRAINT_TUBE:
            obs = obs.to(self.head.weight.dtype)
            if reference is None:
                reference = straight_reference(
                    x0[:, SPEED], self.horizon_steps
                )
            positions, headings = reference
            edges = road_edge_segments(
                obs, self.partner_start_idx, self.backbone
            )
            ego_length_m = obs[:, EGO_LENGTH_IDX] * OBS_NORM_VEH_LENGTH_M
            ego_width_m = obs[:, EGO_WIDTH_IDX] * OBS_NORM_VEH_WIDTH_M
            planes, guarded = tube_planes(
                positions,
                headings,
                *edges,
                ego_length_m,
                ego_width_m,
                self.config.tube_margin_m,
                TUBE_NORMAL_PROJECTION,
                TUBE_SECTOR_FULL,
            )
            anchored = torch.zeros_like(guarded[..., 0])
            if route is not None:
                anchors, tangents, anchored = route_anchors(
                    positions, *route
                )
                side_filter = None
                if self.config.tube_edge_side_filter:
                    side_filter = positions
                route_rows, route_guarded = route_planes(
                    anchors, tangents, anchored, headings, *edges,
                    ego_length_m, ego_width_m, self.config.tube_margin_m,
                    side_filter,
                    math.radians(self.config.tube_edge_parallel_max_deg),
                    self.config.tube_plane_max_offset_m,
                    self.config.tube_edge_along_window_m or math.inf,
                )
                # A node without a route anchor keeps the plan's planes
                planes = torch.where(
                    anchored[..., None, None], route_rows, planes
                )
                guarded = torch.where(
                    anchored[..., None], route_guarded, guarded
                )
                if closed_loop:
                    self.tube_fallback_count += count_fallback_planes(
                        route_rows, anchors, anchored, route_guarded,
                        ego_length_m, ego_width_m,
                        self.config.tube_margin_m,
                        self.config.tube_plane_max_offset_m,
                    )
            # Node 0 has no rows
            if closed_loop:
                self.tube_guard_count += int(guarded[1:].sum())
                self.tube_row_count += guarded[1:].numel()
                self.tube_route_count += 2 * int(anchored[1:].sum())
            # Constant-velocity ellipses of the K nearest partners follow
            # the planes on each node; empty slots are dummies
            ellipses = partner_obstacles(
                obs, self.partner_start_idx, self.config.constraint_count,
                self.horizon_steps,
            )
            return torch.cat([planes.flatten(2), ellipses.flatten(2)], -1)
        return None

    def solve_cost(
            self,
            x0: torch.Tensor,
            wheelbase: torch.Tensor,
            q_diag: torch.Tensor,
            target: torch.Tensor,
            iterations: int,
            u_init: torch.Tensor | None,
            constraints: torch.Tensor | None = None,
            node_slack_price: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Run the MPC on a given cost.

        Args:
            x0 (torch.Tensor): (B,5) ego-frame start state.
            wheelbase (torch.Tensor): (B,) wheelbase [m].
            q_diag (torch.Tensor): (T,B,7) weights on tau.
            target (torch.Tensor): (T,B,7) target for tau.
            iterations (int): iLQR iterations (mpc.pytorch only; WarpMPC
                uses config.sqp_iterations).
            u_init (torch.Tensor | None): (T,B,2) warm start or None.
            constraints (torch.Tensor | None): (T,B,V) tube parameters
                (WarpMPC only), or None.
            node_slack_price (torch.Tensor | None): (T,B) slack price per
                node from node_slack_price, None for slack_prices.

        Returns:
            torch.Tensor: (T,B,2) planned controls.
        """
        if self.config.solver == SOLVER_WARPMPC:
            # Imported here: JAX and Warp exist only in the WarpMPC venv
            from pufferlib.distill.warpmpc_solver import solve_cost_warpmpc
            return solve_cost_warpmpc(
                x0,
                wheelbase,
                q_diag,
                target,
                u_init,
                self.control_bound.to(q_diag.dtype),
                self.config.sqp_iterations,
                self.config.admm_iterations,
                self.config.constraint_type,
                constraints,
                self.config.slack_weight_scale,
                None,
                self.slack_prices(),
                node_slack_price,
                self.config.slack_quadratic_fraction,
            )
        Q, p = target_cost(q_diag, target)
        _, u = solve(
            x0,
            Q,
            p,
            Bicycle(wheelbase),
            self.horizon_steps,
            u_init=u_init,
            iters=iterations,
            u_lower=-self.control_bound.to(q_diag.dtype),
            u_upper=self.control_bound.to(q_diag.dtype),
        )
        return u

    def node_weights(
            self,
            raw: torch.Tensor,
        ) -> torch.Tensor | None:
        """
        Learned node weights kappa_k in [node_weight_min,
        2 - node_weight_min], 1 for equal logits.

        Args:
            raw (torch.Tensor): (B,T,outputs_per_step) head output; kappa
                reads the last output of each step.

        Returns:
            torch.Tensor | None: (T,B) kappa, None unless
                node_weight_mode=learned.
        """
        if not self.learned_node_weights:
            return None
        logits = raw[..., -1].transpose(0, 1)
        centred = logits - logits.mean(dim=0, keepdim=True)
        node_min = self.config.node_weight_min
        return 1 + (1 - node_min) * torch.tanh(centred)

    def node_slack_price(
            self,
            raw: torch.Tensor,
        ) -> torch.Tensor | None:
        """
        Linear slack price per node, S * kappa_k, differentiable.

        Args:
            raw (torch.Tensor): (B,T,outputs_per_step) head output.

        Returns:
            torch.Tensor | None: (T,B) price, None unless
                node_weight_mode=learned.
        """
        node_weights = self.node_weights(raw)
        if node_weights is None:
            return None
        return self.config.weight_sum * node_weights

    def slack_prices(self) -> tuple[float, float] | None:
        """
        Slack prices of the WarpMPC solve.

        Returns:
            tuple[float, float] | None: (linear, quadratic) price of every
                slack in slack_price_mode=sum, None for the solver's
                constants.
        """
        if self.config.slack_price_mode != SLACK_PRICE_SUM:
            return None
        weight_sum = self.config.weight_sum
        return (
            weight_sum, self.config.slack_quadratic_fraction * weight_sum
        )

    def solve_controls(
            self,
            obs: torch.Tensor,
            iterations: int,
            u_init: torch.Tensor | None,
        ) -> torch.Tensor:
        """
        Run the MPC on the head's cost.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations.
            iterations (int): iLQR iterations.
            u_init (torch.Tensor | None): (T,B,2) warm start or None.

        Returns:
            torch.Tensor: (T,B,2) planned controls.
        """
        x0, wheelbase = ego_state_from_obs(obs.to(self.head.weight.dtype))
        raw = self.head_output(obs)
        q_diag, target = self.cost_from_head_output(raw, x0)
        return self.solve_cost(
            x0, wheelbase, q_diag, target, iterations, u_init,
            self.constraints(obs, raw, x0), self.node_slack_price(raw),
        )

    def predict(
            self,
            obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        First control of a cold-started solve, differentiable. Training
        never warm-starts: buffer samples are unrelated states.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of vehicles.

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate).
        """
        # Empty slots (length 0) divide by a zero wheelbase, and one NaN
        # row poisons the whole mpc.pytorch batch; training data has none
        if not bool(vehicle_rows(obs).all()):
            raise ValueError("predict got rows without a vehicle")
        controls = self.solve_controls(
            obs, self.config.train_lqr_iterations, None
        )
        return controls[0]

    def act(
            self,
            obs: torch.Tensor,
            driven_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        First control of a solve warm-started from each agent's shifted
        previous plan; keeps the new plan.

        Deprecated for collection: act_and_grad does the same solve and
        also ships the training gradient. Its plan bookkeeping is to be
        factored out and shared; act stays for the evaluator.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of all agents;
                evaluators also pass empty slots (vehicle length 0).
            driven_mask (torch.Tensor): (B,) bool; all vehicle rows are
                solved so plans stay aligned with agent rows.

        Returns:
            torch.Tensor: (B,2) control (accel, steering_rate); zero for
                empty slots.
        """
        batch_size = obs.shape[0]
        dtype = self.head.weight.dtype
        self.allocate_plan_state(batch_size, dtype, obs.device)
        # Empty slots would make the solve NaN for every row
        solved_rows = vehicle_rows(obs)
        first_controls = torch.zeros(
            batch_size, NU, dtype=dtype, device=obs.device
        )
        if not bool(solved_rows.any()):
            return first_controls
        u_init = None
        if self.config.act_warm_start:
            u_init = self.plans[:, solved_rows]
        solved_obs = obs[solved_rows]
        poses = self.take_agent_poses(batch_size)
        route = self.take_route_window(batch_size, solved_rows)
        # solve_controls inlined: the overlay reuses its cost and planes
        with torch.no_grad():
            x0, wheelbase = ego_state_from_obs(solved_obs.to(dtype))
            reference = self.previous_plan_reference(x0, solved_rows, poses)
            raw = self.head_output(solved_obs)
            q_diag, target = self.cost_from_head_output(raw, x0)
            constraints = self.constraints(
                solved_obs, raw, x0, reference, route, closed_loop=True
            )
            controls = self.solve_cost(
                x0, wheelbase, q_diag, target,
                self.config.act_lqr_iterations, u_init, constraints,
                self.node_slack_price(raw),
            )
            self.keep_reference_plan(
                solved_rows, x0, wheelbase, controls, poses
            )
            if self.config.constraint_type == CONSTRAINT_TUBE:
                self.count_violations(x0, wheelbase, controls, constraints)
        if self.record_overlay:
            self.last_overlay = self.overlay_rows(
                batch_size, solved_rows, x0, wheelbase, controls, q_diag,
                target, constraints,
            )
        # Next step starts from the rest of this plan, last control repeated
        self.plans[:, solved_rows] = torch.cat([controls[1:], controls[-1:]])
        self.plans[:, ~solved_rows] = 0.0
        first_controls[solved_rows] = controls[0]
        return first_controls

    def count_violations(
            self,
            x0: torch.Tensor,
            wheelbase: torch.Tensor,
            controls: torch.Tensor,
            constraints: torch.Tensor,
        ) -> None:
        """
        Count the tube rows the solved plans violate (their slack is in
        use): planes by the planned position's distance outside, ellipses
        by how far the normalized distance falls below 1; nodes >= 1.

        Args:
            x0 (torch.Tensor): (S,5) start states of the solved rows.
            wheelbase (torch.Tensor): (S,) wheelbase [m].
            controls (torch.Tensor): (T,S,2) plans.
            constraints (torch.Tensor): (T,S,V) tube parameters.
        """
        positions = planned_states(x0, wheelbase, controls)[1:, :, :PY + 1]
        planes = constraints[1:, :, :TUBE_PLANE_VALUE_COUNT].unflatten(
            -1, (TUBE_PLANE_COUNT, PLANE_PARAM_COUNT)
        )
        plane_values = (
            (planes[..., :2] * positions[:, :, None]).sum(-1)
            - planes[..., 2]
        )
        plane_violated = plane_values > PLANE_VIOLATION_TOL_M
        self.plane_violation_count += int(plane_violated.sum())
        self.plane_violation_sum_m += float(plane_values[plane_violated].sum())
        ellipses = constraints[1:, :, TUBE_PLANE_VALUE_COUNT:].unflatten(
            -1, (self.config.constraint_count, OBSTACLE_PARAM_COUNT)
        )
        offset = positions[:, :, None] - ellipses[..., :2]
        cos_heading, sin_heading = ellipses[..., 2], ellipses[..., 3]
        along = (
            offset[..., 0] * cos_heading + offset[..., 1] * sin_heading
        ) / ellipses[..., 4]
        across = (
            offset[..., 1] * cos_heading - offset[..., 0] * sin_heading
        ) / ellipses[..., 5]
        shortfall = 1 - torch.sqrt(along.square() + across.square())
        # Empty partner slots are dummies DUMMY_DISTANCE_M ahead
        real = ellipses[..., :2].norm(dim=-1) < DUMMY_DISTANCE_M / 2
        ellipse_violated = real & (shortfall > ELLIPSE_VIOLATION_TOL)
        self.ellipse_row_count += int(real.sum())
        self.ellipse_violation_count += int(ellipse_violated.sum())
        self.ellipse_violation_sum += float(shortfall[ellipse_violated].sum())

    def overlay_rows(
            self,
            batch_size: int,
            solved_rows: torch.Tensor,
            x0: torch.Tensor,
            wheelbase: torch.Tensor,
            controls: torch.Tensor,
            q_diag: torch.Tensor,
            target: torch.Tensor,
            constraints: torch.Tensor | None,
        ) -> torch.Tensor:
        """
        Overlay rows (overlay.py) of one act solve; rows without a vehicle
        stay invalid (all zeros). The tube's planes and partner ellipses
        are drawn.

        Args:
            batch_size (int): B, rows of act's obs.
            solved_rows (torch.Tensor): (B,) bool, the solved rows.
            x0 (torch.Tensor): (S,5) start states of the solved rows.
            wheelbase (torch.Tensor): (S,) wheelbase [m].
            controls (torch.Tensor): (T,S,2) plans.
            q_diag (torch.Tensor): (T,S,7) weights on tau.
            target (torch.Tensor): (T,S,7) targets for tau.
            constraints (torch.Tensor | None): (T,S,V) tube parameters, or
                None.

        Returns:
            torch.Tensor: (B,OVERLAY_FLOATS) float32 rows.
        """
        planes, ellipses = None, None
        if self.config.constraint_type == CONSTRAINT_TUBE:
            planes = constraints[..., :TUBE_PLANE_VALUE_COUNT].unflatten(
                -1, (TUBE_PLANE_COUNT, PLANE_PARAM_COUNT)
            )
            ellipses = constraints[..., TUBE_PLANE_VALUE_COUNT:].unflatten(
                -1, (self.config.constraint_count, OBSTACLE_PARAM_COUNT)
            )
        rows = torch.zeros(
            batch_size, OVERLAY_FLOATS, dtype=torch.float32,
            device=x0.device,
        )
        rows[solved_rows] = pack_overlay(
            planned_positions(x0, wheelbase, controls),
            target[..., PY],
            q_diag[..., PY],
            planes,
            ellipses,
        )
        return rows

    def act_and_grad(
            self,
            obs: torch.Tensor,
            label: torch.Tensor,
            live_mask: torch.Tensor,
            driven_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, GradChunk | None]:
        """
        One warm-started solve of every vehicle row that serves driving
        and training: the control is executed where driven_mask is set,
        and the gradient of the summed per-row loss of live rows with
        respect to the head output is shipped for the learner. The
        network is not differentiated here.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of all agents;
                empty slots (vehicle length 0) are skipped.
            label (torch.Tensor): (B,2) teacher controls.
            live_mask (torch.Tensor): (B,) bool, rows that count as
                labels.
            driven_mask (torch.Tensor): (B,) bool, rows the student
                drives; recorded in the chunk.

        Returns:
            tuple: (B,2) control (accel, steering_rate), zero for empty
                slots, and the chunk of live vehicle rows (None if there
                are none).
        """
        batch_size = obs.shape[0]
        dtype = self.head.weight.dtype
        self.allocate_plan_state(batch_size, dtype, obs.device)
        solved_rows = vehicle_rows(obs)
        first_controls = torch.zeros(
            batch_size, NU, dtype=dtype, device=obs.device
        )
        if not bool(solved_rows.any()):
            return first_controls, None
        u_init = None
        if self.config.act_warm_start:
            u_init = self.plans[:, solved_rows]
        solved_obs = obs[solved_rows]
        counted_rows = live_mask[solved_rows]
        poses = self.take_agent_poses(batch_size)
        route = self.take_route_window(batch_size, solved_rows)
        x0, wheelbase = ego_state_from_obs(solved_obs.to(dtype))
        reference = self.previous_plan_reference(x0, solved_rows, poses)
        controls, raw_grad, loss_sum = self.solve_with_gradient(
            solved_obs,
            label[solved_rows],
            counted_rows,
            self.config.act_lqr_iterations,
            u_init,
            reference,
            route,
        )
        with torch.no_grad():
            self.keep_reference_plan(
                solved_rows, x0, wheelbase, controls, poses
            )
        self.plans[:, solved_rows] = torch.cat(
            [controls[1:], controls[-1:]]
        )
        self.plans[:, ~solved_rows] = 0.0
        first_controls[solved_rows] = controls[0]
        chunk_rows = solved_rows & live_mask
        chunk = GradChunk(
            obs=obs[chunk_rows],
            label=label[chunk_rows],
            raw_grad=raw_grad,
            student_drove=driven_mask[chunk_rows],
            loss_sum=loss_sum,
            weight_version=self.weight_version,
        )
        return first_controls, chunk

    def solve_with_gradient(
            self,
            obs: torch.Tensor,
            label: torch.Tensor,
            counted_rows: torch.Tensor,
            iterations: int,
            u_init: torch.Tensor | None,
            reference: tuple[torch.Tensor, torch.Tensor] | None = None,
            route: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve the MPC of vehicle rows and differentiate the summed row
        loss with respect to the head output. The network runs under
        no_grad; only the solve and its backward are differentiated.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of vehicles.
            label (torch.Tensor): (B,2) teacher controls.
            counted_rows (torch.Tensor): (B,) bool; rows that enter the
                loss.
            iterations (int): iLQR iterations.
            u_init (torch.Tensor | None): (T,B,2) warm start or None.
            reference (tuple | None): previous-plan tube reference
                (constraints), or None.
            route (tuple | None): route windows of a route-anchored tube
                (constraints), or None.

        Returns:
            tuple: planned controls (T,B,2) without graph, head-output
                gradient (counted,T*outputs_per_step), and the
                summed loss of the counted rows (scalar).
        """
        dtype = self.head.weight.dtype
        x0, wheelbase = ego_state_from_obs(obs.to(dtype))
        with torch.no_grad():
            raw = self.head_output(obs)
        raw.requires_grad_(True)
        q_diag, target = self.cost_from_head_output(raw, x0)
        controls = self.solve_cost(
            x0, wheelbase, q_diag, target, iterations, u_init,
            self.constraints(
                obs, raw, x0, reference, route, closed_loop=True
            ),
            self.node_slack_price(raw),
        )
        # Per-row mean over the two controls, summed over rows: the
        # learner divides by its row count to match distillation_loss
        scale = U_HI.to(dtype=dtype, device=obs.device)
        solved_label = label.to(dtype)
        row_loss = (((controls[0] - solved_label) / scale) ** 2).mean(-1)
        row_loss[counted_rows].sum().backward()
        raw_grad = raw.grad
        # A rare float32 adjoint failure is retried in float64 on its rows
        # only; a row that still fails stops the run (fail-fast)
        failed_rows = ~torch.isfinite(raw_grad).flatten(1).all(1)
        if bool(failed_rows.any()):
            raw_grad = raw_grad.clone()
            raw_grad[failed_rows] = self.float64_row_gradient(
                obs, label, counted_rows, iterations, u_init, reference,
                route, raw.detach(), failed_rows,
            )
            self.fp64_retry_rows += int(failed_rows.sum())
        return (
            controls.detach(),
            raw_grad[counted_rows].flatten(1),
            row_loss[counted_rows].detach().sum(),
        )

    def float64_row_gradient(
            self,
            obs: torch.Tensor,
            label: torch.Tensor,
            counted_rows: torch.Tensor,
            iterations: int,
            u_init: torch.Tensor | None,
            reference: tuple[torch.Tensor, torch.Tensor] | None,
            route: tuple[torch.Tensor, torch.Tensor] | None,
            raw: torch.Tensor,
            rows: torch.Tensor,
        ) -> torch.Tensor:
        """
        The first-control loss gradient of some rows, solved again in
        float64 from the same head output, warm start and constraints
        (built in float32 as in the first solve). Rows are independent in
        the batched solve, so they can be solved apart. Dumps the inputs
        (failure_dump_dir) and raises if it is still non-finite.

        Args:
            obs (torch.Tensor): (B,obs_dim) observations of the solve.
            label (torch.Tensor): (B,2) teacher controls.
            counted_rows (torch.Tensor): (B,) bool, rows in the loss.
            iterations (int): solver iterations.
            u_init (torch.Tensor | None): (T,B,2) warm start or None.
            reference (tuple | None): tube reference, positions (T,B,2)
                and headings (T,B), or None.
            route (tuple | None): route windows (B,R,2) and validity
                (B,R), or None.
            raw (torch.Tensor): (B,T,outputs_per_step) head output,
                detached.
            rows (torch.Tensor): (B,) bool, the rows to solve again.

        Returns:
            torch.Tensor: (n,T,outputs_per_step) gradient of the rows, in
                raw's dtype.
        """
        dtype = raw.dtype
        row_obs = obs[rows]
        row_reference = None
        if reference is not None:
            row_reference = tuple(values[:, rows] for values in reference)
        row_route = None
        if route is not None:
            row_route = tuple(values[rows] for values in route)
        x0, wheelbase = ego_state_from_obs(row_obs.to(dtype))
        constraints = self.constraints(
            row_obs, raw[rows], x0, row_reference, row_route
        )
        raw64 = raw[rows].double().requires_grad_(True)
        x0_64, wheelbase_64 = x0.double(), wheelbase.double()
        q_diag, target = self.cost_from_head_output(raw64, x0_64)
        row_u_init = None if u_init is None else u_init[:, rows].double()
        controls = self.solve_cost(
            x0_64, wheelbase_64, q_diag, target, iterations, row_u_init,
            None if constraints is None else constraints.double(),
            self.node_slack_price(raw64),
        )
        scale = U_HI.to(controls)
        row_loss = (
            ((controls[0] - label[rows].double()) / scale) ** 2
        ).mean(-1)
        row_loss[counted_rows[rows]].sum().backward()
        if not bool(torch.isfinite(raw64.grad).all()):
            self.dump_failing_solve(
                row_obs, label[rows], counted_rows[rows], row_u_init,
                row_reference, row_route, raw[rows],
            )
            raise ValueError("non-finite gradient from the MPC solve")
        return raw64.grad.to(dtype)

    def dump_failing_solve(
            self,
            obs: torch.Tensor,
            label: torch.Tensor,
            counted_rows: torch.Tensor,
            u_init: torch.Tensor | None,
            reference: tuple[torch.Tensor, torch.Tensor] | None,
            route: tuple[torch.Tensor, torch.Tensor] | None,
            raw: torch.Tensor,
        ) -> None:
        """
        Save the inputs of the rows whose gradient stayed non-finite, and
        the weights, to failure_dump_dir/FAILING_SOLVE_NAME; nothing
        without a dump directory.

        Args:
            obs (torch.Tensor): (n,obs_dim) observations.
            label (torch.Tensor): (n,2) teacher controls.
            counted_rows (torch.Tensor): (n,) bool.
            u_init (torch.Tensor | None): (T,n,2) warm start.
            reference (tuple | None): tube reference of the rows.
            route (tuple | None): route windows of the rows.
            raw (torch.Tensor): (n,T,outputs_per_step) head output.
        """
        if self.failure_dump_dir is None:
            return
        torch.save(
            {
                "obs": to_cpu(obs),
                "label": to_cpu(label),
                "counted_rows": to_cpu(counted_rows),
                "u_init": to_cpu(u_init),
                "reference": to_cpu(reference),
                "route": to_cpu(route),
                "raw": to_cpu(raw),
                "student_state": {
                    key: value.detach().cpu()
                    for key, value in self.state_dict().items()
                },
            },
            os.path.join(self.failure_dump_dir, FAILING_SOLVE_NAME),
        )

    def pop_solver_stats(self) -> dict:
        """
        Rows the float64 retry solved since the last call, then reset.

        Returns:
            dict: {"solver/fp64_retry_rows": count}.
        """
        stats = {"solver/fp64_retry_rows": self.fp64_retry_rows}
        self.fp64_retry_rows = 0
        return stats

    def apply_cost_gradient(
            self,
            chunk: GradChunk,
            row_count: int,
        ) -> None:
        """
        Add the parameter gradient of a chunk to `.grad`: the network is
        re-run at the current weights and the shipped head-output
        gradient is pushed through it. The caller steps the optimizer.

        Args:
            chunk (GradChunk): rows from act_and_grad, possibly collected
                with older weights.
            row_count (int): rows in the whole update; the loss is the
                mean over them.
        """
        raw = self.head_output(chunk.obs)
        raw.backward(
            chunk.raw_grad.view_as(raw).to(raw.dtype) / row_count
        )

    def reset_plans(
            self,
            new_episode_mask: torch.Tensor,
        ) -> None:
        """
        Zero the warm start and forget the previous-plan tube reference of
        agents that start a new episode.

        Args:
            new_episode_mask (torch.Tensor): (B,) bool.
        """
        if self.plans is None:
            return
        mask = new_episode_mask.to(self.plans.device)
        self.plans[:, mask] = 0.0
        self.reference_valid[mask] = False

    def take_plan_state(self) -> dict:
        """
        Hand over the per-agent state and start fresh: the next act
        allocates it for its own row count.

        Returns:
            dict: attribute name -> value, PLAN_STATE_ATTRIBUTES.
        """
        plan_state = {
            name: getattr(self, name) for name in PLAN_STATE_ATTRIBUTES
        }
        for name in PLAN_STATE_ATTRIBUTES:
            setattr(self, name, None)
        return plan_state

    def restore_plan_state(
            self,
            plan_state: dict,
        ) -> None:
        """
        Put back the per-agent state from take_plan_state.

        Args:
            plan_state (dict): take_plan_state's result.
        """
        for name in PLAN_STATE_ATTRIBUTES:
            setattr(self, name, plan_state[name])

    @property
    def needs_agent_poses(self) -> bool:
        """
        Returns:
            bool: act needs set_agent_poses first (the tube's planes sit
                on the shifted previous plan).
        """
        return self.config.constraint_type == CONSTRAINT_TUBE

    @property
    def needs_route_window(self) -> bool:
        """
        Returns:
            bool: act needs set_route_window first (route-anchored tube).
        """
        return (
            self.config.constraint_type == CONSTRAINT_TUBE
            and self.config.tube_anchor == TUBE_ANCHOR_ROUTE
        )

    def pop_tube_stats(self) -> dict:
        """
        Fractions of closed-loop tube rows (node >= 1, side) the guard
        dropped and, for a route-anchored tube, that had a route anchor,
        since the last call, then reset.

        Returns:
            dict: {"tube/guard_fraction": value} plus
                "tube/route_fraction" for a route-anchored tube, or {}
                without rows.
        """
        row_count = self.tube_row_count
        guard_count = self.tube_guard_count
        route_count = self.tube_route_count
        fallback_count = self.tube_fallback_count
        self.tube_row_count = 0
        self.tube_guard_count = 0
        self.tube_route_count = 0
        self.tube_fallback_count = 0
        if row_count == 0:
            return {}
        stats = {"tube/guard_fraction": guard_count / row_count}
        if self.needs_route_window:
            stats["tube/route_fraction"] = route_count / row_count
            stats["tube/route_fallback_fraction"] = (
                fallback_count / max(route_count, 1)
            )
        stats.update(self.pop_violation_stats(row_count))
        return stats

    def pop_violation_stats(
            self,
            plane_row_count: int,
        ) -> dict:
        """
        Violation shares and mean sizes of the closed-loop plans since the
        last call (count_violations), then reset.

        Args:
            plane_row_count (int): tube plane rows counted meanwhile.

        Returns:
            dict: tube/plane_violation_fraction, tube/plane_violation_mean_m
                and the ellipse ones (fraction of real-partner rows, mean
                normalized shortfall); a mean is 0 without violations.
        """
        plane_count = self.plane_violation_count
        ellipse_count = self.ellipse_violation_count
        stats = {
            "tube/plane_violation_fraction": plane_count / plane_row_count,
            "tube/plane_violation_mean_m": (
                self.plane_violation_sum_m / max(plane_count, 1)
            ),
            "tube/ellipse_violation_fraction": (
                ellipse_count / max(self.ellipse_row_count, 1)
            ),
            "tube/ellipse_violation_mean": (
                self.ellipse_violation_sum / max(ellipse_count, 1)
            ),
        }
        self.plane_violation_count = 0
        self.plane_violation_sum_m = 0.0
        self.ellipse_row_count = 0
        self.ellipse_violation_count = 0
        self.ellipse_violation_sum = 0.0
        return stats

    def set_route_window(
            self,
            window: torch.Tensor,
            valid: torch.Tensor,
        ) -> None:
        """
        Route windows of the agents of the coming act, in its obs row
        order (Drive.get_route_window); one act uses them up.

        Args:
            window (torch.Tensor): (B,R,2) route points in each ego frame
                [m].
            valid (torch.Tensor): (B,R) bool.
        """
        self.route_window = (window, valid)

    def take_route_window(
            self,
            batch_size: int,
            solved_rows: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        The route windows set for this act, of the solved rows, cleared so
        a later act cannot reuse them; None when the student needs none.

        Args:
            batch_size (int): B, rows of act's obs.
            solved_rows (torch.Tensor): (B,) bool.

        Returns:
            tuple | None: windows (S,R,2) on the plan device and validity
                (S,R).
        """
        if not self.needs_route_window:
            return None
        route = self.route_window
        self.route_window = None
        if route is None or route[0].shape[0] != batch_size:
            raise ValueError(
                "tube_anchor=route needs set_route_window with one window "
                "per obs row before every act"
            )
        window, valid = route
        solved = solved_rows.to(window.device)
        return (
            window[solved].to(self.plans),
            valid[solved].to(self.plans.device),
        )

    def set_agent_poses(
            self,
            poses: torch.Tensor,
        ) -> None:
        """
        World poses of the agents of the coming act, in its obs row order;
        one act uses them up.

        Args:
            poses (torch.Tensor): (B,3) (x, y, heading) [m, m, rad].
        """
        self.agent_poses = poses

    def allocate_plan_state(
            self,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
        """
        Per-agent warm starts and tube references, zero and invalid, when
        the row count changes (or on first use).

        Args:
            batch_size (int): B, agent rows.
            dtype (torch.dtype): float dtype.
            device (torch.device): device.
        """
        plan_shape = (self.horizon_steps, batch_size, NU)
        if self.plans is not None and self.plans.shape == plan_shape:
            return
        self.plans = torch.zeros(plan_shape, dtype=dtype, device=device)
        self.reference_states = torch.zeros(
            self.horizon_steps, batch_size, NX, dtype=dtype, device=device
        )
        self.reference_poses = torch.zeros(
            batch_size, POSE_DIM, dtype=dtype, device=device
        )
        self.reference_valid = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

    def take_agent_poses(
            self,
            batch_size: int,
        ) -> torch.Tensor | None:
        """
        The poses set for this act, cleared so a later act cannot reuse
        them; None when the student needs none.

        Args:
            batch_size (int): B, rows of act's obs.

        Returns:
            torch.Tensor | None: (B,3) world poses on the plan device.
        """
        if not self.needs_agent_poses:
            return None
        poses = self.agent_poses
        self.agent_poses = None
        if poses is None or poses.shape != (batch_size, POSE_DIM):
            raise ValueError(
                "the tube needs set_agent_poses with "
                "one pose per obs row before every act"
            )
        return poses.to(self.plans)

    def previous_plan_reference(
            self,
            x0: torch.Tensor,
            solved_rows: torch.Tensor,
            poses: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Tube reference of the solved rows: each agent's previous plan
        shifted into its current frame, or the straight reference where
        it has none.

        Args:
            x0 (torch.Tensor): (S,5) start states of the solved rows.
            solved_rows (torch.Tensor): (B,) bool.
            poses (torch.Tensor | None): (B,3) world poses, None when the
                student needs no previous-plan reference.

        Returns:
            tuple | None: positions (T,S,2) and headings (T,S), or None.
        """
        if poses is None:
            return None
        straight_positions, straight_headings = straight_reference(
            x0[:, SPEED], self.horizon_steps
        )
        shifted_positions, shifted_headings = shifted_reference(
            self.reference_states[:, solved_rows],
            self.reference_poses[solved_rows],
            poses[solved_rows],
        )
        has_plan = self.reference_valid[solved_rows]
        return (
            torch.where(
                has_plan[:, None], shifted_positions, straight_positions
            ),
            torch.where(has_plan, shifted_headings, straight_headings),
        )

    def keep_reference_plan(
            self,
            solved_rows: torch.Tensor,
            x0: torch.Tensor,
            wheelbase: torch.Tensor,
            controls: torch.Tensor,
            poses: torch.Tensor | None,
        ) -> None:
        """
        Store the new plans as the next step's tube reference; rows that
        were not solved have none.

        Args:
            solved_rows (torch.Tensor): (B,) bool.
            x0 (torch.Tensor): (S,5) start states of the solved rows.
            wheelbase (torch.Tensor): (S,) wheelbase [m].
            controls (torch.Tensor): (T,S,2) new plans.
            poses (torch.Tensor | None): (B,3) world poses the plans were
                made at; None stores nothing.
        """
        if poses is None:
            return
        self.reference_states[:, solved_rows] = planned_states(
            x0, wheelbase, controls
        )
        self.reference_poses[:] = poses
        self.reference_valid[:] = solved_rows
