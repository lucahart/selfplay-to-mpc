"""Fixed-sum weight mode of the MPC student's cost (students.sum_mode_weights,
MPCStudentConfig.weight_mode, design doc section 2.3): the range-scaled sum
on every node, zero node-0 state weights, the control floor, the gradient,
and an unchanged default mode."""

import pytest
import torch

from pufferlib.distill.bicycle import U_HI
from pufferlib.distill.students import (
    CONTROL_WEIGHT_MAX,
    CONTROL_WEIGHT_MIN,
    SOLVER_MPC_PYTORCH,
    STATE_TERM_RANGES,
    STATE_WEIGHT_MAX,
    NODE_WEIGHT_LEARNED,
    SLACK_PRICE_SUM,
    SOLVER_WARPMPC,
    WEIGHT_MODE_SIGMOID,
    WEIGHT_MODE_SUM,
    MPCStudentConfig,
    sum_mode_weights,
)
from pufferlib.distill.teacher import Teacher

from conftest import needs_teacher
from test_students import fresh_mpc_student, moving_obs

HORIZON_STEPS = 6
BATCH_SIZE = 5
WEIGHT_SUM = 1.0
FLOOR = 0.01
SUM_TOL = 1e-10
LOGIT_SCALE = 3.0


def random_weights(
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixed-sum weights of random logits.

    Args:
        seed (int): generator seed.
        dtype (torch.dtype): float dtype.

    Returns:
        tuple: (T,B,4) state weights, (T,B,2) control weights, (6,) ranges.
    """
    generator = torch.Generator().manual_seed(seed)
    state_logits = LOGIT_SCALE * torch.randn(
        HORIZON_STEPS, BATCH_SIZE, 4, generator=generator, dtype=dtype
    )
    control_logits = LOGIT_SCALE * torch.randn(
        HORIZON_STEPS, BATCH_SIZE, 2, generator=generator, dtype=dtype
    )
    state_ranges = torch.tensor(STATE_TERM_RANGES, dtype=dtype)
    control_ranges = U_HI.to(dtype)
    state_weights, control_weights = sum_mode_weights(
        state_logits, control_logits, state_ranges, control_ranges,
        WEIGHT_SUM, FLOOR,
    )
    ranges = torch.cat([state_ranges, control_ranges])
    return state_weights, control_weights, ranges


def test_range_scaled_sum_is_s_on_every_node() -> None:
    """sum_{j in A_k} r_j^2 q_kj = S at k = 0 (controls), in the middle
    (all six) and at T-1 (states)."""
    state_weights, control_weights, ranges = random_weights()
    scaled = torch.cat([state_weights, control_weights], -1) * ranges.square()
    node0 = scaled[0, :, 4:].sum(-1)
    middle = scaled[1:-1].sum(-1)
    terminal = scaled[-1, :, :4].sum(-1)
    for sums in (node0, middle, terminal):
        assert torch.allclose(
            sums, torch.full_like(sums, WEIGHT_SUM), atol=SUM_TOL
        )


def test_node0_states_are_zero_and_the_control_floor_holds() -> None:
    """Node-0 state weights are 0; every control share is at least rho on
    nodes 0..T-2, and the terminal controls sit exactly at the floor."""
    state_weights, control_weights, ranges = random_weights(seed=1)
    assert float(state_weights[0].abs().max()) == 0.0
    control_shares = control_weights * ranges[4:].square() / WEIGHT_SUM
    assert bool((control_shares[:-1] >= FLOOR - SUM_TOL).all())
    assert torch.allclose(
        control_shares[-1], torch.full_like(control_shares[-1], FLOOR),
        atol=SUM_TOL,
    )


def test_sum_mode_weights_pass_gradcheck() -> None:
    """The weights are differentiable in the logits (float64 gradcheck)."""
    generator = torch.Generator().manual_seed(2)
    state_logits = torch.randn(
        3, 2, 4, generator=generator, dtype=torch.float64,
        requires_grad=True,
    )
    control_logits = torch.randn(
        3, 2, 2, generator=generator, dtype=torch.float64,
        requires_grad=True,
    )
    state_ranges = torch.tensor(STATE_TERM_RANGES, dtype=torch.float64)
    control_ranges = U_HI.to(torch.float64)

    def weights(state, control):
        return sum_mode_weights(
            state, control, state_ranges, control_ranges, WEIGHT_SUM, FLOOR
        )

    assert torch.autograd.gradcheck(weights, (state_logits, control_logits))


@pytest.mark.parametrize("fields", [
    {"weight_mode": "softmax"},
    {"weight_mode": WEIGHT_MODE_SUM, "weight_sum": 0.0},
    {"weight_mode": WEIGHT_MODE_SUM, "control_share_floor": 0.5},
])
def test_invalid_weight_mode_settings_are_rejected(
        fields: dict,
    ) -> None:
    """Unknown modes, a cap with the fixed sum, S <= 0 and 2 rho >= 1
    raise."""
    with pytest.raises(ValueError):
        MPCStudentConfig(solver=SOLVER_MPC_PYTORCH, **fields)


@needs_teacher
def test_default_mode_is_unchanged(
        teacher: Teacher,
    ) -> None:
    """The default is the sigmoid mode, and its weights are the original
    formula; the sum mode gives other weights from the same head."""
    assert MPCStudentConfig().weight_mode == WEIGHT_MODE_SIGMOID
    obs = moving_obs(teacher)[:8]
    student = fresh_mpc_student(
        teacher, teacher.device, solver=SOLVER_MPC_PYTORCH
    )
    with torch.no_grad():
        raw = student.head_output(obs)
        _, _, q_diag, _ = student.cost_terms(obs)
    logits = raw.transpose(0, 1)
    expected_states = STATE_WEIGHT_MAX * torch.sigmoid(logits[..., :4])
    expected_controls = CONTROL_WEIGHT_MIN + (
        CONTROL_WEIGHT_MAX - CONTROL_WEIGHT_MIN
    ) * torch.sigmoid(logits[..., 4:6])
    assert torch.allclose(q_diag[..., 1:5], expected_states)
    assert torch.allclose(q_diag[..., 5:7], expected_controls)
    assert float(q_diag[..., 0].abs().max()) == 0.0
    summed = fresh_mpc_student(
        teacher, teacher.device, solver=SOLVER_MPC_PYTORCH,
        weight_mode=WEIGHT_MODE_SUM,
    )
    with torch.no_grad():
        _, _, summed_q, _ = summed.cost_terms(obs)
    assert not torch.allclose(summed_q, q_diag)


TUBE_SETTINGS = {
    "solver": SOLVER_WARPMPC,
    "horizon_steps": HORIZON_STEPS,
    "constraint_type": "tube",
    "constraint_count": 2,
}


@pytest.mark.parametrize("fields", [
    {"slack_price_mode": "quadratic"},
    {"slack_price_mode": SLACK_PRICE_SUM},
    {"slack_price_mode": SLACK_PRICE_SUM, "weight_mode": WEIGHT_MODE_SUM,
     "constraint_type": "none", "constraint_count": 0},
    {"slack_price_mode": SLACK_PRICE_SUM, "weight_mode": WEIGHT_MODE_SUM,
     "slack_weight_scale": 2.0},
    {"slack_price_mode": SLACK_PRICE_SUM, "weight_mode": WEIGHT_MODE_SUM,
     "slack_quadratic_fraction": -0.1},
])
def test_invalid_slack_price_settings_are_rejected(
        fields: dict,
    ) -> None:
    """Unknown modes, a sum price without the sum weights or without
    constraints, with a slack scale or a negative curvature raise."""
    pytest.importorskip("warpmpc")
    settings = dict(TUBE_SETTINGS)
    settings.update(fields)
    with pytest.raises(ValueError):
        MPCStudentConfig(**settings)


def test_sum_slack_price_is_valid_with_the_sum_weights() -> None:
    """slack_price_mode=sum with weight_mode=sum and constraints is a
    valid config, linear only by default."""
    pytest.importorskip("warpmpc")
    config = MPCStudentConfig(
        weight_mode=WEIGHT_MODE_SUM, weight_sum=40.0,
        slack_price_mode=SLACK_PRICE_SUM, **TUBE_SETTINGS,
    )
    assert config.slack_quadratic_fraction == 0.0


LEARNED_NODES = {
    "weight_mode": WEIGHT_MODE_SUM,
    "weight_sum": 40.0,
    "slack_price_mode": SLACK_PRICE_SUM,
    "node_weight_mode": NODE_WEIGHT_LEARNED,
}
NODE_WEIGHT_MIN = 0.25


@pytest.mark.parametrize("fields", [
    {"node_weight_mode": "softmax"},
    {"node_weight_mode": NODE_WEIGHT_LEARNED,
     "weight_mode": WEIGHT_MODE_SUM},
    {**LEARNED_NODES, "constraint_type": "plane", "constraint_count": 2},
    {**LEARNED_NODES, "node_weight_min": 0.0},
    {**LEARNED_NODES, "node_weight_min": 1.0},
])
def test_invalid_node_weight_settings_are_rejected(
        fields: dict,
    ) -> None:
    """Unknown modes, learned node weights without the sum slack price or
    with planes, and a minimum outside (0, 1) raise."""
    pytest.importorskip("warpmpc")
    settings = dict(TUBE_SETTINGS)
    settings.update(fields)
    with pytest.raises(ValueError):
        MPCStudentConfig(**settings)


@needs_teacher
def test_learned_node_weights_are_bounded_and_shift_free(
        teacher: Teacher,
    ) -> None:
    """kappa stays in [min, 2 - min], is 1 for equal logits and ignores a
    common shift of the logits; node k's weights scale with kappa_k and
    the slack price is S * kappa."""
    pytest.importorskip("warpmpc")
    student = fresh_mpc_student(
        teacher, "cuda", **TUBE_SETTINGS, **LEARNED_NODES
    )
    obs = moving_obs(teacher)[:8]
    with torch.no_grad():
        raw = student.head_output(obs)
        raw[..., -1] = 0.0
        assert torch.allclose(
            student.node_weights(raw), torch.ones(HORIZON_STEPS, 8,
                                                  device=raw.device)
        )
        raw[..., -1] = 50.0 * torch.linspace(
            -1, 1, HORIZON_STEPS, device=raw.device
        )
        kappa = student.node_weights(raw)
        assert float(kappa.min()) >= NODE_WEIGHT_MIN - 1e-6
        assert float(kappa.max()) <= 2 - NODE_WEIGHT_MIN + 1e-6
        shifted = raw.clone()
        shifted[..., -1] += 3.0
        assert torch.allclose(student.node_weights(shifted), kappa)
        price = student.node_slack_price(raw)
        assert torch.allclose(price, 40.0 * kappa)
        x0 = torch.zeros(8, 5, device=raw.device)
        q_kappa, _ = student.cost_from_head_output(raw, x0)
        raw[..., -1] = 0.0
        q_one, _ = student.cost_from_head_output(raw, x0)
    assert torch.allclose(q_kappa, kappa[..., None] * q_one, atol=1e-6)


@needs_teacher
def test_learned_node_weights_at_one_match_the_fixed_nodes(
        teacher: Teacher,
    ) -> None:
    """With kappa = 1 on every node the learned-node student solves the
    same problem as the fixed S = 40 sum-price student (up to float
    rounding: the price is a solver parameter instead of a constant)."""
    pytest.importorskip("warpmpc")
    fixed_fields = {k: v for k, v in LEARNED_NODES.items()
                    if k != "node_weight_mode"}
    fixed = fresh_mpc_student(teacher, "cuda", **TUBE_SETTINGS, **fixed_fields)
    learned = fresh_mpc_student(
        teacher, "cuda", **TUBE_SETTINGS, **LEARNED_NODES
    )
    learned.backbone.load_state_dict(fixed.backbone.state_dict())
    per_step = learned.outputs_per_step
    with torch.no_grad():
        weight = learned.head.weight.view(HORIZON_STEPS, per_step, -1)
        bias = learned.head.bias.view(HORIZON_STEPS, per_step)
        weight[:, :-1] = fixed.head.weight.view(HORIZON_STEPS, per_step - 1, -1)
        bias[:, :-1] = fixed.head.bias.view(HORIZON_STEPS, per_step - 1)
        weight[:, -1] = 0.0
        bias[:, -1] = 0.0
        obs = moving_obs(teacher)[:64]
        expected = fixed.solve_controls(obs, 1, None)
        actual = learned.solve_controls(obs, 1, None)
    assert torch.allclose(expected, actual, atol=1e-4)


def test_edge_side_filter_needs_the_route_anchor() -> None:
    """tube_edge_side_filter without tube_anchor=route raises."""
    pytest.importorskip("warpmpc")
    with pytest.raises(ValueError):
        MPCStudentConfig(**TUBE_SETTINGS, tube_edge_side_filter=True)
