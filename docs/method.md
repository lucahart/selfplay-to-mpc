# Method and code map

How the code in this repository implements the paper: the self-play teacher,
the three students, the constrained MPC problem and the DAgger loop.

## 1. Teacher

The teacher is the PufferDrive PPO policy (`Drive` in
`pufferlib/ocean/torch.py`): per-entity encoders (ego, partners, lanes, road
boundaries, traffic controls, context), an MLP backbone and a discrete action
head over acceleration and steering rate. Every car in the scene is driven by
the same policy (self-play). It is trained with `puffer train puffer_drive`
on CARLA maps with a staged reward curriculum and fine-tuned on a mix of
CARLA and nuPlan maps. The paper uses two teachers: a small one (4x512
backbone) and a large one (4x1024 backbone).

`pufferlib/distill/teacher.py` freezes a teacher checkpoint and decodes its
most likely discrete action into a continuous control label (acceleration,
steering rate), which is what the students regress on.
`scripts/distill/check_teacher.py` confirms that the teacher drives as well
through these continuous labels as through its own discrete actions.

## 2. Students

All students copy the teacher's encoders and backbone and differ only in the
head (`pufferlib/distill/students.py`).

| Student | Head | Config |
|---|---|---|
| MLP | linear layer to the two controls, `tanh` into the control box | `student=mlp` |
| Unconstrained DiffMPC | MPC adapter (linear layer to targets and cost weights) + DiffMPC with a control box only | `student=mpc mpc.solver=warpmpc` |
| Constrained DiffMPC | MPC adapter (targets, cost weights, node weights) + DiffMPC with road planes and agent ellipses | `student=mpc mpc.solver=warpmpc mpc.constraint_type=tube ...` |

The exact settings of each student and teacher are in
`scripts/paper/settings.sh`.

## 3. The MPC problem

State $x = (p_x, p_y, \psi, v, \delta)$ in the ego frame, control
$u = (a, \omega)$ (acceleration, steering rate), $\Delta t = 0.1$ s,
horizon $T = 10$ nodes (one second). The dynamics $f$ are PufferDrive's
classic kinematic bicycle with a per-agent wheelbase, matched step for step to
the C simulator (`pufferlib/distill/bicycle.py`, tested in
`tests/distill/test_bicycle.py`). The student solves

$$
\min_{u_0,\dots,u_{T-1},\,s \ge 0} \ \sum_{k} \kappa_k \Big( \big\| x_k - \bar x_k \big\|^2_{Q_k} + \big\| u_k \big\|^2_{R_k} \Big) + \rho(s)
\quad \text{s.t.} \quad x_0 = x_{\text{ego}},\ x_{k+1} = f(x_k, u_k),\ u_k \in \mathcal U,\ g(x_k) \le s_k .
$$

The adapter predicts the targets $\bar x_k$ (lateral offset, heading, speed,
steering) and the diagonal weights $Q_k, R_k$. For the constrained student the
weights of each node sum to a fixed $S = 40$, the adapter also predicts the
node weights $\kappa_k$, and each slack $s$ is priced relative to $S$
(`mpc.weight_mode=sum`, `mpc.node_weight_mode=learned`,
`mpc.slack_price_mode=sum`). The unconstrained student has $\kappa_k = 1$
and no slacks. $\mathcal U$ is the control box
($\pm 4$ m/s², $\pm 0.75$ rad/s).

**Solver and gradient.** The problem is solved with WarpMPC, a batched SQP on
the GPU whose QPs are solved by an ADMM with Warp kernels (5 SQP iterations of
25 ADMM steps; `pufferlib/distill/warpmpc_solver.py`). The gradient of the
first control $u_0^\star$ with respect to the targets and weights comes from
the adjoint of the last QP, through `jax.vjp`, and is handed back to PyTorch
so that the loss trains the adapter and the backbone end to end.

## 4. Constraints

Both kinds of constraints are computed from the observation at every step;
nothing about them is learned.

**Road boundary planes** (`pufferlib/distill/tube.py`). For every node
$k \ge 1$ the agent's previous plan, shifted into the current ego frame, gives
a reference position. Each reference is anchored on the agent's route, and on
each side a half-plane is placed through the nearest observed road-edge point
that lies on that side of both the route and the reference. The plane keeps
the car's body 0.2 m inside the edge. A side without a nearby edge gets a far
plane (50 m) that never binds.

**Agent ellipses** (`pufferlib/distill/obstacles.py`). The four nearest
partners are predicted forward at constant speed along their heading. Each
partner's box is covered by an ellipse widened by the ego's half-size, and the
ego's centre must stay outside it at every node.

Each constraint row has a slack, so the problem is always feasible; the row
layout is in `pufferlib/distill/constraints.py`.

## 5. Distillation with DAgger

`pufferlib/distill/loop.py` runs the online loop
(`scripts/distill/train.py` is the entry point):

1. 1024 agents drive in the teacher's simulator. For each episode an agent is
   driven by the teacher with probability $\beta$ and by the student otherwise.
   $\beta = 1$ for the first 4M labels, then drops by 0.2 every 2M labels to 0
   (`beta_schedule=dagger`).
2. The teacher labels every state that any agent visits.
3. The loss is the squared error of the student's first control against the
   teacher's label, scaled by the control box.

The MLP student trains from a replay buffer of labels. The DiffMPC students
use the solve that drives an agent to also compute its gradient
(`fresh_gradient=true`), so every state is solved once.

Every 10M labels the student is evaluated on 250 CARLA scenarios. Training
stops when two evaluations in a row improve neither the collision plus
off-road rate nor the goals reached. The checkpoint with the lowest collision plus off-road
rate is then evaluated on the test sets.

## 6. Evaluation

`scripts/distill/evaluate_checkpoint.py` runs a student closed-loop through
PufferDrive's evaluators (`pufferlib/ocean/benchmark/`), in self-play: every
car is controlled by the student. The paper's test sets are:

| Test set | Evaluator | Scenarios |
|---|---|---|
| CARLA | `validation_gigaflow` | 1000 |
| nuPlan | `nuplan_multi_replan` | 3645 |
| Waymo (never used for training) | `womd_multi` | 1000 |

The reported failure rate is the collision rate plus the off-road rate per
vehicle. `docs/evaluation.md` describes the evaluator system.
