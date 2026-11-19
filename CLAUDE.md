# Self-play RL to constrained control: DiffMPC students distilled with DAgger
C simulation engine (PufferDrive) + Python/PyTorch training. A self-play PPO teacher (classic bicycle, discrete actions, MLP) is distilled with DAgger into an MLP student and two differentiable-MPC students (unconstrained, and constrained by road planes and agent ellipses).
Activate venv before `python`/`puffer`: `source .venv/bin/activate`

## Start here
`docs/method.md` explains how the code implements the paper; `docs/setup.md` covers dependencies and data.

## Structure
- `pufferlib/ocean/drive/`: `drive.h/c` (sim core), `binding.c` (C-ext), `drive.py` (Gym wrapper), `visualize.c`
- `pufferlib/ocean/`: `env_binding.h` (C env utils), `torch.py` (NN, incl. the `Drive` teacher policy)
- `pufferlib/distill/`: distillation library: `loop.py` (DAgger loop, `DistillConfig`), `students.py` (MLP, MPC), `bicycle.py` (dynamics), `warpmpc_solver.py` (WarpMPC problem), `tube.py` and `obstacles.py` (constraints), `buffer.py`, `teacher.py`, `env.py`, `vec_env.py` (`VecDrive`: env worker processes)
- `scripts/distill/`: entry points: `train.py`, `evaluate_checkpoint.py`, `plot_run.py`, `check_teacher.py`
- `scripts/paper/`: the paper's settings and runs (`settings.sh`, `train_student.sh`, `evaluate_student.sh`, `evaluate_teacher.sh`, `make_table.py`, `run_all.sh`)
- `tests/distill/`: distillation tests (`pytest tests/distill`)
- `pufferlib/config/`: monolithic Hydra YAML — `puffer_drive.yaml` (PufferDrive)
- `experiments/`: run outputs, **gitignored**; `checkpoints/`: teacher checkpoints, **gitignored**

## Commands
- **Rebuild C (mandatory after .c/.h change):** `python setup.py build_ext --inplace --force`
- **Train teacher:** `puffer train puffer_drive [train.learning_rate=0.001 env.num_agents=512]` (see the `teacher-training` skill)
- **Distill:** `python scripts/distill/train.py student=mlp|mpc key=value ...` (see the `distill-run` skill)
- **Reproduce the paper:** `scripts/paper/run_all.sh`
- **Tests:** `python -m pytest tests/distill -q`

## Skills (`.claude/skills/`)
- `distill-run`: launch, evaluate and plot a distillation run.
- `teacher-training`: train and evaluate the PPO teacher.

## Coding Standards
- **Naming:** explicit (`active_agent_count`, `closest_lane_idx`); never `n/tmp/val/foo` except tiny local math. Keep units in names: `_seconds/_meters/_mps/_idx/_count`.
- **Helpers:** add a function only for a major sim concept (`move_expert`, `compute_rewards`). No one-off wrappers hiding 2 lines.
- **One function mutates one subsystem.** Don't update dynamics+rewards+metrics+logs together.
- **Control flow:** max 2 nesting levels (flatten via `continue`/`return`; deeper → extract a major helper). No recursion. No function pointers (use `if`/`switch` on mode constants).
- **Data:** flat struct arrays + integer indices. No nested ownership / `**` unless the map/grid truly needs it.
- **Loops bounded:** iterate known counts; every `while` needs an explicit max-iteration counter.
- **Named constants for all magic values** (`GRID_CELL_SIZE`, `DEFAULT_TTC`); never raw `15.0f`/`64` in logic. Centralize enum-like mode constants near the top.
- **Comments explain invariants, not syntax** (good: "reward flags mutually exclusive"). Code readable without comments.
- **Check non-void returns when correctness depends on it** (loading, init, map/grid build, spawning).
- **Zero compiler warnings** — warnings are bugs until proven otherwise.
- **Perf:** zero Python overhead in C hot paths; no malloc/free in `c_step`, obs gen, collision, reward.
- **Style:** match surrounding code; touch only what the request needs; flag dead/broken adjacent code in text, don't edit it.

## Trust Boundaries
**External (untrusted): configs, CLI overrides, `*.bin` maps, Python params, dataset/scenario metadata.**
Validate aggressively *before* init/reset/step; invalid data is unrecoverable → abort env creation with explicit error. Check: magic bytes/version (reject truncated/unknown/trailing garbage); counts before alloc (reject negative/excessive, guard mul overflow, verify byte size before bulk read); ranges (reject bad enums/indices, negative length/width/dt, NaN/Inf); topology (lanes/routes/traffic-controls reference existing elements; trajectory length matches scenario). Match every `free`; clean up on partial-load failure.

**Internal (trusted): hot paths — `c_step`, dynamics, rewards, metrics, collision, observations, grid queries, movement.**
Assume invariants hold; optimize. No redundant null checks, range rechecks, silent clamping, default-for-impossible-state, or catch-all branches. Use `assert(idx < env->num_agents)` for dev debugging only — never as runtime recovery.

## Guardrails
- **Fail-fast:** abort on impossible state, never recover/pad. Good: `if (len <= 0) return ERROR;` / `assert(size == expected);` — Bad: `len = 1;` / `while(size<expected) obs[size++]=0;`
- **Determinism:** input+config+seed → identical trajectories. Stable init, deterministic iteration order, one explicit RNG path. No unordered containers, hidden randomness, or wall-clock in sim logic.

## Issue and PR Guidelines
- Add a PR description with What/Why/Notes structure
