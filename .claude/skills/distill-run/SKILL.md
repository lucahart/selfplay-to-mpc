---
name: distill-run
description: >-
  Launch, evaluate and plot a distillation run (DAgger, MLP or DiffMPC
  student) with scripts/distill/train.py, evaluate_checkpoint.py and
  plot_run.py, or reproduce the paper's runs with scripts/paper/. Use when
  asked to start, evaluate, compare or plot a distillation run, or to pick
  DistillConfig / MLPStudentConfig / MPCStudentConfig settings.
---

# Distillation run

`docs/method.md` explains the students, the MPC problem and the DAgger
loop; `docs/setup.md` the WarpMPC install and the test data.

## Settings

- `DistillConfig` (`pufferlib/distill/loop.py`), `MLPStudentConfig` and
  `MPCStudentConfig` (`pufferlib/distill/students.py`) are the source of
  truth for keys and defaults; the `scripts/distill/train.py` docstring
  lists the script-only keys.
- The paper's settings for every student and teacher are in
  `scripts/paper/settings.sh`. Start from them.
- The teacher defaults to `TEACHER_CHECKPOINT` in
  `pufferlib/distill/env.py`; `teacher_checkpoint=` and `policy.*`
  select another one (see `teacher_settings` in `settings.sh`).
- DiffMPC students need WarpMPC and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; otherwise JAX takes most of the
  GPU memory.
- Compare students only at the same learning rate, data reuse and label
  budget.

## Launch

```bash
source .venv/bin/activate
python scripts/distill/train.py student=mpc seed=1 \
  output_dir=experiments/distill/<run> <overrides> \
  > experiments/distill/<run>/train.out 2>&1
```

The run folder gets `log.jsonl` (per iteration), `eval.jsonl` (per
evaluation), `student_<labels>.pt` per evaluation and `student.pt` at the
end. `stop.txt` in the run folder ends the run after the current
iteration; `resume_checkpoint=<pt>` continues from a saved checkpoint.

For the paper's runs use `scripts/paper/train_student.sh <small|large>
<mlp|unconstrained|constrained> <seed>`, which also applies the stopping
rule of `scripts/paper/plateau.py`.

## Evaluate and plot

```bash
python scripts/distill/evaluate_checkpoint.py \
  checkpoint=experiments/distill/<run>/student_<labels>.pt student=mpc \
  [mpc.<field>=<value> ...] evaluator=validation_gigaflow num_scenarios=1000
python scripts/distill/plot_run.py run_dir=experiments/distill/<run>
```

The student settings must match the checkpoint (the state dict loads
strictly). The evaluators are deterministic per checkpoint. The paper's
test sets are `validation_gigaflow` (1000 CARLA scenarios),
`nuplan_multi_replan` (3645 nuPlan scenarios) and `womd_multi` (1000
Waymo scenarios); `scripts/paper/evaluate_student.sh` runs all three, and
`scripts/paper/make_table.py` builds Table 1.

## Before finishing

Run `python -m pytest tests/distill -q` after code changes.
