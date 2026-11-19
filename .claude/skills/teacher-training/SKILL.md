---
name: teacher-training
description: >-
  Train, resume and evaluate the PufferDrive PPO teacher (puffer train /
  eval, classic dynamics) and check it before distillation. Use when asked
  to train or fine-tune a self-play policy, run an evaluator on a teacher
  checkpoint, or change teacher config values.
---

# Teacher training

- Parameters and defaults: `pufferlib/config/puffer_drive.yaml`.
- The distillation teachers are `TEACHER_CHECKPOINT` (4x1024) in
  `pufferlib/distill/env.py`, and the 4x512 teacher of
  `scripts/paper/settings.sh`. Their network is `TEACHER_POLICY_OVERRIDES`
  (4x1024) or the `policy.*` overrides in `settings.sh` (4x512), and their
  environment is `TEACHER_ENV_OVERRIDES`; distillation uses the same
  environment the teacher was trained in.
- `docs/evaluation.md` describes the evaluators.

## Rules

- Never train with `env.collision_behavior=0` or `env.offroad_behavior=0`
  (policies collapse to standing still).
- `load_model_path` also restores the optimizer and step count from the
  run folder; copy a checkpoint elsewhere for a fresh start.
  `train.total_timesteps` is an absolute target on resume.
- After any `.c`/`.h` change: `python setup.py build_ext --inplace --force`.

## Checking a teacher

A teacher must pass `python scripts/distill/check_teacher.py
num_scenarios=1000` before it labels anything: its discrete actions and
the decoded continuous labels the students learn from must drive equally
well. `scripts/paper/evaluate_teacher.sh` runs the check on the paper's
three test sets.
