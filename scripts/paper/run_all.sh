#!/bin/bash
# Every run of the paper, one after the other: both teachers' evaluations,
# then 2 sizes x 3 students x 9 seeds of DAgger and their evaluations, then
# Table 1. On one RTX 5090 a DiffMPC run takes 2 to 5 hours.
# Usage: scripts/paper/run_all.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source scripts/paper/settings.sh

for size in $PAPER_SIZES; do
  scripts/paper/evaluate_teacher.sh "$size"
  for student in $PAPER_STUDENTS; do
    for seed in $PAPER_SEEDS; do
      scripts/paper/train_student.sh "$size" "$student" "$seed"
      scripts/paper/evaluate_student.sh "$size" "$student" "$seed"
    done
  done
done
python scripts/paper/make_table.py "$OUTPUT_ROOT"
