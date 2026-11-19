#!/bin/bash
# Evaluate a teacher on the paper's three test sets. check_teacher.py runs
# the teacher with its own discrete actions (the reported number) and with
# the decoded continuous labels the students learn from.
# Usage: scripts/paper/evaluate_teacher.sh <small|large>
# Output: $OUTPUT_ROOT/teacher_<size>/<evaluator>.out
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source scripts/paper/settings.sh
SIZE=$1
OUT=$OUTPUT_ROOT/teacher_$SIZE
mkdir -p "$OUT"

# teacher_checkpoint= is check_teacher.py's checkpoint=
teacher=$(teacher_settings "$SIZE")
teacher=${teacher/teacher_checkpoint=/checkpoint=}
for test_set in $PAPER_TEST_SETS; do
  evaluator=${test_set%%:*}
  scenarios=${test_set##*:}
  python scripts/distill/check_teacher.py $teacher \
    evaluator="$evaluator" num_scenarios="$scenarios" \
    > "$OUT/$evaluator.out" 2>&1
done
