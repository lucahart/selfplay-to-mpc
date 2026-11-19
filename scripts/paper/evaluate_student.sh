#!/bin/bash
# Evaluate the chosen checkpoint of a trained student on the paper's three
# test sets (CARLA, nuPlan, Waymo); results go to final_eval.jsonl in the
# run folder.
# Usage: scripts/paper/evaluate_student.sh <small|large> <mlp|unconstrained|constrained> <seed>
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source scripts/paper/settings.sh
SIZE=$1; STUDENT=$2; SEED=$3
RUN=$(run_dir "$SIZE" "$STUDENT" "$SEED")
export XLA_PYTHON_CLIENT_PREALLOCATE=false

labels=$(python scripts/paper/plateau.py best "$RUN/eval.jsonl")
for test_set in $PAPER_TEST_SETS; do
  evaluator=${test_set%%:*}
  scenarios=${test_set##*:}
  python scripts/distill/evaluate_checkpoint.py \
    checkpoint="$RUN/student_$labels.pt" \
    $(teacher_settings "$SIZE") $(student_settings "$STUDENT") \
    evaluator="$evaluator" num_scenarios="$scenarios" \
    > "$RUN/eval_${evaluator}.out" 2>&1
done
