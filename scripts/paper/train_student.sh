#!/bin/bash
# Distill one student with DAgger as in the paper, until the stopping rule
# of plateau.py ends the run.
# Usage: scripts/paper/train_student.sh <small|large> <mlp|unconstrained|constrained> <seed>
# Output: $OUTPUT_ROOT/<size>_<student>_s<seed>/ (default experiments/paper)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source scripts/paper/settings.sh
SIZE=$1; STUDENT=$2; SEED=$3
RUN=$(run_dir "$SIZE" "$STUDENT" "$SEED")
# A watcher process checks for new evaluations this often [s]
POLL_SECONDS=60
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$RUN"
git rev-parse HEAD > "$RUN/git_hash.txt"

python scripts/distill/train.py output_dir="$RUN" seed="$SEED" \
  $LOOP_SETTINGS $(training_settings "$STUDENT") \
  $(teacher_settings "$SIZE") $(student_settings "$STUDENT") \
  > "$RUN/train.out" 2>&1 &
train_pid=$!

# stop.txt ends the run cleanly after the current iteration
seen=0
while kill -0 $train_pid 2>/dev/null; do
  sleep $POLL_SECONDS
  [ -f "$RUN/eval.jsonl" ] || continue
  count=$(wc -l < "$RUN/eval.jsonl")
  [ "$count" -gt "$seen" ] || continue
  seen=$count
  if [ "$(python scripts/paper/plateau.py stop "$RUN/eval.jsonl")" = stop ]; then
    echo "plateau after $count evaluations" > "$RUN/stop.txt"
  fi
done
wait $train_pid
echo "$RUN: best checkpoint at $(python scripts/paper/plateau.py best "$RUN/eval.jsonl") labels"
