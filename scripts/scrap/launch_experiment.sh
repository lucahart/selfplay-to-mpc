#!/bin/bash
# Usage: scripts/scrap/launch_experiment.sh <experiment_dir> <command...>
# Saves the exact command as <experiment_dir>/command_<git_id>.sh, then runs it with output in <experiment_dir>/train.log.
set -euo pipefail
repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo_root"
experiment_dir=$1
shift
mkdir -p "$experiment_dir"
git_id=$(git rev-parse --short HEAD)
command_file="$experiment_dir/command_${git_id}.sh"
{
  echo "#!/bin/bash"
  echo "cd $repo_root && source .venv/bin/activate"
  printf '%q ' "$@"
  echo
} > "$command_file"
chmod +x "$command_file"
source .venv/bin/activate
"$@" 2>&1 | tee "$experiment_dir/train.log"
