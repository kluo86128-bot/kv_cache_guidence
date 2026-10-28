#!/usr/bin/env bash
set -euo pipefail
project_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_directory"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/train_finance_pcgrad_background.sh GPU_LIST [--epoch-validation]' >&2
  exit 1
fi
gpus="$1"
shift
python_binary="${MEDICAL_PRIVACY_PYTHON:-$(command -v python)}"
mkdir -p logs
run_directory="runs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs"
log_file="logs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs.log"
if [[ -e "$run_directory" || -e "$log_file" ]]; then
  echo 'Run/log already exists. Use training_control.py recover --run-root with this run directory.' >&2
  exit 1
fi
set -o noclobber
exec 3> "$log_file"
nohup setsid "$python_binary" -u scripts/run_multiscene_experiment.py \
  --scene finance --loss field_focal_pcgrad --gpus "$gpus" \
  --no-epoch-validation "$@" >&3 2>&1 < /dev/null &
echo "PID: $!"
echo "Python: $python_binary"
echo "Log: $project_directory/$log_file"
echo "Run: $project_directory/$run_directory"
