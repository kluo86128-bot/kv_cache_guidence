#!/usr/bin/env bash
set -euo pipefail
project_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_directory"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/train_hr_split_u_background.sh GPU_LIST [--epoch-validation]' >&2
  exit 1
fi
gpus="$1"
shift
python_binary="${MEDICAL_PRIVACY_PYTHON:-$(command -v python)}"
run_directory="runs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs"
log_file="logs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs.log"
if [[ -e "$run_directory" || -e "$log_file" ]]; then
  echo 'Run/log already exists. Use training_control.py recover --run-root with this run directory.' >&2
  exit 1
fi
# Read-only CPU preflight catches missing/stale new data or token caches before
# launching a detached CUDA job. No model weights are loaded here.
"$python_binary" - <<'PY'
from pathlib import Path
from medical_kv.config import load_config
from medical_kv.data import dataset_identity
cfg = load_config('configs/experiment_v14_hr_split_u.json')
ev = load_config('configs/evaluation_v10_hr_split_u.json')
if {k for k in cfg if cfg[k] != ev[k]} != {'output_directory'}:
    raise ValueError('HR training/evaluation settings differ beyond output_directory')
if not Path(cfg['model_name_or_path']).is_dir():
    raise FileNotFoundError('Set model_name_or_path in BOTH new HR configs to the local model directory: ' + cfg['model_name_or_path'])
identity = dataset_identity(cfg)
print('HR preflight passed:', identity['dataset_id'], '| scene:', cfg['scene'], '| U:', cfg['prompt_length'])
PY
mkdir -p logs
set -o noclobber
exec 3> "$log_file"
nohup setsid "$python_binary" -u scripts/run_multiscene_experiment.py \
  --scene hr --loss field_focal_split --gpus "$gpus" \
  --no-epoch-validation "$@" >&3 2>&1 < /dev/null &
echo "PID: $!"
echo "Python: $python_binary"
echo "Log: $project_directory/$log_file"
echo "Run: $project_directory/$run_directory"
