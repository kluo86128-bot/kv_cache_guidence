#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/train_medical_split_u_background.sh GPU_LIST [--epoch-validation]' >&2
  exit 1
fi
gpus="$1"
shift
mkdir -p logs
run_directory="runs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs"
log_file="logs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs.log"
if [[ -e "$run_directory" || -e "$log_file" ]]; then
  echo 'Run/log already exists. Use training_control.py recover --run-root with this run directory.' >&2
  exit 1
fi
set -o noclobber
exec 3> "$log_file"
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u \
  scripts/run_multiscene_experiment.py --scene medical --loss birth_focal_split \
  --gpus "$gpus" --no-epoch-validation "$@" \
  >&3 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: /data/kevin/medical_privacy_prompt/$log_file"
echo "Run: /data/kevin/medical_privacy_prompt/$run_directory"
