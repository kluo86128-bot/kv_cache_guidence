#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p logs
run_directory="runs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs"
log_file="logs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs.log"
if [[ -e "$run_directory" || -e "$log_file" ]]; then
  echo 'Run/log already exists. Use training_control.py recover --run-root with this run directory.' >&2
  exit 1
fi
set -o noclobber
exec 3> "$log_file"
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u \
  scripts/run_multiscene_experiment.py --scene medical --loss birth_focal_pcgrad \
  --gpus "${1:-0,1}" --no-epoch-validation "${@:2}" \
  >&3 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: /data/kevin/medical_privacy_prompt/$log_file"
echo "Run: /data/kevin/medical_privacy_prompt/$run_directory"
