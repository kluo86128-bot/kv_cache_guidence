#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
case "${1:-}" in medical|finance|hr) ;; *) echo 'Usage: bash train_scene_background.sh medical|finance|hr [0,1] [--epoch-validation]' >&2; exit 1 ;; esac
mkdir -p /data/kevin/medical_privacy_prompt/logs
if [[ -e "/data/kevin/medical_privacy_prompt/logs/soft_${1}_v2_clean_seed17_100epochs.log" || -e "/data/kevin/medical_privacy_prompt/runs/soft_${1}_v2_clean_seed17_100epochs/latest.pt" ]]; then
  echo 'Run/log already exists. Use training_control.py recover to resume it.' >&2
  exit 1
fi
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u \
  /data/kevin/medical_privacy_prompt/scripts/run_multiscene_experiment.py \
  --scene "$1" --gpus "${2:-0,1}" --no-epoch-validation "${@:3}" \
  > "/data/kevin/medical_privacy_prompt/logs/soft_${1}_v2_clean_seed17_100epochs.log" 2>&1 < /dev/null &
echo $!
