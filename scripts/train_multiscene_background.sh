#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u \
  /data/kevin/medical_privacy_prompt/scripts/run_multiscene_experiment.py \
  --gpus "${1:-0,1}" \
  "${@:2}" \
  > /data/kevin/medical_privacy_prompt/logs/soft_multiscene_v1_seed17_100epochs.log 2>&1 < /dev/null &
echo $!
