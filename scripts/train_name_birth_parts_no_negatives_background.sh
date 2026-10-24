#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv train-soft \
  --config /data/kevin/medical_privacy_prompt/configs/experiment_v4_name_birth_parts_no_negatives.json \
  --gpus "${1:-0,1}" \
  --output-directory /data/kevin/medical_privacy_prompt/runs/soft_name_birth_parts_no_negatives_seed17_100epochs \
  > /data/kevin/medical_privacy_prompt/logs/soft_name_birth_parts_no_negatives_seed17_100epochs.log 2>&1 < /dev/null &
echo $!
