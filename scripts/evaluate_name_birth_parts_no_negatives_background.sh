#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv evaluate \
  --config /data/kevin/medical_privacy_prompt/configs/evaluation_name_birth_parts_overlap10.json \
  --gpus "${1:-0,1}" \
  --condition soft --split test \
  --checkpoint /data/kevin/medical_privacy_prompt/runs/soft_name_birth_parts_no_negatives_seed17_100epochs/best.pt \
  --output-directory /data/kevin/medical_privacy_prompt/runs/test_name_birth_parts_no_negatives_overlap10_best \
  > /data/kevin/medical_privacy_prompt/logs/test_name_birth_parts_no_negatives_overlap10_best.log 2>&1 < /dev/null &
echo $!
