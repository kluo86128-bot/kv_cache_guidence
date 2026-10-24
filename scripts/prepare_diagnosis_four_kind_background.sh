#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv prepare-negatives \
  --config /data/kevin/medical_privacy_prompt/configs/experiment_v3_diagnosis_contrastive.json \
  --output-directory /data/kevin/medical_privacy_prompt/runs/diagnosis_four_kind_preparation_v3_top5 \
  > /data/kevin/medical_privacy_prompt/logs/diagnosis_four_kind_preparation_v3_top5.log 2>&1 < /dev/null &
echo $!
