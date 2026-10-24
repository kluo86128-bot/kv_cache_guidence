#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
if [[ -e /data/kevin/medical_privacy_prompt/runs/test_multiscene_epoch25_latest || -e /data/kevin/medical_privacy_prompt/logs/test_multiscene_epoch25_latest.log ]]; then
  echo 'Interim output or log already exists; preserve it and choose a new output/log path.' >&2
  exit 1
fi
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv evaluate \
  --config /data/kevin/medical_privacy_prompt/configs/evaluation_multiscene_v1.json \
  --gpus "${1:-0,1}" \
  --condition soft --split test --interim-test \
  --checkpoint /data/kevin/medical_privacy_prompt/runs/soft_multiscene_v1_seed17_100epochs/latest.pt \
  --output-directory /data/kevin/medical_privacy_prompt/runs/test_multiscene_epoch25_latest \
  > /data/kevin/medical_privacy_prompt/logs/test_multiscene_epoch25_latest.log 2>&1 < /dev/null &
echo $!
