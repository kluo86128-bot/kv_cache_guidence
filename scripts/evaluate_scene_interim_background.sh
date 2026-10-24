#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
case "${1:-}" in medical|finance|hr) ;; *) echo 'Usage: bash evaluate_scene_interim_background.sh medical|finance|hr [0,1]' >&2; exit 1 ;; esac
mkdir -p /data/kevin/medical_privacy_prompt/logs
evaluation_stamp="$(date +%Y%m%d_%H%M%S_%N)"
evaluation_output="/data/kevin/medical_privacy_prompt/runs/test_${1}_v2_clean_interim_${evaluation_stamp}"
evaluation_log="/data/kevin/medical_privacy_prompt/logs/test_${1}_v2_clean_interim_${evaluation_stamp}.log"
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv evaluate \
  --config "/data/kevin/medical_privacy_prompt/configs/evaluation_v2_${1}.json" \
  --gpus "${2:-0,1}" --condition soft --split test --interim-test \
  --checkpoint "/data/kevin/medical_privacy_prompt/runs/soft_${1}_v2_clean_seed17_100epochs/latest.pt" \
  --output-directory "$evaluation_output" \
  > "$evaluation_log" 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: $evaluation_log"
echo "Results: $evaluation_output"
