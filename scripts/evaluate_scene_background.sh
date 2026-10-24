#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
case "${1:-}" in medical|finance|hr) ;; *) echo 'Usage: bash evaluate_scene_background.sh medical|finance|hr [0,1]' >&2; exit 1 ;; esac
mkdir -p /data/kevin/medical_privacy_prompt/logs
if [[ -e "/data/kevin/medical_privacy_prompt/logs/test_${1}_v2_clean_best_manual.log" || -e "/data/kevin/medical_privacy_prompt/runs/test_${1}_v2_clean_best_manual" ]]; then
  echo 'Manual evaluation output/log already exists; preserve it and choose another path.' >&2
  exit 1
fi
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u -m medical_kv evaluate \
  --config "/data/kevin/medical_privacy_prompt/configs/evaluation_v2_${1}.json" \
  --gpus "${2:-0,1}" --condition soft --split test \
  --checkpoint "/data/kevin/medical_privacy_prompt/runs/soft_${1}_v2_clean_seed17_100epochs/best.pt" \
  --output-directory "/data/kevin/medical_privacy_prompt/runs/test_${1}_v2_clean_best_manual" \
  > "/data/kevin/medical_privacy_prompt/logs/test_${1}_v2_clean_best_manual.log" 2>&1 < /dev/null &
echo $!
