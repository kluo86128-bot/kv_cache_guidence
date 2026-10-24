#!/usr/bin/env bash
set -euo pipefail
cd /data/kevin/medical_privacy_prompt
mkdir -p /data/kevin/medical_privacy_prompt/logs
nohup setsid /home/kevin/.conda/envs/medical_privacy/bin/python -u \
  /data/kevin/medical_privacy_prompt/scripts/prepare_scene_input_caches.py --rebuild \
  > /data/kevin/medical_privacy_prompt/logs/prepare_scene_input_caches_v2_clean.log 2>&1 < /dev/null &
echo $!
