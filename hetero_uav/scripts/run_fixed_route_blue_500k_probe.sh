#!/usr/bin/env bash
set -e

OUT=outputs/brma_recurrent_masked_fixed_route_blue_500k_probe
rm -rf "$OUT"
mkdir -p "$OUT"

python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir "$OUT" \
  --total-env-steps 500000 \
  --rollout-length 256 \
  --num-envs 4 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch brma_recurrent_masked \
  --opponent-policy fixed_route \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 25000 \
  --keep-checkpoints 30 \
  --enable-rich-logging \
  --rich-log-dir "$OUT/rich_logs" \
  --heartbeat-log "$OUT/heartbeat.log" \
  --heartbeat-every-steps 50
