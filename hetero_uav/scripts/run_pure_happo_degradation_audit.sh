#!/usr/bin/env bash
set -e

OUT=${OUT:-outputs/pure_happo_fixed_route_blue_500k_probe}
DEVICE=${DEVICE:-cuda}

echo "[audit] output dir: ${OUT}"
python scripts/audit_training_degradation.py --output-dir "${OUT}"
python scripts/audit_terminal_reward_semantics.py --output-dir "${OUT}"
python scripts/audit_pure_happo_update_invariants.py \
  --device "${DEVICE}" \
  --output-dir outputs/audits/pure_happo_update_invariants

echo "[audit] reports:"
echo "  ${OUT}/degradation_audit/degradation_summary.md"
echo "  ${OUT}/degradation_audit/terminal_reward_semantics.md"
echo "  outputs/audits/pure_happo_update_invariants/summary.json"
