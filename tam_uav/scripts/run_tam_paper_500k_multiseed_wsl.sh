#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/mnt/c/Users/HPK/Desktop/train_environment/tam_uav"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TOTAL_STEPS=512000
ROLLOUT_LENGTH=128
EVALUATION_INTERVAL=51200
CHECKPOINT_INTERVAL=102400

SELECTION_ACTION_SEED_BASE=40260717
HOLDOUT_ACTION_SEED_BASE=50260717
EVALUATION_ENVIRONMENT_SEED=30260717
PANEL_SIZE=50

SEEDS=(20260717 20260718 20260719)

CHECKPOINT_STEPS=(
  0
  102400
  204800
  307200
  409600
  512000
)

LOG_DIR="outputs/console_logs_512000"
REPORT_DIR="outputs/tam_paper_v4_2v2_512000_multiseed_holdout_report"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "TAM paper v4 2v2 Vanilla-HAPPO multi-seed run"
echo "Root: ${ROOT}"
echo "Total steps per seed: ${TOTAL_STEPS}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"

python - <<'PY'
import torch
from uav_env.JSBSim.paper.protocol import ENVIRONMENT_FIDELITY_REVISION
from algorithms.happo.vanilla_happo_checkpoint import FORMAT

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("environment revision:", ENVIRONMENT_FIDELITY_REVISION)
print("checkpoint format:", FORMAT)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this run")

if ENVIRONMENT_FIDELITY_REVISION != "published_rules_simplified_v4":
    raise SystemExit("Unexpected environment revision")

if FORMAT != "tam_paper_heterogeneous_reward_vanilla_happo_v4":
    raise SystemExit("Unexpected checkpoint format")

print("gpu:", torch.cuda.get_device_name(0))
PY

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="outputs/tam_paper_v4_learnability_2v2_seed${SEED}_${TOTAL_STEPS}"
    SELECTION_DIR="${RUN_DIR}/panel_selection_500k"
    HOLDOUT_DIR="${RUN_DIR}/panel_holdout_500k"

    if [[ -e "${RUN_DIR}" ]]; then
        echo "ERROR: output directory already exists:"
        echo "  ${RUN_DIR}"
        echo "Refusing to overwrite or create a timestamped replacement."
        exit 2
    fi

    echo
    echo "============================================================"
    echo "TRAINING SEED ${SEED}"
    echo "OUTPUT ${RUN_DIR}"
    echo "============================================================"

    python -u scripts/train_tam_paper_vanilla_happo.py \
      --scenario 2v2 \
      --seed "${SEED}" \
      --total-environment-steps "${TOTAL_STEPS}" \
      --rollout-length "${ROLLOUT_LENGTH}" \
      --actor-lr 5e-4 \
      --critic-lr 5e-4 \
      --ppo-epochs 2 \
      --minibatch-size 256 \
      --clip-param 0.2 \
      --value-loss-coef 0.5 \
      --entropy-coef 0.01 \
      --max-gradient-norm 10 \
      --gamma 0.99 \
      --gae-lambda 0.95 \
      --evaluation-interval "${EVALUATION_INTERVAL}" \
      --evaluation-episodes 1 \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --actor-sharing independent \
      --hidden-dim 128 \
      --evaluation-seed-base "${EVALUATION_ENVIRONMENT_SEED}" \
      --device cuda \
      --output-directory "${RUN_DIR}" \
      2>&1 | tee "${LOG_DIR}/train_seed_${SEED}.log"

    echo
    echo "============================================================"
    echo "SELECTION PANEL SEED ${SEED}"
    echo "============================================================"

    python -u scripts/evaluate_tam_paper_checkpoint_panel.py \
      --run-directory "${RUN_DIR}" \
      --checkpoint-steps "${CHECKPOINT_STEPS[@]}" \
      --panel-size "${PANEL_SIZE}" \
      --environment-seed "${EVALUATION_ENVIRONMENT_SEED}" \
      --policy-action-seed-base "${SELECTION_ACTION_SEED_BASE}" \
      --bootstrap-samples 10000 \
      --device cuda \
      --output "${SELECTION_DIR}" \
      2>&1 | tee "${LOG_DIR}/selection_panel_seed_${SEED}.log"

    echo
    echo "============================================================"
    echo "INDEPENDENT HOLDOUT PANEL SEED ${SEED}"
    echo "============================================================"

    python -u scripts/evaluate_tam_paper_checkpoint_panel.py \
      --run-directory "${RUN_DIR}" \
      --checkpoint-steps "${CHECKPOINT_STEPS[@]}" \
      --panel-size "${PANEL_SIZE}" \
      --environment-seed "${EVALUATION_ENVIRONMENT_SEED}" \
      --policy-action-seed-base "${HOLDOUT_ACTION_SEED_BASE}" \
      --bootstrap-samples 10000 \
      --device cuda \
      --output "${HOLDOUT_DIR}" \
      2>&1 | tee "${LOG_DIR}/holdout_panel_seed_${SEED}.log"

    echo
    echo "Completed seed ${SEED}"
done

RUN_1="${ROOT}/outputs/tam_paper_v4_learnability_2v2_seed20260717_${TOTAL_STEPS}"
RUN_2="${ROOT}/outputs/tam_paper_v4_learnability_2v2_seed20260718_${TOTAL_STEPS}"
RUN_3="${ROOT}/outputs/tam_paper_v4_learnability_2v2_seed20260719_${TOTAL_STEPS}"

echo
echo "============================================================"
echo "GENERATING MULTI-SEED HOLDOUT REPORT"
echo "============================================================"

python -u scripts/report_tam_paper_500k_holdout.py \
  --run-directories \
    "${RUN_1}" \
    "${RUN_2}" \
    "${RUN_3}" \
  --selection-subdir panel_selection_500k \
  --holdout-subdir panel_holdout_500k \
  --expected-steps "${TOTAL_STEPS}" \
  --output-directory "${ROOT}/${REPORT_DIR}" \
  2>&1 | tee "${LOG_DIR}/multiseed_holdout_report.log"

echo
echo "============================================================"
echo "ALL RUNS COMPLETED"
echo "Report directory:"
echo "  ${ROOT}/${REPORT_DIR}"
echo "Main report:"
echo "  ${ROOT}/${REPORT_DIR}/holdout_multiseed_report.md"
echo "============================================================"
