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

ENVIRONMENT_SEED=30260717
SELECTION_ACTION_SEED_BASE=70260717
HOLDOUT_ACTION_SEED_BASE=80260717
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

LOG_DIR="outputs/console_logs_table4_512000_final"
REPORT_DIR="outputs/tam_paper_v4_table4_happo_2v2_512000_multiseed_holdout_report"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "TAM paper v4 final learnability validation"
echo "Scenario: 2v2 nominal"
echo "Algorithm: pure feedforward HAPPO"
echo "Actor hidden sizes: [256, 128]"
echo "Critic hidden sizes: [256, 128]"
echo "Value loss: clipped Huber, delta=10"
echo "Steps per seed: ${TOTAL_STEPS}"
echo "Seeds: ${SEEDS[*]}"
echo "Selection panel size: ${PANEL_SIZE}"
echo "Holdout panel size: ${PANEL_SIZE}"
echo "============================================================"

python - <<'PY'
from pathlib import Path

import torch

from algorithms.happo.vanilla_happo_checkpoint import FORMAT
from scripts.train_tam_paper_vanilla_happo import parse_args
from uav_env.JSBSim.paper.protocol import ENVIRONMENT_FIDELITY_REVISION

args = parse_args([])

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print(
    "gpu:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else None,
)
print("environment revision:", ENVIRONMENT_FIDELITY_REVISION)
print("checkpoint format:", FORMAT)
print("actor hidden sizes:", args.actor_hidden_sizes)
print("critic hidden sizes:", args.critic_hidden_sizes)
print("value loss type:", args.value_loss_type)
print("huber delta:", args.huber_delta)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA不可用")

if ENVIRONMENT_FIDELITY_REVISION != "published_rules_simplified_v4":
    raise SystemExit("ERROR: 环境revision不是冻结v4")

if FORMAT != "tam_paper_heterogeneous_reward_vanilla_happo_v4":
    raise SystemExit("ERROR: checkpoint format异常")

if args.actor_hidden_sizes != [256, 128]:
    raise SystemExit("ERROR: actor隐藏层不是[256,128]")

if args.critic_hidden_sizes != [256, 128]:
    raise SystemExit("ERROR: critic隐藏层不是[256,128]")

if args.value_loss_type != "clipped_huber":
    raise SystemExit("ERROR: value loss不是clipped_huber")

if float(args.huber_delta) != 10.0:
    raise SystemExit("ERROR: Huber delta不是10")

if not Path("scripts/report_tam_paper_500k_holdout.py").is_file():
    raise SystemExit(
        "ERROR: 缺少scripts/report_tam_paper_500k_holdout.py"
    )

print("PRECHECK: PASS")
PY

if [[ -e "${REPORT_DIR}" ]]; then
    echo "ERROR: 报告目录已经存在："
    echo "  ${REPORT_DIR}"
    echo "为避免覆盖旧结果，本次运行中止。"
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="outputs/tam_paper_v4_table4_happo_2v2_seed${SEED}_${TOTAL_STEPS}"
    SELECTION_DIR="${RUN_DIR}/panel_selection_table4_512k"
    HOLDOUT_DIR="${RUN_DIR}/panel_holdout_table4_512k"

    if [[ -e "${RUN_DIR}" ]]; then
        echo "ERROR: 训练目录已经存在："
        echo "  ${RUN_DIR}"
        echo "为避免覆盖或混合实验，本次运行中止。"
        exit 2
    fi

    echo
    echo "============================================================"
    echo "TRAINING SEED ${SEED}"
    echo "OUTPUT: ${RUN_DIR}"
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
      --actor-hidden-sizes 256 128 \
      --critic-hidden-sizes 256 128 \
      --value-loss-type clipped_huber \
      --huber-delta 10 \
      --evaluation-interval "${EVALUATION_INTERVAL}" \
      --evaluation-episodes 1 \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --actor-sharing independent \
      --evaluation-seed-base "${ENVIRONMENT_SEED}" \
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
      --environment-seed "${ENVIRONMENT_SEED}" \
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
      --environment-seed "${ENVIRONMENT_SEED}" \
      --policy-action-seed-base "${HOLDOUT_ACTION_SEED_BASE}" \
      --bootstrap-samples 10000 \
      --device cuda \
      --output "${HOLDOUT_DIR}" \
      2>&1 | tee "${LOG_DIR}/holdout_panel_seed_${SEED}.log"

    echo
    echo "SEED ${SEED} COMPLETED"
done

RUN_1="${ROOT}/outputs/tam_paper_v4_table4_happo_2v2_seed20260717_${TOTAL_STEPS}"
RUN_2="${ROOT}/outputs/tam_paper_v4_table4_happo_2v2_seed20260718_${TOTAL_STEPS}"
RUN_3="${ROOT}/outputs/tam_paper_v4_table4_happo_2v2_seed20260719_${TOTAL_STEPS}"

echo
echo "============================================================"
echo "GENERATING MULTI-SEED INDEPENDENT HOLDOUT REPORT"
echo "============================================================"

python -u scripts/report_tam_paper_500k_holdout.py \
  --run-directories \
    "${RUN_1}" \
    "${RUN_2}" \
    "${RUN_3}" \
  --selection-subdir panel_selection_table4_512k \
  --holdout-subdir panel_holdout_table4_512k \
  --expected-steps "${TOTAL_STEPS}" \
  --output-directory "${ROOT}/${REPORT_DIR}" \
  2>&1 | tee "${LOG_DIR}/multiseed_holdout_report.log"

echo
echo "============================================================"
echo "FINAL COMPACT REPORT"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "outputs/"
    "tam_paper_v4_table4_happo_2v2_512000_"
    "multiseed_holdout_report/"
    "holdout_multiseed_report.json"
)

report = json.loads(path.read_text(encoding="utf-8"))
aggregate = report["aggregate"]

print(json.dumps(aggregate, indent=2))

print("\nPer seed:")
for row in report["seeds"]:
    print(
        f"seed={row['training_seed']} "
        f"runtime={row['runtime_valid']} "
        f"selected_step={row['selected_checkpoint_steps']} "
        f"selected_delta="
        f"{row['holdout_selected_delta_vs_step_0']:.3f} "
        f"selected_ci=["
        f"{row['holdout_selected_delta_ci_low']:.3f},"
        f"{row['holdout_selected_delta_ci_high']:.3f}] "
        f"beats_random={row['holdout_selected_beats_random']} "
        f"combat_gains="
        f"{row['holdout_selected_combat_improvement_count']} "
        f"final_delta="
        f"{row['holdout_final_delta_vs_step_0']:.3f} "
        f"verdict={row['seed_verdict']}"
    )

print("\nMain report:")
print(path)
PY

echo
echo "============================================================"
echo "ALL TABLE-4 512000 VALIDATION STEPS COMPLETED"
echo "REPORT:"
echo "  ${ROOT}/${REPORT_DIR}/holdout_multiseed_report.md"
echo "LOGS:"
echo "  ${ROOT}/${LOG_DIR}"
echo "============================================================"
