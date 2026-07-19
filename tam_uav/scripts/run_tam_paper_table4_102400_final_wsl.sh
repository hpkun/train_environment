#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/mnt/c/Users/HPK/Desktop/train_environment/tam_uav"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TOTAL_STEPS=102400
ROLLOUT_LENGTH=128
EVALUATION_INTERVAL=5120
CHECKPOINT_INTERVAL=102400

PANEL_SIZE=50
ENVIRONMENT_SEED=30260717
POLICY_ACTION_SEED_BASE=60260717

SEEDS=(20260717 20260718 20260719)

LOG_DIR="outputs/console_logs_table4_102400_final"
REPORT_DIR="outputs/tam_paper_v4_table4_happo_2v2_multiseed_20260717_20260719_102400"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "TAM paper v4 final environment learnability validation"
echo "Algorithm: pure feedforward HAPPO"
echo "Actor: [256, 128]"
echo "Critic: [256, 128]"
echo "Value loss: clipped Huber, delta=10"
echo "Scenario: 2v2"
echo "Steps per seed: ${TOTAL_STEPS}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"

python - <<'PY'
import torch

from algorithms.happo.vanilla_happo import (
    PAPER_ACTOR_HIDDEN_SIZES,
    PAPER_CRITIC_HIDDEN_SIZES,
)
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
print("actor hidden sizes:", args.actor_hidden_sizes)
print("critic hidden sizes:", args.critic_hidden_sizes)
print("value loss:", args.value_loss_type)
print("huber delta:", args.huber_delta)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA不可用")

if ENVIRONMENT_FIDELITY_REVISION != "published_rules_simplified_v4":
    raise SystemExit("ERROR: 环境revision不是冻结的v4")

if tuple(PAPER_ACTOR_HIDDEN_SIZES) != (256, 128):
    raise SystemExit("ERROR: actor论文宽度异常")

if tuple(PAPER_CRITIC_HIDDEN_SIZES) != (256, 128):
    raise SystemExit("ERROR: critic论文宽度异常")

if args.actor_hidden_sizes != [256, 128]:
    raise SystemExit("ERROR: actor CLI默认值异常")

if args.critic_hidden_sizes != [256, 128]:
    raise SystemExit("ERROR: critic CLI默认值异常")

if args.value_loss_type != "clipped_huber":
    raise SystemExit("ERROR: value loss默认值异常")

if args.huber_delta != 10.0:
    raise SystemExit("ERROR: Huber delta异常")

print("PRECHECK: PASS")
PY

if [[ -e "${REPORT_DIR}" ]]; then
    echo "ERROR: 多种子报告目录已存在："
    echo "  ${REPORT_DIR}"
    echo "为避免覆盖旧结果，本次运行中止。"
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="outputs/tam_paper_v4_table4_happo_2v2_seed${SEED}_${TOTAL_STEPS}"
    PANEL_DIR="${RUN_DIR}/robust_evaluation_v2"

    if [[ -e "${RUN_DIR}" ]]; then
        echo "ERROR: 训练目录已存在："
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
    echo "STOCHASTIC PANEL SEED ${SEED}"
    echo "CHECKPOINTS: 0, ${TOTAL_STEPS}"
    echo "============================================================"

    python -u scripts/evaluate_tam_paper_checkpoint_panel.py \
      --run-directory "${RUN_DIR}" \
      --checkpoint-steps 0 "${TOTAL_STEPS}" \
      --panel-size "${PANEL_SIZE}" \
      --environment-seed "${ENVIRONMENT_SEED}" \
      --policy-action-seed-base "${POLICY_ACTION_SEED_BASE}" \
      --bootstrap-samples 10000 \
      --device cuda \
      --output "${PANEL_DIR}" \
      2>&1 | tee "${LOG_DIR}/panel_seed_${SEED}.log"

    echo
    echo "============================================================"
    echo "SINGLE-SEED ANALYSIS ${SEED}"
    echo "============================================================"

    python -u scripts/analyze_tam_paper_learnability.py \
      --run-directory "${RUN_DIR}" \
      --output-directory "${PANEL_DIR}" \
      2>&1 | tee "${LOG_DIR}/analysis_seed_${SEED}.log"

    echo
    echo "SEED ${SEED} COMPLETED"
done

RUN_1="outputs/tam_paper_v4_table4_happo_2v2_seed20260717_${TOTAL_STEPS}"
RUN_2="outputs/tam_paper_v4_table4_happo_2v2_seed20260718_${TOTAL_STEPS}"
RUN_3="outputs/tam_paper_v4_table4_happo_2v2_seed20260719_${TOTAL_STEPS}"

echo
echo "============================================================"
echo "MULTI-SEED ANALYSIS"
echo "============================================================"

python -u scripts/analyze_tam_paper_multiseed_learnability.py \
  --run-directories \
    "${RUN_1}" \
    "${RUN_2}" \
    "${RUN_3}" \
  --output-directory "${REPORT_DIR}" \
  2>&1 | tee "${LOG_DIR}/multiseed_analysis.log"

echo
echo "============================================================"
echo "FINAL COMPACT RESULT"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

report_path = Path(
    "outputs/"
    "tam_paper_v4_table4_happo_2v2_multiseed_"
    "20260717_20260719_102400/"
    "multiseed_learnability_report.json"
)

report = json.loads(report_path.read_text(encoding="utf-8"))
cross = report["cross_seed"]

print("multiseed_verdict:", report["multiseed_verdict"])
print("all_runtime_valid:", cross["all_runtime_valid"])
print(
    "return_improved_seed_count:",
    cross["return_improved_seed_count"],
)
print(
    "combat_improved_seed_count:",
    cross["combat_improved_seed_count"],
)
print(
    "beats_random_seed_count:",
    cross["beats_random_seed_count"],
)
print(
    "mean_paired_return_delta:",
    cross["mean_paired_return_delta"],
)
print(
    "seed_level_bootstrap_ci95:",
    cross["seed_level_mean_delta_bootstrap_ci95"],
)

print("\nPer seed:")
for row in report["seed_summaries"]:
    print(
        f"seed={row['training_seed']} "
        f"runtime={row['runtime_valid']} "
        f"step0={row['step_0_panel_mean_return']:.3f} "
        f"final={row['final_panel_mean_return']:.3f} "
        f"delta={row['paired_return_delta_vs_step_0']:.3f} "
        f"ci=[{row['paired_return_ci_low']:.3f},"
        f"{row['paired_return_ci_high']:.3f}] "
        f"combat_gains={row['combat_improvement_count']} "
        f"beats_random={row['beats_uniform_random']} "
        f"verdict={row['single_seed_verdict']}"
    )

print("\nReport:")
print(report_path)
PY

echo
echo "============================================================"
echo "ALL FINAL VALIDATION STEPS COMPLETED"
echo "REPORT:"
echo "  ${ROOT}/${REPORT_DIR}/multiseed_learnability_report.md"
echo "LOGS:"
echo "  ${ROOT}/${LOG_DIR}"
echo "============================================================"
