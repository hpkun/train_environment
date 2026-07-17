"""Independent readiness audit for the vanilla-HAPPO training pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENVIRONMENT_HASHES = {
    "uav_env/JSBSim/paper/env.py": "9a9ae99738ba9a8ea8fefeef99f941c345fb96e5c41eb4e4492cc347c8f884ba",
    "uav_env/JSBSim/paper/task.py": "60c06a9ce6035210d84cdfbf40eb4c039ac5077571194734a6a0e6b3b345749a",
    "uav_env/JSBSim/paper/reward.py": "8f06d7f3279c016cb82d1e790e866ca1f07aabcd2934e5f5180f25d4f2a86a80",
    "uav_env/JSBSim/paper/observation.py": "92b76c39b443e7849fcb34f005c5494a33ee9158338276f86c603431a35fd7af",
    "uav_env/JSBSim/paper/missile.py": "84524ddb491a89ba482db9d84ad5f1240c8d272f65ceaee690b5784803451638",
    "uav_env/JSBSim/paper/weapon.py": "45cb936eebc53bdfeb147fb5fe73de9e6bfa71587ab15b6af8b3a6710a0a737c",
    "uav_env/JSBSim/paper/protocol.py": "bb5462d7da563a6c75a87dea908282a6fc0f20903c3efb85b9f36f4f3fcc1699",
    "uav_env/JSBSim/core/aircraft.py": "4da145ebd6fe478d1f25ae24ee507cb6427a413ec0549a7d939e1d95ead138bf",
    "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml": "595653b8197cff513c044055966febca1d2c2997ae1c4cb6d8579ef68294df94",
    "uav_env/JSBSim/configs/tam_paper_env_v1_3v2.yaml": "eebc364c006017624efc60b2e0480964ccd848b820c9cbd20103a69f4434734b",
    "uav_env/JSBSim/configs/tam_paper_env_v1_5v4.yaml": "d83e52bd547a5825bea89e3784c1392b1f26a7d0b886b6c008ca91fdf920b30c",
}


def environment_contract_unchanged():
    actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ENVIRONMENT_HASHES}
    return actual == ENVIRONMENT_HASHES, actual


def audit(*, tests_passed, smoke_summaries=()):
    env_ok, actual_hashes = environment_contract_unchanged()
    source = (ROOT / "algorithms/happo/vanilla_happo.py").read_text(encoding="utf-8")
    forbidden = [token for token in ("nn.GRU", "nn.LSTM", "Transformer",
                                      "MultiheadAttention") if token in source]
    summaries = [json.loads(Path(path).read_text(encoding="utf-8"))
                 for path in smoke_summaries]
    optimization = bool(summaries and all(
        item.get("OPTIMIZATION_PIPELINE_ACTIVE") is True for item in summaries))
    runtime = bool(summaries and all(
        item.get("HAPPO_RUNTIME_INVARIANTS_VALID") is True for item in summaries))
    algorithm = bool(tests_passed and runtime and not forbidden)
    smoke = bool(summaries and all(
        item.get("latest_metrics", {}).get("nan_inf_count") == 0
        and item.get("latest_metrics", {}).get("target_consistency_violation") == 0
        for item in summaries))
    return {
        "environment_hashes": actual_hashes,
        "forbidden_network_tokens": forbidden,
        "unit_tests_passed": bool(tests_passed),
        "ENVIRONMENT_CONTRACT_UNCHANGED": env_ok,
        "HAPPO_ALGORITHM_CONTRACT_READY": algorithm,
        "HAPPO_RUNTIME_INVARIANTS_VALID": runtime,
        "HAPPO_TRAINING_PIPELINE_READY": bool(algorithm and optimization),
        "JSBSIM_TRAINING_SMOKE_READY": smoke,
        "OPTIMIZATION_PIPELINE_ACTIVE": optimization,
        "EARLY_PERFORMANCE_SIGNAL_OBSERVED": False,
        "early_performance_signal_reason": "insufficient_evaluation_samples",
        "LEARNING_CONVERGENCE_NOT_VALIDATED": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--smoke-summary", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(tests_passed=args.tests_passed, smoke_summaries=args.smoke_summary)
    text = json.dumps(result, indent=2)
    if args.output:
        from scripts.tam_output_paths import resolve_tam_output
        path = resolve_tam_output(ROOT, args.output, create_parent=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["HAPPO_ALGORITHM_CONTRACT_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
