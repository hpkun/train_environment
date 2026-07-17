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
    "uav_env/JSBSim/paper/task.py": "9f3a56b76387ea9d4707e73df9d3d62e4c5f977f15d42e1452349931f06c8270",
    "uav_env/JSBSim/paper/reward.py": "2db4aa290d652340faccbc8651acf148531d8eca6794785d63cc54442a739218",
    "uav_env/JSBSim/paper/observation.py": "50910f7e715b3d4197b20f72fef61601f3061178f8093eb43b2790a7bb36054a",
    "uav_env/JSBSim/paper/missile.py": "1ef8261909a5355a666b46d9b1afe9cb378bed953ae7c2b84da06d98f424f533",
    "uav_env/JSBSim/paper/weapon.py": "4155f26ddcfaf76ad9f4314275ef6c4af4a8966e127cce2f4e2ea31fc5b3fcf1",
    "uav_env/JSBSim/paper/protocol.py": "58fcc78ec4de759d08ca75d6f85446648d9cf74d8f2565593dd000af27ea4175",
    "uav_env/JSBSim/core/aircraft.py": "4da145ebd6fe478d1f25ae24ee507cb6427a413ec0549a7d939e1d95ead138bf",
    "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml": "4c35649960441cf0bbcd8e992c96023873bca7e8ad30b84a8e3fb66ab43eeb13",
    "uav_env/JSBSim/configs/tam_paper_env_v1_3v2.yaml": "a66b2cd332fc88dd8e357949521e204bed1d5476e3c73c473793401d97626ada",
    "uav_env/JSBSim/configs/tam_paper_env_v1_5v4.yaml": "07656d7664cd7c86ca6ed7a3a45b4ee8d6a6b6ee5cca9424d1a0b1ac657732ad",
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
