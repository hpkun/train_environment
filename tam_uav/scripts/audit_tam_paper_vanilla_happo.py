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
    "uav_env/JSBSim/paper/env.py": "b712133166eddf396d8b77862303f541c97c4f524fb877b7822da87171cacb3f",
    "uav_env/JSBSim/paper/task.py": "92c40e8090a696b53ec50be9e7115f74e9c36ddc7030e6a346984b822a1e0132",
    "uav_env/JSBSim/paper/reward.py": "815acc1dc0d756dea9ca47f2d9223af248bbc9b80a1fdfce434bc9fe74e2d710",
    "uav_env/JSBSim/paper/observation.py": "50910f7e715b3d4197b20f72fef61601f3061178f8093eb43b2790a7bb36054a",
    "uav_env/JSBSim/paper/missile.py": "8d8e79b31b697ef3b75c3ba50ddbeb0271302707f468bed78f90ff17ffc92121",
    "uav_env/JSBSim/paper/weapon.py": "4155f26ddcfaf76ad9f4314275ef6c4af4a8966e127cce2f4e2ea31fc5b3fcf1",
    "uav_env/JSBSim/paper/protocol.py": "74b0698d5a1fa28c3b551efc041b9fa16542b575f1b64e44a1a0ca86ac26878a",
    "uav_env/JSBSim/paper/action_semantics.py": "daf980b98448e23d47d07fa159ad71a1dad5d518f6eefe44360f02e43957569f",
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
