"""Read-only contract audit for the v5 vanilla-HAPPO environment interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_HASHES = {
    "uav_env/JSBSim/paper/env.py": "64bf3ec6647e15096a1eafe79ea7c9d8140c9c4027c7296c33236a1a5a230432",
    "uav_env/JSBSim/paper/task.py": "a31a5a19255301173d8e4f8a4900a808d99d34a8030bbc41dbc83e034d740fd7",
    "uav_env/JSBSim/paper/reward.py": "59365b7fef8d1b170280560a5de14984a0c4e82c50b173c33d23a21e954c0aa0",
    "uav_env/JSBSim/paper/observation.py": "b7ec2903e97163b8050e37329b76ceef0882ce4a0b5a65d73fb958488e5dbe17",
    "uav_env/JSBSim/paper/missile.py": "df97893430a3e9288170e8138cdbe6325c140cdef26d43923ad933efaf904e83",
    "uav_env/JSBSim/paper/weapon.py": "b765490f5c1f6a3608e73e91e5c13dac8346364dfc94a801323d2e6012d87b64",
    "uav_env/JSBSim/paper/protocol.py": "3b61ebd989a5d25a0371e074fc5d59d034d3b2cd108c53052e9cad87a19676e8",
    "uav_env/JSBSim/paper/opponent.py": "f73e8d6be519ad398e9bc38bac9e6e72b50414402f314f655b740a48db1f9482",
    "uav_env/JSBSim/paper/action_semantics.py": "daf980b98448e23d47d07fa159ad71a1dad5d518f6eefe44360f02e43957569f",
    "uav_env/JSBSim/core/aircraft.py": "eb53f61a3680e84f6db090e3a85dc2a9d3351e3dbe1c7618e867c6a34750463f",
    "uav_env/JSBSim/core/aircraft_types.py": "8d3470dc4c2a2133ef9f3c022805bfb6cf43a96715fd97d0d94da5ed5f4e0c0b",
    "uav_env/JSBSim/core/scenario.py": "88d1c689696a43b34d8a6e1befd96bd7565e61f90c255bfa1bc46db322a28b85",
    "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml": "f328c7fd2cb8e5f696f6a394c05d4bd75956b604bae049c94781f7a97eb5317c",
    "uav_env/JSBSim/configs/tam_paper_env_v1_3v2.yaml": "0a2f43f8f1af566af44d95dbe8fc66e5e59f35d3635633a9da6024c393acb6c4",
    "uav_env/JSBSim/configs/tam_paper_env_v1_5v4.yaml": "5302a73495e2ef2c19857b7661b597152d86ca08c5a5ed7729358c235fb1c6cc",
}


def environment_contract_unchanged():
    actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ENVIRONMENT_HASHES}
    return actual == ENVIRONMENT_HASHES, actual


def audit(*, tests_passed, smoke_summaries=()):
    environment_ok, actual_hashes = environment_contract_unchanged()
    source = (ROOT / "algorithms/happo/vanilla_happo.py").read_text(encoding="utf-8")
    forbidden = [token for token in (
        "nn.GRU", "nn.LSTM", "Transformer", "MultiheadAttention")
        if token in source]
    summaries = [json.loads(Path(path).read_text(encoding="utf-8"))
                 for path in smoke_summaries]
    optimization = bool(summaries and all(
        item.get("OPTIMIZATION_PIPELINE_ACTIVE") is True for item in summaries))
    finite = bool(summaries and all(item.get("nan_inf_count", 0) == 0
                                    for item in summaries))
    ready = bool(tests_passed and environment_ok and optimization and finite
                 and not forbidden)
    return {
        "environment_fidelity_revision": "published_environment_reconstruction_v5",
        "ENVIRONMENT_CONTRACT_UNCHANGED": environment_ok,
        "environment_hashes": actual_hashes,
        "tests_passed": bool(tests_passed),
        "forbidden_network_tokens": forbidden,
        "OPTIMIZATION_PIPELINE_ACTIVE": optimization,
        "HAPPO_ALGORITHM_CONTRACT_READY": ready,
        "EARLY_PERFORMANCE_SIGNAL_OBSERVED": False,
        "LEARNING_CONVERGENCE_NOT_VALIDATED": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--smoke-summary", action="append", default=[])
    args = parser.parse_args(argv)
    print(json.dumps(audit(tests_passed=args.tests_passed,
                           smoke_summaries=args.smoke_summary), indent=2))


if __name__ == "__main__":
    main()
