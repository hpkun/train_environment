"""Read checkpoint lineage without loading a policy or running an environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo_checkpoint import (
    FORMAT, read_vanilla_happo_checkpoint_metadata)
from scripts.tam_output_paths import resolve_tam_output
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL)


def inspect_checkpoint(path):
    metadata = read_vanilla_happo_checkpoint_metadata(path)
    expected = {
        "format": FORMAT,
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "experiment_protocol": PAPER_NOMINAL_PROTOCOL,
        "initial_perturbation": NOMINAL_PERTURBATION,
        "dynamics_backend": "jsbsim",
    }
    reasons = [f"{key}: {metadata.get(key)!r} != {value!r}"
               for key, value in expected.items() if metadata.get(key) != value]
    revision = metadata.get("environment_fidelity_revision")
    return {
        "checkpoint": str(Path(path)),
        "metadata": metadata,
        "formal_environment_compatible": not reasons,
        "incompatibility_reasons": reasons,
        "pre_fidelity_checkpoint": revision != ENVIRONMENT_FIDELITY_REVISION,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    checkpoint = resolve_tam_output(ROOT, args.checkpoint)
    result = inspect_checkpoint(checkpoint)
    if args.output:
        output = resolve_tam_output(ROOT, args.output, create_parent=True)
        output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
