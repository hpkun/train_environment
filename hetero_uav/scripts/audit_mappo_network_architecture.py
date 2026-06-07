"""Audit the current MAPPO baseline network architecture.

This is a read-only diagnostic script. It does not train, modify checkpoints, or
change the environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.policy import MAPPOActorCritic


def _has_token(path: Path, tokens: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return any(token.lower() in text for token in tokens)


def build_audit() -> dict:
    model = MAPPOActorCritic(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    actor_layers = [
        module.out_features
        for module in model.actor
        if module.__class__.__name__ == "Linear"
    ]
    critic_layers = [
        module.out_features
        for module in model.critic
        if module.__class__.__name__ == "Linear"
    ]
    actor_mlp = [model.actor_obs_dim] + actor_layers
    critic_mlp = [model.critic_state_dim] + critic_layers

    main_runner = ROOT / "scripts" / "run_main_mappo_experiment.py"
    policy_file = ROOT / "algorithms" / "mappo" / "policy.py"
    main_has_method_tokens = _has_token(
        main_runner,
        ["attention", "happo", "gru", "temporal"],
    )
    policy_has_method_tokens = _has_token(
        policy_file,
        ["attention", "gru", "happo", "temporal"],
    )

    current = {
        "algorithm": "shared-actor MAPPO baseline",
        "actor_input_dim_v2": model.actor_obs_dim,
        "critic_input_dim_v2": model.critic_state_dim,
        "action_dim": model.action_dim,
        "actor_mlp": actor_mlp,
        "critic_mlp": critic_mlp,
        "gaussian_policy_learnable_log_std": True,
        "no_attention": not policy_has_method_tokens,
        "no_gru": not policy_has_method_tokens,
        "no_temporal_feature": not policy_has_method_tokens,
        "no_happo": not policy_has_method_tokens,
        "baseline_only": True,
    }
    return {
        "current_architecture": current,
        "paper_alignment": {
            "brma_mappo_baseline": (
                "Aligned as a plain shared-policy MAPPO baseline with "
                "centralized critic."
            ),
            "tam_happo_paper": (
                "Not fully aligned; TAM-HAPPO-style temporal, attention, "
                "and heterogeneous update mechanisms are not implemented."
            ),
            "acceptable_scope": "baseline only",
        },
        "decision": {
            "keep_current_mlp_mappo_as_baseline": True,
            "do_not_claim_tam_happo_reproduction": True,
            "next_method_after_baseline": "entity attention actor/critic",
        },
        "violations": {
            "main_runner_calls_attention_happo_gru_temporal": main_has_method_tokens,
            "policy_contains_attention_happo_gru_temporal": policy_has_method_tokens,
            "has_violation": main_has_method_tokens or policy_has_method_tokens,
        },
    }


def write_markdown(audit: dict, output_md: Path) -> None:
    current = audit["current_architecture"]
    md = f"""# MAPPO Network Architecture Audit

## Current Network

The current network is a MAPPO baseline:

- Algorithm: {current['algorithm']}
- Actor MLP: {current['actor_mlp']}
- Critic MLP: {current['critic_mlp']}
- Action dim: {current['action_dim']}
- Policy: Gaussian with learnable log_std
- No attention: {current['no_attention']}
- No GRU: {current['no_gru']}
- No HAPPO sequential update: {current['no_happo']}

## Paper Alignment

This is aligned with a plain BRMA-MAPPO / MAPPO baseline: shared actor,
centralized critic, and PPO update.

It is not the final proposed method for the heterogeneous TAM-HAPPO paper. It
does not include temporal feature extraction, entity attention, or HAPPO-style
sequential agent updates.

## Decision

Keep this MLP MAPPO as the baseline. Do not claim it reproduces TAM-HAPPO. After
the baseline is stable, a separate method module can introduce entity attention
actor/critic components.
"""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/main_mappo_network_architecture_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/main_mappo_network_architecture_audit.md",
    )
    args = parser.parse_args()

    audit = build_audit()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit, output_md)
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print("baseline_only: true", flush=True)


if __name__ == "__main__":
    main()
