"""Read-only 2v2 side-mirror audit for the frozen TAM paper environment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import stack_controlled_rule_actions
from uav_env.JSBSim.paper.env import TAMPaperEnv


def mirrored_config(config):
    mirrored = copy.deepcopy(config)
    mirrored.pop("env_type", None)
    original_red = copy.deepcopy(config["red_agents"])
    original_blue = copy.deepcopy(config["blue_agents"])
    mirrored["red_agents"] = [dict(agent, id=f"red_{index}")
                              for index, agent in enumerate(original_blue)]
    mirrored["blue_agents"] = [dict(agent, id=f"blue_{index}")
                               for index, agent in enumerate(original_red)]
    mirrored["scenario"] = "2v2"
    return mirrored


def _target_signature(targets, swap_sides=False):
    def mapped(agent_id):
        side, index = agent_id.split("_", 1)
        if swap_sides:
            side = "blue" if side == "red" else "red"
        return f"{side}_{index}"
    return sorted((mapped(aid), mapped(target))
                  for aid, target in targets.items() if target is not None)


def run_rule_episode(env, seed, *, swap_signature=False):
    env.reset(seed=seed)
    target_signatures, all_events = [], []
    finite, violations, ordering = True, 0, True
    while True:
        env.prepare_decision_context()
        actions = stack_controlled_rule_actions(env)
        _obs, rewards, terminated, truncated, info = env.step(
            {aid: actions[index] for index, aid in enumerate(env.agent_ids)})
        target_signatures.append(_target_signature(
            info["target_used_by_rule_action"], swap_signature))
        all_events.extend(info["missile_events"])
        finite &= bool(np.isfinite(list(rewards.values())).all()
                       and np.isfinite(env.get_state()).all())
        violations += len(info["target_consistency_violation"])
        ordering &= bool(info["event_ordering_consistent"])
        if all(bool(terminated[aid] or truncated[aid]) for aid in env.agent_ids):
            break
    metrics = info["aircraft_metrics"]
    result = {
        "winner": info["winner"],
        "termination_reason": info["termination_reason"],
        "episode_length": info["episode_step"],
        "missiles": {side: 0 for side in ("red", "blue")},
        "hits": {side: 0 for side in ("red", "blue")},
        "kills": dict(info["kills"]),
        "crashes": {side: 0 for side in ("red", "blue")},
        "boundary": {side: 0 for side in ("red", "blue")},
        "target_signatures": target_signatures,
        "target_consistency_violation": violations,
        "event_ordering_consistent": ordering,
        "finite": finite,
        "events": all_events,
    }
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    for event in all_events:
        shooter = by_id.get(event.get("shooter_id"))
        if shooter and event.get("reason") == "launched":
            result["missiles"][shooter.side] += 1
        if shooter and event.get("reason") == "hit":
            result["hits"][shooter.side] += 1
    for agent in env.task.agents:
        reason = metrics[agent.agent_id]["death_reason"]
        result["crashes"][agent.side] += int(reason in {"crash", "nonfinite"})
        result["boundary"][agent.side] += int(reason == "boundary")
    return result


def audit(seed=30260717):
    yaml_path = ROOT / "uav_env" / "JSBSim" / "configs" / "tam_paper_env_v1_2v2.yaml"
    original_text = yaml_path.read_bytes()
    config = yaml.safe_load(original_text.decode("utf-8"))
    normal_config = copy.deepcopy(config); normal_config.pop("env_type", None)
    normal_config["scenario"] = "2v2"
    normal, mirror = TAMPaperEnv(**normal_config), TAMPaperEnv(**mirrored_config(config))
    try:
        a = run_rule_episode(normal, seed)
        b = run_rule_episode(mirror, seed, swap_signature=True)
    finally:
        normal.close(); mirror.close()
    if yaml_path.read_bytes() != original_text:
        raise RuntimeError("symmetry audit modified the formal YAML")
    swapped_winner = {"red": "blue", "blue": "red", "draw": "draw"}[a["winner"]]
    exchanged_metrics = all((
        a[name]["red"] == b[name]["blue"]
        and a[name]["blue"] == b[name]["red"])
        for name in ("missiles", "hits", "kills", "crashes", "boundary"))
    target_consistent = a["target_signatures"] == b["target_signatures"]
    event_consistent = (a["event_ordering_consistent"]
                        and b["event_ordering_consistent"])
    mirror_outcome = b["winner"] == swapped_winner and exchanged_metrics
    return {
        "seed": seed,
        "normal": a,
        "mirrored": b,
        "episode_length_difference": b["episode_length"] - a["episode_length"],
        "metrics_exchange_consistent": exchanged_metrics,
        "MIRROR_OUTCOME_CONSISTENT": mirror_outcome,
        "SIDE_ORDER_BIAS_DETECTED": not mirror_outcome,
        "TARGET_ORDER_BIAS_DETECTED": not target_consistent,
        "EVENT_ORDER_BIAS_DETECTED": not event_consistent,
        "target_consistency_violation": (
            a["target_consistency_violation"] + b["target_consistency_violation"]),
        "finite": bool(a["finite"] and b["finite"]),
        "formal_yaml_unchanged": True,
    }


def _markdown(report):
    return "\n".join([
        "# TAM paper 2v2 symmetry audit", "",
        f"- Mirror outcome consistent: `{report['MIRROR_OUTCOME_CONSISTENT']}`",
        f"- Side-order bias detected: `{report['SIDE_ORDER_BIAS_DETECTED']}`",
        f"- Target-order bias detected: `{report['TARGET_ORDER_BIAS_DETECTED']}`",
        f"- Event-order bias detected: `{report['EVENT_ORDER_BIAS_DETECTED']}`",
        f"- Target consistency violations: `{report['target_consistency_violation']}`",
        f"- Formal YAML unchanged: `{report['formal_yaml_unchanged']}`", "",
        "This audit records evidence only and does not modify the environment.",
    ]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=30260717)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    output = resolve_tam_output(ROOT, args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = audit(args.seed)
    (output / "tam_paper_2v2_symmetry_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output / "tam_paper_2v2_symmetry_audit.md").write_text(
        _markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "MIRROR_OUTCOME_CONSISTENT", "SIDE_ORDER_BIAS_DETECTED",
        "TARGET_ORDER_BIAS_DETECTED", "EVENT_ORDER_BIAS_DETECTED")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
