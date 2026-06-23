from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from algorithms.mappo.opponent_policy import OpponentPolicy


DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml"
)


def _zero_actions(env) -> dict[str, np.ndarray]:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _blue_zero_actions(env) -> dict[str, np.ndarray]:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.blue_ids}


def _load_policy(checkpoint: str | None, device_name: str):
    if not checkpoint:
        return None, None, None
    import torch
    from scripts.eval_happo_reference import _build_policy_from_meta, _load_meta
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    device = torch.device(device_name)
    model_path = Path(checkpoint)
    meta = _load_meta(model_path)
    policy = _build_policy_from_meta(meta, device)
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    adapter = HeteroEntitySetAdapter() if meta.get("policy_arch") == "hetero_entity_recurrent" else HeteroObsAdapterV2()
    return policy, adapter, device


def _active_mask(env, info: dict) -> np.ndarray:
    active = np.zeros(len(env.red_ids), dtype=np.float32)
    for i, rid in enumerate(env.red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = env.red_planes.get(rid)
            alive = bool(sim is not None and sim.is_alive)
        active[i] = 1.0 if alive else 0.0
    return active


def _policy_actions(policy, adapter, env, obs, info, device, rnn_hidden, blue_opponent=None):
    if policy is None:
        return _zero_actions(env), rnn_hidden
    import torch
    from algorithms.happo.rollout_safety import (
        sanitize_policy_inputs,
        zero_inactive_actions,
        zero_inactive_hidden,
    )
    from scripts.eval_happo_reference import _role_ids
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter

    roles = _role_ids(env)
    active = _active_mask(env, info)
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    entity_mode = isinstance(adapter, HeteroEntitySetAdapter)
    act_kwargs = {}
    if rnn_hidden is not None:
        rnn_hidden = zero_inactive_hidden(rnn_hidden, active)
        act_kwargs["rnn_hidden"] = torch.as_tensor(rnn_hidden, device=device)

    with torch.no_grad():
        if entity_mode:
            actor_tokens = adapted["actor_entity_tokens"].copy()
            actor_keep = adapted["actor_keep_mask"].copy()
            critic_tokens = adapted["critic_entity_tokens"].copy()
            critic_keep = adapted["critic_keep_mask"].copy()
            inactive = active <= 0.5
            actor_tokens[inactive] = 0.0
            actor_keep[inactive] = 0.0
            actor_keep[inactive, 0] = 1.0
            out = policy.act(
                actor_tokens,
                actor_keep,
                roles,
                critic_tokens,
                critic_keep,
                deterministic=True,
                critic_counts=torch.as_tensor(
                    adapted.get("critic_counts", np.zeros(4, dtype=np.float32)),
                    device=device,
                ),
                **act_kwargs,
            )
        else:
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                for rid in env.red_ids
            ])
            critic = adapted["critic_state"]
            san = sanitize_policy_inputs(actor_obs, active, critic_state=critic, rnn_hidden=rnn_hidden)
            actor_obs = san["actor_obs"]
            critic = san["critic_state"] if san["critic_state"] is not None else critic
            rnn_hidden = san["rnn_hidden"] if san["rnn_hidden"] is not None else rnn_hidden
            out = policy.act(
                torch.as_tensor(actor_obs, device=device),
                roles=roles,
                critic_state=torch.as_tensor(critic, device=device),
                deterministic=True,
                **act_kwargs,
            )

    if rnn_hidden is not None and "rnn_hidden" in out:
        rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].detach().cpu().numpy(), active)
    red_actions = zero_inactive_actions(out["action"].detach().cpu().numpy(), active)
    actions = {rid: red_actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
    if blue_opponent is not None:
        actions.update(blue_opponent.act(obs, env.blue_ids, env=env))
    else:
        actions.update(_blue_zero_actions(env))
    return actions, rnn_hidden


def _record_is_red(record: dict[str, Any]) -> bool:
    return str(record.get("team") or record.get("shooter_team") or "").lower() == "red"


def _summarize_records(mode: str, launch_records: list[dict], done_records: list[dict]) -> dict:
    red_launches = [r for r in launch_records if _record_is_red(r)]
    red_done = [r for r in done_records if _record_is_red(r)]
    red_hits = [
        r for r in red_done
        if bool(r.get("is_success")) or str(r.get("termination_reason")) == "hit"
    ]
    threat_values = [
        float(r.get("selected_target_threat_score", 0.0) or 0.0)
        for r in red_launches
    ]
    mav_observed = [
        bool(r.get("selected_target_is_mav_observed", False))
        for r in red_launches
    ]
    return {
        "mode": mode,
        "red_launches": len(red_launches),
        "red_hits": len(red_hits),
        "red_hit_rate": (len(red_hits) / len(red_launches)) if red_launches else 0.0,
        "red_launches_by_target": dict(Counter(str(r.get("target_id", "")) for r in red_launches)),
        "red_hits_by_target": dict(Counter(str(r.get("target_id", "")) for r in red_hits)),
        "blue_threat_selected_fraction": (
            sum(1 for value in threat_values if value >= 0.5) / len(threat_values)
            if threat_values else 0.0
        ),
        "mav_observed_target_fraction": (
            sum(1 for value in mav_observed if value) / len(mav_observed)
            if mav_observed else 0.0
        ),
    }


def run_mode(
    config: str,
    mode: str,
    episodes: int,
    max_steps: int,
    policy,
    adapter,
    device: torch.device,
    blue_policy: str = "zero",
) -> dict:
    env = make_env(
        config,
        env_type="jsbsim_hetero",
        red_target_selection_mode=mode,
        max_steps=max_steps,
    )
    launch_records: list[dict] = []
    done_records: list[dict] = []
    print(f"[audit] mode={mode} blue={blue_policy} episodes={episodes} max_steps={max_steps}", flush=True)
    try:
        for episode in range(episodes):
            if episode % 5 == 0:
                print(f"[audit]   episode {episode}/{episodes}...", flush=True)
            _obs, _info = env.reset(seed=episode)
            obs, info = _obs, _info
            rnn_hidden = None
            if policy is not None and getattr(policy, "rnn_hidden_size", 0) > 0:
                rnn_hidden = np.zeros((len(env.red_ids), policy.rnn_hidden_size), dtype=np.float32)
            for _ in range(max_steps):
                blue_opp = OpponentPolicy(mode=blue_policy, seed=episode + 33) if blue_policy != "zero" else None
                actions, rnn_hidden = _policy_actions(policy, adapter, env, obs, info, device, rnn_hidden, blue_opponent=blue_opp)
                obs, _rewards, terminated, truncated, info = env.step(actions)
                launch_records.extend(info.get("__launch_quality_step__", []) or [])
                done_records.extend(info.get("__launch_quality_done__", []) or [])
                if all(terminated.values()) or all(truncated.values()):
                    break
    finally:
        env.close()
    return _summarize_records(mode, launch_records, done_records)


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "red_target_selection_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Red Target Selection Audit",
        "",
        f"config: `{summary['config']}`",
        f"episodes: {summary['episodes']}",
        f"max_steps: {summary['max_steps']}",
        "",
        "| mode | red launches | red hits | red hit rate | threat selected fraction | MAV observed fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["modes"]:
        lines.append(
            f"| {row['mode']} | {row['red_launches']} | {row['red_hits']} | "
            f"{row['red_hit_rate']:.3f} | {row['blue_threat_selected_fraction']:.3f} | "
            f"{row['mav_observed_target_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "The audit changes only target ranking mode. BRMA launch gates, lock delay, cooldown, "
        "deconfliction, kill cooldown, missile dynamics, and blue rule behavior are not modified.",
    ])
    (output_dir / "red_target_selection_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit red closest target selection against MAV-aware threat ranking."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None, help="Optional HAPPO checkpoint to use for red actions.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blue-policy", default="zero", choices=["zero", "brma_rule"])
    parser.add_argument("--output-dir", default="outputs/environment_audit/red_target_selection")
    args = parser.parse_args()

    policy, adapter, device = _load_policy(args.checkpoint, args.device)
    summaries = [
        run_mode(args.config, "closest", args.episodes, args.max_steps, policy, adapter, device, blue_policy=args.blue_policy),
        run_mode(args.config, "mav_threat_rank", args.episodes, args.max_steps, policy, adapter, device, blue_policy=args.blue_policy),
    ]
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint or "",
        "policy_source": "checkpoint" if args.checkpoint else "zero_action",
        "blue_policy": args.blue_policy,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "modes": summaries,
    }
    write_outputs(payload, Path(args.output_dir))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
