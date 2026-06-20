"""Compare initial and trained categorical MAV policies over long horizons."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_hidden
from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.validate_tam_categorical_initial_policy_flight import (
    _alive_mask, _disable_blue_missiles, _fixed_blue_actions, _mav_trace_row,
)
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


def summarize_policy_traces(traces):
    rows = [row for trace in traces for row in trace]
    if not rows:
        return {}
    actions = np.asarray([row["action_indices"] for row in rows], dtype=np.int64)
    elevator_counts = Counter(actions[:, 2].tolist())
    predeath = [row for trace in traces for row in trace[-100:]]
    return {
        "action_bin_usage": [int(np.unique(actions[:, axis]).size) for axis in range(4)],
        "dominant_action_bins": [
            int(Counter(actions[:, axis].tolist()).most_common(1)[0][0])
            for axis in range(4)
        ],
        "dominant_elevator_bin": int(elevator_counts.most_common(1)[0][0]),
        "mean_elevator_bin": float(actions[:, 2].mean()),
        "predeath_100_mean_altitude_m": float(np.mean([row["altitude_m"] for row in predeath])),
        "predeath_100_mean_pitch_rad": float(np.mean([row.get("pitch_rad", 0.0) for row in predeath])),
        "predeath_100_mean_vertical_speed_mps": float(np.mean([row["vertical_speed_mps"] for row in predeath])),
    }


def _load_checkpoint_policy(checkpoint, device):
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint.parent / "meta.json").read_text(encoding="utf-8"))
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=int(meta.get("entity_dim", 19)),
        actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
        critic_state_dim=int(meta.get("critic_state_dim", 480)),
        action_dim=int(meta.get("action_dim", 4)),
        action_levels=int(meta.get("tam_action_levels", meta.get("action_levels", 40))),
        rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        neutral_action_init=bool(meta.get("neutral_action_init", True)),
        neutral_action_init_std_bins=float(meta.get("neutral_action_init_std_bins", 4.0)),
    ).to(device)
    policy.load(checkpoint, map_location=device)
    policy.eval()
    return policy


def _nearest_blue_range(env):
    mav = env.red_planes["red_0"]
    ranges = [
        float(np.linalg.norm(blue.get_position() - mav.get_position()))
        for blue in env.blue_planes.values() if blue.is_alive
    ]
    return min(ranges) if ranges else float("nan")


def _run_episode(config, policy, adapter, *, deterministic, no_blue_missile,
                 max_steps, seed, device):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    obs, info = env.reset(seed=seed)
    if no_blue_missile:
        _disable_blue_missiles(env)
    opponent = OpponentPolicy("tam_direct_fsm", seed=seed + 1000)
    roles = [0 if rid == "red_0" else 1 for rid in env.red_ids]
    hidden = policy.init_hidden(len(env.red_ids), torch.device(device))
    trace, launch_records = [], []
    for _step in range(max_steps):
        adapted = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids
        )
        actor_obs = np.stack([
            adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
            for rid in env.red_ids
        ])
        active = _alive_mask(env)
        sanitized = sanitize_policy_inputs(
            actor_obs, active, critic_state=adapted["critic_state"],
            rnn_hidden=hidden.detach().cpu().numpy(),
        )
        with torch.no_grad():
            out = policy.act(
                torch.as_tensor(sanitized["actor_obs"], device=device),
                roles=roles,
                critic_state=torch.as_tensor(sanitized["critic_state"], device=device),
                deterministic=deterministic,
                rnn_hidden=torch.as_tensor(sanitized["rnn_hidden"], device=device),
            )
        actions = out["action"].detach().cpu().numpy().astype(np.int64)
        hidden = torch.as_tensor(zero_inactive_hidden(
            out["rnn_hidden"].detach().cpu().numpy(), active
        ), device=device)
        action_dict = {rid: actions[i].copy() for i, rid in enumerate(env.red_ids)}
        if no_blue_missile:
            action_dict.update(_fixed_blue_actions(env))
        else:
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
        obs, _rewards, _terminated, _truncated, info = env.step(action_dict)
        probability_max = out["action_probs"][0].max(-1).values.detach().cpu().numpy()
        row = _mav_trace_row(
            env, actions[0], action_probs_max=float(probability_max.mean()),
            expected_action=out["mean"][0].detach().cpu().numpy(),
        )
        row["action_probs_max_per_axis"] = probability_max.tolist()
        row["nearest_blue_range_m"] = _nearest_blue_range(env)
        row["missile_warning"] = bool(env.red_planes["red_0"].check_missile_warning())
        trace.append(row)
        launch_records.extend(info.get("__launch_quality_step__", []) or [])
        launch_records.extend(info.get("__launch_quality_done__", []) or [])
        if not env.red_planes["red_0"].is_alive:
            break
    alive = bool(env.red_planes["red_0"].is_alive)
    result = {
        "seed": seed, "survived": bool(alive and len(trace) >= max_steps),
        "death_step": -1 if alive else int(env.current_step),
        "death_reason": env._death_reasons.get("red_0") or ("alive" if alive else "unknown"),
        "trace": trace,
        "launch_quality": launch_records,
    }
    env.close()
    return result


def _scenario_summary(episodes):
    traces = [episode["trace"] for episode in episodes]
    summary = summarize_policy_traces(traces)
    summary.update({
        "mav_survival_rate": float(np.mean([episode["survived"] for episode in episodes])),
        "death_reasons": dict(Counter(episode["death_reason"] for episode in episodes)),
        "mean_death_step": float(np.mean([
            episode["death_step"] for episode in episodes if episode["death_step"] > 0
        ])) if any(episode["death_step"] > 0 for episode in episodes) else -1.0,
    })
    return summary


def _launch_summary(scenarios):
    records = [
        record for scenario in scenarios.values() for episode in scenario["episodes"]
        for record in episode["launch_quality"]
        if record.get("shooter_team", record.get("team")) == "red"
    ]
    launches = [record for record in records if not record.get("termination_reason")]
    completed = [record for record in records if record.get("termination_reason")]
    def mean(field):
        values = [float(record[field]) for record in launches if record.get(field) not in ("", None)]
        return float(np.mean(values)) if values else float("nan")
    return {
        "red_launches": len(launches),
        "red_hits": sum(bool(record.get("is_success")) for record in completed),
        "mean_range_m": mean("range_m"),
        "mean_AO_deg": mean("AO_deg"),
        "mean_TA_deg": mean("TA_deg"),
        "mean_shooter_speed_mps": mean("shooter_speed_mps"),
        "termination_reasons": dict(Counter(
            str(record.get("termination_reason")) for record in completed
        )),
    }


def run_diagnosis(config, checkpoint, *, output_dir, episodes=10,
                  max_steps=1000, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapter = HeteroObsAdapterV2()
    trained = _load_checkpoint_policy(checkpoint, device)
    torch.manual_seed(seed)
    initial = TAMCategoricalRecurrentHAPPOPolicy().to(device)
    initial.eval()
    specs = {
        "initial_deterministic_no_blue_missile": (initial, True, True),
        "initial_stochastic_no_blue_missile": (initial, False, True),
        "checkpoint_deterministic": (trained, True, False),
        "checkpoint_stochastic": (trained, False, False),
        "checkpoint_deterministic_no_blue_missile": (trained, True, True),
    }
    scenarios = {}
    for name, (policy, deterministic, no_blue_missile) in specs.items():
        records = [
            _run_episode(
                config, policy, adapter, deterministic=deterministic,
                no_blue_missile=no_blue_missile, max_steps=max_steps,
                seed=seed + episode, device=device,
            ) for episode in range(episodes)
        ]
        scenarios[name] = {"summary": _scenario_summary(records), "episodes": records}
    result = {
        "config": config, "checkpoint": str(checkpoint),
        "episodes": episodes, "max_steps": max_steps,
        "scenarios": scenarios,
        "launch_quality": _launch_summary(scenarios),
    }
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mav_policy_drift.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = ["# TAM MAV Policy Drift", ""]
    for name, scenario in scenarios.items():
        summary = scenario["summary"]
        lines.extend([
            f"## {name}",
            f"- Survival: {summary['mav_survival_rate']:.3f}",
            f"- Death reasons: {summary['death_reasons']}",
            f"- Dominant bins: {summary['dominant_action_bins']}",
            f"- Mean elevator bin: {summary['mean_elevator_bin']:.2f}",
            f"- Pre-death mean vertical speed: {summary['predeath_100_mean_vertical_speed_mps']:.2f} m/s", "",
        ])
    lines.extend(["## Red launch quality", "", f"```json\n{json.dumps(result['launch_quality'], indent=2)}\n```", ""])
    (out_dir / "mav_policy_drift.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_diagnosis(
        args.config, args.checkpoint, output_dir=args.output_dir,
        episodes=args.episodes, max_steps=args.max_steps,
        device=args.device, seed=args.seed,
    )
    print(json.dumps({
        name: scenario["summary"] for name, scenario in result["scenarios"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
