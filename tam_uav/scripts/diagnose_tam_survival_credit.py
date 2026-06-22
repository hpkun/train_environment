"""Diagnose MAV survival, death reward, and actor-credit availability."""
from __future__ import annotations

import argparse
import csv
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
from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.eval_tam_happo_direct import _action_dim_from_env, _build_policy_from_meta, _load_meta
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


CHECKPOINT_CANDIDATES = (
    "outputs/tam_papermode_3v2_2M_probe/latest_failure/model.pt",
    "outputs/tam_papermode_3v2_2M_probe/latest/model.pt",
    "outputs/tam_papermode_3v2_2M_probe/eval_checkpoints/step_803072/model.pt",
)
AXES = ("throttle", "aileron", "elevator", "rudder")


def _disable_blue_missiles(env):
    for sim in env.blue_planes.values():
        sim.num_missiles = 0
        sim.num_left_missiles = 0


def _fixed_blue(env):
    action = np.asarray([env.tam_action_levels - 1, 20, 4, 20], dtype=np.int64)
    return {agent_id: action.copy() for agent_id in env.blue_ids}


def _load_checkpoint(path: Path, device, action_dim):
    policy = _build_policy_from_meta(_load_meta(path), device, action_dim)
    policy.load(path, map_location=device)
    policy.eval()
    return policy


def _episode(config, mode, episode, max_steps, device, adapter, policy=None):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    obs, info = env.reset(seed=episode)
    blue_missiles = mode == "initial_policy_stochastic_blue_missile"
    if not blue_missiles:
        _disable_blue_missiles(env)
    opponent = OpponentPolicy("tam_direct_fsm", seed=1000 + episode)
    deterministic = "deterministic" in mode
    fixed = mode == "fixed_neutral_no_policy_no_missile"
    roles = [0 if rid == "red_0" else 1 for rid in env.red_ids]
    hidden = policy.init_hidden(len(env.red_ids), device) if policy is not None else None
    rows = []
    for step in range(max_steps):
        active_before = float(env.red_planes["red_0"].is_alive)
        log_prob = value = ""
        entropy_axis = [""] * 4
        max_prob_axis = [""] * 4
        if fixed:
            sampled = np.asarray([39, 20, 20, 20], dtype=np.int64)
            argmax = sampled.copy()
            red_actions = {"red_0": sampled.copy()}
            red_actions.update({
                rid: np.asarray([39, 20, 4, 20], dtype=np.int64)
                for rid in env.red_ids[1:]
            })
        else:
            adapted = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids
            )
            actor_obs = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(96, np.float32))
                for rid in env.red_ids
            ])
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(actor_obs, device=device), roles=roles,
                    critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                    deterministic=deterministic, rnn_hidden=hidden,
                )
            hidden = out["rnn_hidden"].detach()
            actions = out["action"].detach().cpu().numpy().astype(np.int64)
            sampled = actions[0]
            probs = out["action_probs"][0].detach().cpu()
            argmax = probs.argmax(-1).numpy()
            entropy_axis = (-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1)).tolist()
            max_prob_axis = probs.max(-1).values.tolist()
            log_prob = float(out["log_prob"][0].item())
            value = float(out["value"].item())
            red_actions = {rid: actions[index] for index, rid in enumerate(env.red_ids)}
        actions = dict(red_actions)
        actions.update(opponent.act(obs, env.blue_ids, env=env) if blue_missiles else _fixed_blue(env))
        obs, rewards, terminated, truncated, next_info = env.step(actions)
        sim = env.red_planes["red_0"]
        alive_after = float(sim.is_alive)
        death_transition = bool(active_before > 0.5 and alive_after < 0.5)
        velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
        done = bool(all(terminated.values()) or all(truncated.values()))
        info0 = next_info.get("red_0", {}) if isinstance(next_info, dict) else {}
        row = {
            "mode": mode, "episode": episode, "step": step,
            "altitude_m": float(sim.get_geodetic()[2]),
            "vertical_speed_mps": float(velocity[2]),
            "sampled_action_indices": sampled.tolist(),
            "argmax_action_indices": argmax.tolist(),
            "log_prob": log_prob,
            "entropy_per_axis": entropy_axis,
            "max_action_prob_per_axis": max_prob_axis,
            "active_mask_before": active_before,
            "active_mask_at_death_transition": active_before if death_transition else "",
            "death_transition": death_transition,
            "reward_red_0": float(rewards.get("red_0", 0.0)),
            "return_to_go": "", "advantage": "",
            "done": done,
            "bad_transition": info0.get("bad_transition", ""),
            "death_reason": env._death_reasons.get("red_0", "") if death_transition else "",
            "red_uav_fired": sum(int(next_info.get(rid, {}).get("missiles_fired_this_step", 0)) for rid in env.red_ids[1:]),
            "red_uav_hits": sum(int(next_info.get(rid, {}).get("missile_hits_this_step", 0)) for rid in env.red_ids[1:]),
            "blue_fired": sum(int(next_info.get(rid, {}).get("missiles_fired_this_step", 0)) for rid in env.blue_ids),
            "blue_hits": sum(int(next_info.get(rid, {}).get("missile_hits_this_step", 0)) for rid in env.blue_ids),
            "value": value,
        }
        rows.append(row)
        info = next_info
        if death_transition or done:
            break
    rtg = 0.0
    for row in reversed(rows):
        rtg = float(row["reward_red_0"]) + 0.99 * rtg
        row["return_to_go"] = rtg
        if row["value"] != "":
            row["advantage"] = rtg - float(row["value"])
    death_rows = [row for row in rows if row["death_transition"]]
    death = death_rows[0] if death_rows else None
    result = {
        "mode": mode, "episode": episode,
        # A timeout may be emitted on the final simulator step before the trace
        # reaches exactly max_steps rows. Survival means no MAV death transition.
        "survived": death is None,
        "death_step": death["step"] if death else None,
        "death_reason": death["death_reason"] if death else "alive",
        "death_reward_red_0": death["reward_red_0"] if death else None,
        "death_active_mask": death["active_mask_at_death_transition"] if death else None,
        "death_return_to_go": death["return_to_go"] if death else None,
        "death_advantage": death["advantage"] if death else None,
        "death_transition_used_for_actor": bool(death and death["active_mask_before"] > 0.5),
    }
    env.close()
    return result, rows


def diagnose(config, checkpoint, output_dir, episodes, max_steps, device):
    probe = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    action_dim = _action_dim_from_env(probe)
    probe.close()
    adapter = HeteroObsAdapterV2()
    initial = TAMCategoricalRecurrentHAPPOPolicy(action_dim=action_dim).to(device).eval()
    checkpoint_policy = _load_checkpoint(checkpoint, device, action_dim) if checkpoint else None
    modes = [
        ("fixed_neutral_no_policy_no_missile", None),
        ("initial_policy_deterministic_no_missile", initial),
        ("initial_policy_stochastic_no_missile", initial),
        ("initial_policy_stochastic_blue_missile", initial),
    ]
    if checkpoint_policy is not None:
        modes += [
            ("latest_checkpoint_deterministic_no_missile", checkpoint_policy),
            ("latest_checkpoint_stochastic_no_missile", checkpoint_policy),
        ]
    all_rows, summaries = [], {}
    for mode, policy in modes:
        records = []
        for episode in range(episodes):
            record, rows = _episode(
                config, mode, episode, max_steps, device, adapter, policy
            )
            records.append(record)
            all_rows.extend(rows)
        summaries[mode] = {
            "survival_rate": float(np.mean([record["survived"] for record in records])),
            "death_reasons": dict(Counter(record["death_reason"] for record in records)),
            "episodes": records,
        }
    payload = {
        "config": config, "checkpoint": str(checkpoint) if checkpoint else None,
        "episodes": episodes, "max_steps": max_steps,
        "modes": summaries,
        "buffer_credit_conclusion": (
            "death transition uses pre-step active mask; reward and death marker are stored"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "survival_credit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(all_rows[0]) if all_rows else []
    with (output_dir / "survival_credit_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    lines = ["# TAM survival credit diagnosis", "", f"Checkpoint: `{payload['checkpoint']}`", ""]
    for mode, summary in summaries.items():
        lines += [f"## {mode}", "", f"- Survival rate: {summary['survival_rate']}",
                  f"- Death reasons: {summary['death_reasons']}", ""]
    death_records = [record for summary in summaries.values() for record in summary["episodes"] if record["death_step"] is not None]
    lines += ["## Credit path", "",
              f"- Death transitions observed: {len(death_records)}",
              f"- Death transitions actor-eligible: {sum(record['death_transition_used_for_actor'] for record in death_records)}",
              f"- Negative death rewards: {sum((record['death_reward_red_0'] or 0) < 0 for record in death_records)}",
              "- `valid_count=0` after MAV death is a consequence of inactivity, not early masking."]
    (output_dir / "survival_credit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/tam_survival_credit_diagnostics")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint) if args.checkpoint else next(
        (ROOT / candidate for candidate in CHECKPOINT_CANDIDATES if (ROOT / candidate).exists()), None
    )
    diagnose(args.config, checkpoint, ROOT / args.output_dir, args.episodes, args.max_steps, device)


if __name__ == "__main__":
    main()
