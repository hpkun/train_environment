"""Validate flight behavior of the randomly initialized categorical TAM policy."""
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
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_hidden
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


ELEVATOR_SWEEP_BINS = (6, 8, 10, 11, 12, 14, 16, 18, 20)


def summarize_flight_trace(trace, *, death_reason, death_step):
    altitudes = [float(row["altitude_m"]) for row in trace]
    vertical_speeds = [float(row["vertical_speed_mps"]) for row in trace]
    speeds = [float(row["speed_mps"]) for row in trace]
    return {
        "steps_recorded": len(trace),
        "final_altitude_m": altitudes[-1] if altitudes else float("nan"),
        "min_altitude_m": min(altitudes) if altitudes else float("nan"),
        "mean_vertical_speed_mps": float(np.mean(vertical_speeds)) if vertical_speeds else float("nan"),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else float("nan"),
        "death_reason": death_reason,
        "death_step": int(death_step),
    }


def choose_stable_elevator(sweep):
    stable = [
        (int(action_bin), float(record["min_altitude_m"]))
        for action_bin, record in sweep.items()
        if record.get("survived_1000")
    ]
    if not stable:
        crashed = [
            (int(action_bin), int(record.get("death_step", -1)),
             float(record.get("min_altitude_m", float("-inf"))))
            for action_bin, record in sweep.items()
        ]
        return max(crashed, key=lambda item: (item[1], item[2]))[0] if crashed else None
    return max(stable, key=lambda item: item[1])[0]


def _alive_mask(env):
    return np.asarray([
        float(env.red_planes[rid].is_alive) for rid in env.red_ids
    ], dtype=np.float32)


def _fixed_blue_actions(env):
    neutral = np.asarray([env.tam_action_levels - 1, 20, 0, 20], dtype=np.int64)
    return {bid: neutral.copy() for bid in env.blue_ids}


def _disable_blue_missiles(env):
    for sim in env.blue_planes.values():
        sim.num_missiles = 0
        sim.num_left_missiles = 0


def _mav_trace_row(env, action, *, action_probs_max="", expected_action=None):
    sim = env.red_planes["red_0"]
    geodetic = sim.get_geodetic()
    velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
    roll, pitch, _yaw = sim.get_rpy()
    command = env._last_tam_action_commands.get("red_0", {})
    return {
        "step": int(env.current_step),
        "altitude_m": float(geodetic[2]),
        "speed_mps": float(np.linalg.norm(velocity)),
        "pitch_rad": float(pitch),
        "roll_rad": float(roll),
        "vertical_speed_mps": float(velocity[2]),
        "action_indices": np.asarray(action, dtype=np.int64).tolist(),
        "action_probs_max": (float(action_probs_max) if action_probs_max != "" else ""),
        "expected_normalized_action": (
            np.asarray(expected_action, dtype=np.float64).tolist()
            if expected_action is not None else []
        ),
        "fcs_command": {
            key: command.get(key, "") for key in (
                "throttle_cmd_norm", "aileron_cmd_norm",
                "elevator_cmd_norm", "rudder_cmd_norm",
            )
        },
        "calibration_profile": command.get("calibration_profile", {}),
        "written_fcs_paths": command.get("written_fcs_paths", []),
        "readback_values": command.get("readback_values", {}),
    }


def _run_policy_trace(config, policy, adapter, *, deterministic, max_steps,
                      seed, device, formal_blue=False, no_blue_missile=True):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    obs, info = env.reset(seed=seed)
    if no_blue_missile:
        _disable_blue_missiles(env)
    opponent = OpponentPolicy("tam_direct_fsm", seed=seed + 1000)
    roles = [0 if rid == "red_0" else 1 for rid in env.red_ids]
    hidden = policy.init_hidden(len(env.red_ids), torch.device(device))
    trace = []
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
        action_dict.update(
            opponent.act(obs, env.blue_ids, env=env)
            if formal_blue else _fixed_blue_actions(env)
        )
        obs, _rewards, _terminated, _truncated, info = env.step(action_dict)
        trace.append(_mav_trace_row(
            env, actions[0],
            action_probs_max=out["action_probs"][0].max(-1).values.mean().item(),
            expected_action=out["mean"][0].detach().cpu().numpy(),
        ))
        if not env.red_planes["red_0"].is_alive:
            break
    alive = bool(env.red_planes["red_0"].is_alive)
    reason = env._death_reasons.get("red_0") or ("alive" if alive else "unknown")
    death_step = -1 if alive else int(env.current_step)
    summary = summarize_flight_trace(trace, death_reason=reason, death_step=death_step)
    summary["survived_1000"] = bool(alive and len(trace) >= max_steps)
    summary["trace"] = trace
    env.close()
    return summary


def _run_fixed_trace(config, *, mav_elevator_bin, max_steps, seed):
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
    env.reset(seed=seed)
    _disable_blue_missiles(env)
    trace = []
    mav_action = np.asarray([39, 20, mav_elevator_bin, 20], dtype=np.int64)
    uav_action = np.asarray([39, 20, 4, 20], dtype=np.int64)
    for _step in range(max_steps):
        actions = {"red_0": mav_action.copy()}
        actions.update({rid: uav_action.copy() for rid in env.red_ids[1:]})
        actions.update(_fixed_blue_actions(env))
        env.step(actions)
        trace.append(_mav_trace_row(env, mav_action))
        if not env.red_planes["red_0"].is_alive:
            break
    alive = bool(env.red_planes["red_0"].is_alive)
    reason = env._death_reasons.get("red_0") or ("alive" if alive else "unknown")
    death_step = -1 if alive else int(env.current_step)
    summary = summarize_flight_trace(trace, death_reason=reason, death_step=death_step)
    summary["survived_1000"] = bool(alive and len(trace) >= max_steps)
    summary["trace"] = trace
    env.close()
    return summary


def run_long_horizon_validation(config, *, output_dir, episodes=10,
                                max_steps=1000, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = TAMCategoricalRecurrentHAPPOPolicy().to(torch.device(device))
    policy.eval()
    adapter = HeteroObsAdapterV2()
    result = {
        "config": config, "episodes": int(episodes), "max_steps": int(max_steps),
        "initial_policy": {}, "fixed_neutral": [], "f22_elevator_sweep": {},
        "scenarios": {},
    }
    for mode in ("deterministic", "stochastic"):
        records = [
            _run_policy_trace(
                config, policy, adapter, deterministic=(mode == "deterministic"),
                max_steps=max_steps, seed=seed + episode, device=device,
            )
            for episode in range(episodes)
        ]
        result["initial_policy"][mode] = {
            "mav_survival_rate": float(np.mean([record["survived_1000"] for record in records])),
            "death_reasons": dict(Counter(record["death_reason"] for record in records)),
            "episodes": records,
        }
        result["scenarios"][f"initial_{mode}_no_missile"] = result["initial_policy"][mode]
    result["fixed_neutral"] = [
        _run_fixed_trace(
            config, mav_elevator_bin=policy.neutral_action_centers_mav[2],
            max_steps=max_steps,
            seed=seed + episode,
        ) for episode in range(episodes)
    ]
    result["scenarios"]["fixed_calibrated_neutral"] = {
        "mav_survival_rate": float(np.mean([
            record["survived_1000"] for record in result["fixed_neutral"]
        ])),
        "death_reasons": dict(Counter(
            record["death_reason"] for record in result["fixed_neutral"]
        )),
        "episodes": result["fixed_neutral"],
    }
    for name, no_blue_missile in (
        ("formal_blue_missiles_disabled", True),
        ("formal_blue_missiles_enabled", False),
    ):
        records = [
            _run_policy_trace(
                config, policy, adapter, deterministic=True,
                max_steps=max_steps, seed=seed + episode, device=device,
                formal_blue=True, no_blue_missile=no_blue_missile,
            ) for episode in range(episodes)
        ]
        result["scenarios"][name] = {
            "mav_survival_rate": float(np.mean([
                record["survived_1000"] for record in records
            ])),
            "death_reasons": dict(Counter(
                record["death_reason"] for record in records
            )),
            "episodes": records,
        }
    for elevator_bin in ELEVATOR_SWEEP_BINS:
        record = _run_fixed_trace(
            config, mav_elevator_bin=elevator_bin,
            max_steps=max_steps, seed=seed,
        )
        result["f22_elevator_sweep"][str(elevator_bin)] = record
    result["selected_elevator_bin"] = choose_stable_elevator(
        result["f22_elevator_sweep"]
    )

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "initial_policy_1000step.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (out_dir / "tam_initial_policy_long_horizon.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = ["# TAM MAV Long-Horizon Flight Stability", ""]
    for mode, record in result["initial_policy"].items():
        lines.extend([
            f"## Initial policy: {mode}",
            f"- MAV survival: {record['mav_survival_rate']:.3f}",
            f"- Death reasons: {record['death_reasons']}", "",
        ])
    lines.extend(["## Fixed F22 elevator sweep", "", "| Bin | Survived | Death step | Min altitude | Final altitude | Mean vertical speed |", "|---:|:---:|---:|---:|---:|---:|"])
    for action_bin, record in result["f22_elevator_sweep"].items():
        lines.append(
            f"| {action_bin} | {record['survived_1000']} | {record['death_step']} | "
            f"{record['min_altitude_m']:.1f} | {record['final_altitude_m']:.1f} | "
            f"{record['mean_vertical_speed_mps']:.2f} |"
        )
    lines.extend(["", f"Selected elevator bin: `{result['selected_elevator_bin']}`", ""])
    (out_dir / "initial_policy_1000step.md").write_text("\n".join(lines), encoding="utf-8")
    scenario_lines = ["# TAM Initial Policy Long-Horizon Audit", ""]
    for name, scenario in result["scenarios"].items():
        scenario_lines.extend([
            f"## {name}", "",
            f"- MAV survival: `{scenario['mav_survival_rate']:.3f}`",
            f"- Death reasons: `{scenario['death_reasons']}`", "",
        ])
    (out_dir / "tam_initial_policy_long_horizon.md").write_text(
        "\n".join(scenario_lines), encoding="utf-8"
    )
    with (out_dir / "tam_initial_policy_long_horizon_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "scenario", "episode", "step", "altitude_m", "speed_mps",
            "pitch_rad", "roll_rad", "vertical_speed_mps", "action_indices",
            "fcs_command", "calibration_profile", "written_fcs_paths",
            "readback_values",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_name, scenario in result["scenarios"].items():
            for episode_index, episode in enumerate(scenario["episodes"]):
                for row in episode["trace"]:
                    payload = {key: row.get(key, "") for key in fieldnames}
                    payload["scenario"] = scenario_name
                    payload["episode"] = episode_index
                    for key in (
                        "action_indices", "fcs_command", "calibration_profile",
                        "written_fcs_paths", "readback_values",
                    ):
                        payload[key] = json.dumps(payload[key])
                    writer.writerow(payload)
    return result


def run_validation(
    config: str, *, output_dir: str | Path, episodes: int = 10,
    steps: int = 300, device: str = "cpu",
    modes=("stochastic", "deterministic"), seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = TAMCategoricalRecurrentHAPPOPolicy().to(torch.device(device))
    policy.eval()
    adapter = HeteroObsAdapterV2()
    result = {
        "config": config, "episodes_per_mode": int(episodes),
        "steps_per_episode": int(steps),
        "neutral_action_centers": policy.neutral_action_centers,
        "neutral_action_init_std_bins": policy.neutral_action_init_std_bins,
        "modes": {},
    }
    for mode in modes:
        deterministic = mode == "deterministic"
        episode_rows = []
        all_actions = []
        death_reasons = Counter()
        for episode in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max(steps, 300))
            obs, info = env.reset(seed=seed + episode)
            _disable_blue_missiles(env)
            roles = [0 if rid == "red_0" else 1 for rid in env.red_ids]
            hidden = policy.init_hidden(len(env.red_ids), torch.device(device))
            override_detected = False
            executed_steps = 0
            for step in range(steps):
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
                red_actions = out["action"].detach().cpu().numpy().astype(np.int64)
                all_actions.append(red_actions.copy())
                hidden_np = zero_inactive_hidden(
                    out["rnn_hidden"].detach().cpu().numpy(), active
                )
                hidden = torch.as_tensor(hidden_np, device=device)
                action_dict = {
                    rid: red_actions[index].copy()
                    for index, rid in enumerate(env.red_ids)
                }
                action_dict.update(_fixed_blue_actions(env))
                obs, _rewards, terminated, truncated, info = env.step(action_dict)
                executed_steps = step + 1
                for index, rid in enumerate(env.red_ids):
                    effective = np.asarray(env._last_effective_actions.get(rid, []))
                    if not np.array_equal(effective, red_actions[index]):
                        override_detected = True
                if all(terminated.values()) or all(truncated.values()):
                    break
            mav_alive = bool(env.red_planes["red_0"].is_alive)
            uav_alive = [bool(env.red_planes[rid].is_alive) for rid in env.red_ids[1:]]
            reason = env._death_reasons.get("red_0") or ("alive" if mav_alive else "unknown")
            death_reasons[reason] += 1
            episode_rows.append({
                "episode": episode, "env_steps": executed_steps,
                "mav_alive": mav_alive,
                "red_uav_survival_rate": float(np.mean(uav_alive)) if uav_alive else 1.0,
                "mav_death_reason": reason,
                "red_action_override_detected": override_detected,
            })
            env.close()
        actions = np.concatenate(all_actions, axis=0)
        result["modes"][mode] = {
            "mav_survival_rate": float(np.mean([row["mav_alive"] for row in episode_rows])),
            "red_uav_survival_rate": float(np.mean([row["red_uav_survival_rate"] for row in episode_rows])),
            "death_reasons": dict(death_reasons),
            "throttle_high_rate": float(np.mean(actions[:, 0] >= policy.action_levels - 4)),
            "surface_middle_rate": float(np.mean((actions[:, [1, 3]] >= 12) & (actions[:, [1, 3]] <= 28))),
            "elevator_bin_mean": float(np.mean(actions[:, 2])),
            "action_bin_usage": [int(np.unique(actions[:, axis]).size) for axis in range(4)],
            "episodes": episode_rows,
        }

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tam_categorical_initial_policy_flight_validation.json"
    md_path = out_dir / "tam_categorical_initial_policy_flight_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# TAM Categorical Initial Policy Flight Validation", ""]
    for mode, record in result["modes"].items():
        lines.extend([
            f"## {mode}",
            f"- MAV survival: {record['mav_survival_rate']:.3f}",
            f"- Red UAV survival: {record['red_uav_survival_rate']:.3f}",
            f"- Death reasons: {record['death_reasons']}",
            f"- Throttle high-bin rate: {record['throttle_high_rate']:.3f}",
            f"- Surface middle-bin rate: {record['surface_middle_rate']:.3f}",
            f"- Per-axis bin usage: {record['action_bin_usage']}", "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps is not None:
        result = run_long_horizon_validation(
            args.config, output_dir=args.output_dir, episodes=args.episodes,
            max_steps=args.max_steps, device=args.device, seed=args.seed,
        )
    else:
        result = run_validation(
            args.config, output_dir=args.output_dir, episodes=args.episodes,
            steps=args.steps, device=args.device, seed=args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
