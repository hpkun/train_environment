"""Diagnose F-22 MAV flight stability envelope for categorical action bins.

Tests fixed actions (no missile, F-22, 1000 steps) to determine which
action combinations cause Crash_LowAlt and which are stable.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_single_action(env_factory, actions: list[int], seed: int, max_steps: int) -> dict:
    env = env_factory()
    obs, info = env.reset(seed=seed)
    action_dict = {rid: np.array(actions, dtype=np.int64) for rid in env.red_ids}

    altitudes = []
    speeds = []
    rolls = []
    pitches = []
    death_step = None
    death_reason = None

    for step in range(max_steps):
        # Check MAV state before step
        mav_sim = env.red_planes.get("red_0")
        if mav_sim is not None and mav_sim.is_alive:
            pos = mav_sim.get_position()
            vel = mav_sim.get_velocity()
            rpy = mav_sim.get_rpy()
            altitudes.append(float(pos[2]))
            speeds.append(float(np.linalg.norm(vel)))
            rolls.append(float(rpy[0]))
            pitches.append(float(rpy[1]))

        next_obs, rewards, terminated, truncated, next_info = env.step(action_dict)

        mav_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
        if not mav_alive and death_step is None:
            death_step = step + 1
            death_reason = env._death_reasons.get("red_0", "unknown")

        if all(terminated.values()) or all(truncated.values()):
            break

        obs, info = next_obs, next_info

    env.close()

    survival = death_step is None
    return {
        "actions": list(actions),
        "survival": survival,
        "death_step": death_step,
        "death_reason": death_reason,
        "final_speed": float(np.mean(speeds[-10:])) if speeds else 0.0,
        "min_altitude": float(np.min(altitudes)) if altitudes else 0.0,
        "final_altitude": float(altitudes[-1]) if altitudes else 0.0,
        "mean_altitude": float(np.mean(altitudes)) if altitudes else 0.0,
        "mean_vertical_speed": float(
            np.mean(np.diff(altitudes)) / 0.2  # 0.2s per env step
        ) if len(altitudes) > 1 else 0.0,
        "max_abs_roll": float(np.max(np.abs(rolls))) if rolls else 0.0,
        "max_abs_pitch": float(np.max(np.abs(pitches))) if pitches else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    parser.add_argument("--output-dir", default="outputs/tam_action_stability_envelope")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds to test per action")
    args = parser.parse_args()

    from uav_env import make_env

    def _make():
        return make_env(
            str(ROOT / args.config),
            env_type="jsbsim_hetero",
            hetero_reward_mode="happo_ref_v0",
            max_steps=args.max_steps,
        )

    # Action grid
    throttles = [37, 38, 39]
    ailerons = [19, 20, 21]
    elevators = [18, 19, 20, 21, 22, 23]
    rudders = [19, 20, 21]

    all_actions = []
    for t in throttles:
        for a in ailerons:
            for e in elevators:
                for r in rudders:
                    all_actions.append([t, a, e, r])

    print(f"Testing {len(all_actions)} action combinations "
          f"× {args.seeds} seeds = {len(all_actions) * args.seeds} runs", flush=True)

    results = []
    for i, actions in enumerate(all_actions):
        for seed in range(args.seeds):
            r = _run_single_action(_make, actions, seed, args.max_steps)
            results.append(r)
        if (i + 1) % 20 == 0:
            survived = sum(1 for r in results if r["survival"])
            print(f"  {i+1}/{len(all_actions)}: {survived}/{len(results)} survived", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "action_stability_envelope.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "actions", "survival", "death_step", "death_reason",
            "final_speed", "min_altitude", "final_altitude", "mean_altitude",
            "mean_vertical_speed", "max_abs_roll", "max_abs_pitch", "mean_speed",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "actions": str(r["actions"]), **{k: v for k, v in r.items() if k != "actions"}
            })

    # Summaries
    all_survived = sum(1 for r in results if r["survival"])
    all_total = len(results)

    # Per-action summary (average across seeds)
    from collections import defaultdict
    by_action = defaultdict(list)
    for r in results:
        by_action[tuple(r["actions"])].append(r)

    action_summaries = []
    for action_tuple, records in sorted(by_action.items()):
        surv = sum(1 for r in records if r["survival"])
        deaths = [r for r in records if not r["survival"]]
        action_summaries.append({
            "actions": list(action_tuple),
            "survival_rate": surv / len(records),
            "death_steps": [r["death_step"] for r in deaths],
            "death_reasons": [r["death_reason"] for r in deaths],
            "min_altitude_mean": float(np.mean([r["min_altitude"] for r in records])),
            "mean_speed": float(np.mean([r["mean_speed"] for r in records])),
        })

    # Key questions
    neutral_actions = [39, 20, 20, 20]
    learned_actions = [38, 20, 22, 21]
    neutral_records = [r for r in results if r["actions"] == neutral_actions]
    learned_records = [r for r in results if r["actions"] == learned_actions]

    neutral_stable = all(r["survival"] for r in neutral_records)
    learned_stable = all(r["survival"] for r in learned_records)

    # Which axis shift is most dangerous
    axis_danger = {}
    for axis_name, axis_idx, neutral_val in [
        ("throttle", 0, 39), ("aileron", 1, 20), ("elevator", 2, 20), ("rudder", 3, 20)
    ]:
        stable_count = 0
        unstable_count = 0
        for action_tuple, records in by_action.items():
            if all(action_tuple[j] == neutral_actions[j] for j in range(4) if j != axis_idx):
                surv = sum(1 for r in records if r["survival"])
                if surv == len(records):
                    stable_count += 1
                else:
                    unstable_count += 1
        axis_danger[axis_name] = {
            "stable_variants": stable_count,
            "unstable_variants": unstable_count,
            "fatal_shift": unstable_count > 0,
        }

    report = {
        "total_actions_tested": len(all_actions),
        "seeds_per_action": args.seeds,
        "total_runs": all_total,
        "overall_survival_rate": all_survived / all_total if all_total else 0,
        "neutral_stable": neutral_stable,
        "learned_stable": learned_stable,
        "neutral_detail": neutral_records,
        "learned_detail": learned_records,
        "axis_danger": axis_danger,
        "action_summaries": action_summaries,
    }

    (out_dir / "action_stability_envelope.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Markdown
    lines = [
        "# Action Stability Envelope Diagnostic",
        "",
        f"Total: {all_total} runs ({len(all_actions)} actions × {args.seeds} seeds)",
        f"Overall survival rate: {all_survived}/{all_total} = {all_survived/all_total:.2%}",
        "",
        "## Key Questions",
        "",
        f"1. [39,20,20,20] stable: **{neutral_stable}**",
    ]
    if neutral_records:
        r = neutral_records[0]
        lines.append(f"   - min_altitude={r['min_altitude']:.0f}m, mean_speed={r['mean_speed']:.0f}m/s")

    lines.append(f"2. [38,20,22,21] crashes: **{not learned_stable}**")
    if learned_records:
        for r in learned_records:
            lines.append(f"   - survival={r['survival']}, death_step={r['death_step']}, "
                        f"death_reason={r['death_reason']}, min_alt={r['min_altitude']:.0f}m")

    lines.extend([
        "",
        "## Axis Danger Assessment",
    ])
    for axis, info in axis_danger.items():
        status = "DANGEROUS" if info["fatal_shift"] else "safe"
        lines.append(f"- **{axis}**: {status} (stable={info['stable_variants']}, unstable={info['unstable_variants']})")

    # All fatal actions
    fatal_actions = [s for s in action_summaries if s["survival_rate"] < 1.0]
    if fatal_actions:
        lines.extend(["", "## Fatal Action Combinations", ""])
        for s in fatal_actions:
            lines.append(f"- {s['actions']}: survival_rate={s['survival_rate']:.0%} "
                        f"death_steps={s['death_steps']} reasons={s['death_reasons']}")

    (out_dir / "action_stability_envelope.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nReports written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
