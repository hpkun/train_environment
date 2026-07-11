"""Compare Pure HAPPO update hyperparameters on one frozen rollout."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.happo_buffer import HAPPORolloutBuffer  # noqa: E402
from algorithms.mappo.opponent_policy import OpponentPolicy  # noqa: E402
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer  # noqa: E402
from scripts.train_happo_reference import _build_red_alive_mask  # noqa: E402
from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2  # noqa: E402

DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v1.yaml"


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _collect_rollout(config, opponent_mode, length, seed, device):
    _seed(seed)
    env = make_env(config, max_steps=1000)
    adapter = HeteroObsAdapterV2()
    roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]
    policy = PureHAPPOPolicy(
        actor_obs_dim=adapter.flat_actor_obs_dim,
        critic_state_dim=adapter.critic_state_dim,
        action_dim=3, num_agents=len(env.red_ids),
    ).to(device)
    initial_state = copy.deepcopy(policy.state_dict())
    buffer = HAPPORolloutBuffer(length, len(env.red_ids), adapter.flat_actor_obs_dim,
                                adapter.critic_state_dim, 3, roles)
    opponent = OpponentPolicy(opponent_mode, seed=seed + 1000)
    try:
        obs, info = env.reset(seed=seed); opponent.reset_memory()
        for _ in range(length):
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs = np.stack([adapted["actor_obs"][rid] for rid in env.red_ids])
            critic = adapted["critic_state"]
            active = _build_red_alive_mask(info, env, env.red_ids)
            with torch.no_grad():
                out = policy.act(torch.as_tensor(actor_obs, device=device), critic_state=torch.as_tensor(critic, device=device))
            red_actions = {rid: out["action"][i].cpu().numpy() for i, rid in enumerate(env.red_ids)}
            blue_actions = opponent.act(obs, env.blue_ids, deterministic=True, env=env)
            next_obs, rewards, terminated, truncated, next_info = env.step({**red_actions, **blue_actions})
            reward_np = np.asarray([rewards[rid] for rid in env.red_ids], np.float32)
            done_np = np.asarray([float(terminated[rid] or truncated[rid]) for rid in env.red_ids], np.float32)
            next_adapted = adapter.adapt_all(next_obs, info=next_info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            with torch.no_grad():
                next_value = float(policy.value(torch.as_tensor(next_adapted["critic_state"], device=device)).item())
            buffer.store(actor_obs, critic, out["action"].cpu().numpy(), out["log_prob"].cpu().numpy(),
                         reward_np, done_np, float(out["value"].item()), active,
                         next_value=next_value, env_id=0)
            obs, info = next_obs, next_info
            if all(terminated.values()) or all(truncated.values()):
                obs, info = env.reset(seed=seed + len(buffer)); opponent.reset_memory()
        return buffer, initial_state, {
            "actor_obs_dim": adapter.flat_actor_obs_dim,
            "critic_state_dim": adapter.critic_state_dim,
            "num_agents": len(env.red_ids), "roles": roles,
        }
    finally:
        env.close()


def _finite_metrics(stats):
    count = 0
    for value in stats.values():
        values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
        for item in values:
            if isinstance(item, (int, float, np.integer, np.floating)) and not math.isfinite(float(item)):
                count += 1
    return count


def _run_candidate(buffer, initial_state, spec, actor_lr, ppo_epochs, seed, device):
    _seed(seed)
    policy = PureHAPPOPolicy(spec["actor_obs_dim"], spec["critic_state_dim"], 3, spec["num_agents"]).to(device)
    policy.load_state_dict(copy.deepcopy(initial_state))
    trainer = PureHAPPOTrainer(
        policy, actor_lr=actor_lr, critic_lr=5e-4, clip_param=.2,
        entropy_coef=.01, value_coef=.5, max_grad_norm=10.0,
        ppo_epochs=ppo_epochs, critic_epochs=5, gamma=.99, gae_lambda=.95,
        seed=seed,
    )
    stats = trainer.update(copy.deepcopy(buffer))
    row = {
        "actor_lr": actor_lr, "ppo_epochs": ppo_epochs,
        "actor_loss_mav": stats["actor_loss_mav"], "actor_loss_uav": stats["actor_loss_uav"],
        "critic_loss_scaled": stats["critic_loss_scaled"], "critic_loss_unscaled": stats["critic_loss_unscaled"],
        "explained_variance_old": stats["value_explained_variance_old"],
        "explained_variance_new": stats["value_explained_variance_new"],
        "KL_mav": stats["approx_kl_mav"], "KL_uav": stats["approx_kl_uav"],
        "clip_fraction_mav": stats["clip_fraction_mav"], "clip_fraction_uav": stats["clip_fraction_uav"],
        "actor_grad_mav": stats["actor_grad_norm_mav"], "actor_grad_uav": stats["actor_grad_norm_uav"],
        "critic_grad": stats["critic_grad_norm"], "actor_update_norm_mav": stats["policy_update_norm_mav"],
        "actor_update_norm_uav": stats["policy_update_norm_uav"], "critic_update_norm": stats["critic_update_norm"],
        "entropy_mav": stats["entropy_mav"], "entropy_uav": stats["entropy_uav"],
        "action_log_std_mav": stats["action_log_std_mav_mean"], "action_log_std_uav": stats["action_log_std_uav_mean"],
        "action_saturation_mav": stats["mav_action_saturation_rate"], "action_saturation_uav": stats["uav_action_saturation_rate"],
        "advantage_raw_mean": stats["advantage_raw_mean"], "advantage_raw_std": stats["advantage_raw_std"],
        "return_target_mean": stats["return_mean"], "return_target_std": stats["return_std"],
        "last_update_order": json.dumps([int(v) for v in stats["last_update_order"]]),
        "nonfinite_count": _finite_metrics(stats),
    }
    for prefix in ("ratio_mean", "ratio_std", "ratio_p95", "ratio_p99"):
        row[f"{prefix}_mav"] = stats[f"{prefix}_mav"]
        row[f"{prefix}_uav"] = stats[f"{prefix}_uav"]
    for key in ("m_mean_after_each_agent", "m_std_after_each_agent", "m_abs_mean_after_each_agent", "m_abs_max_after_each_agent"):
        row[key] = json.dumps([float(v) for v in stats[key]])
    for key in ("reward_nan_count", "action_nan_count", "value_nan_count", "log_prob_nan_count", "gradient_nonfinite_count"):
        row[key] = stats.get(key, 0.0)
    return row


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/pure_happo_scale_aligned_v1_update_sweep")
    args = parser.parse_args()
    if args.num_envs != 1: raise ValueError("stability audit requires num_envs=1")
    device = torch.device(args.device)
    buffer, initial_state, spec = _collect_rollout(args.config, args.opponent_policy, args.rollout_length, args.seed, device)
    rows = [_run_candidate(buffer, initial_state, spec, lr, epochs, args.seed, device)
            for lr in (5e-4, 2e-4, 1e-4) for epochs in (3, 5)]
    baseline = next(r for r in rows if r["actor_lr"] == 5e-4 and r["ppo_epochs"] == 5)
    eligible = []
    for row in rows:
        finite = row["nonfinite_count"] == 0
        updates = min(row["actor_update_norm_mav"], row["actor_update_norm_uav"], row["critic_update_norm"]) > 0
        ev_ok = row["explained_variance_new"] >= baseline["explained_variance_new"] - .05
        kl_ok = max(abs(row["KL_mav"]), abs(row["KL_uav"])) <= .75 * max(abs(baseline["KL_mav"]), abs(baseline["KL_uav"]))
        clip_ok = max(row["clip_fraction_mav"], row["clip_fraction_uav"]) <= .75 * max(baseline["clip_fraction_mav"], baseline["clip_fraction_uav"])
        entropy_ok = min(row["entropy_mav"], row["entropy_uav"]) > .1
        row["recommendation_eligible"] = bool(finite and updates and ev_ok and kl_ok and clip_ok and entropy_ok)
        if row["recommendation_eligible"]: eligible.append(row)
    recommended = min(eligible, key=lambda r: max(r["KL_mav"], r["KL_uav"]) + max(r["clip_fraction_mav"], r["clip_fraction_uav"])) if eligible else None
    summary = {
        "rollout_length": len(buffer), "same_initial_state_dict": True,
        "same_rollout_for_all_candidates": True, "critic_lr": 5e-4, "critic_epochs": 5,
        "baseline": baseline, "recommended": None if recommended is None else {
            "actor_lr": recommended["actor_lr"], "ppo_epochs": recommended["ppo_epochs"]
        },
        "verdict": "CANDIDATE_FOUND" if recommended else "PURE_HAPPO_UPDATE_STABILITY_NOT_RESOLVED",
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "pure_happo_update_sweep.csv", rows)
    (out / "pure_happo_update_sweep.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    report = [
        "# Pure HAPPO update stability sweep", "",
        "- Same rollout: `True`", "- Same initial state dict: `True`",
        f"- Verdict: `{summary['verdict']}`", f"- Recommended: `{summary['recommended']}`",
        "", "| actor_lr | epochs | KL MAV/UAV | clip MAV/UAV | EV new | eligible |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    report.extend(
        f"| {r['actor_lr']:.1e} | {r['ppo_epochs']} | {r['KL_mav']:.6f}/{r['KL_uav']:.6f} | "
        f"{r['clip_fraction_mav']:.6f}/{r['clip_fraction_uav']:.6f} | "
        f"{r['explained_variance_new']:.6f} | {r['recommendation_eligible']} |"
        for r in rows
    )
    (out / "pure_happo_update_sweep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__": main()
