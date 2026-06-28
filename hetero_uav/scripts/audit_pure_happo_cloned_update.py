"""Clone-only PPO update diagnostics for Pure-HAPPO checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOTrainer
from algorithms.pure_happo.trainer import _compute_grouped_gae
from scripts.audit_pure_happo_low_level_diagnostics import (
    _alive_mask,
    _default_config,
    _find_run_dir,
    _load_policy,
    _policy_distribution_rows,
    _write_md,
)
from scripts.full_review_audit_utils import explained_variance, pearson_corr, write_csv_rows

DEFAULT_OUT = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_low_level" / "cloned_update"


def _collect_buffer(run_dir: Path, checkpoint: Path, config: str, device: torch.device,
                    episodes: int, max_steps: int) -> tuple[HAPPORolloutBuffer, list[dict]]:
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy, _meta = _load_policy(checkpoint, device)
    env = make_env(config, env_type="jsbsim_hetero")
    adapter = HeteroObsAdapterV2()
    opponent = OpponentPolicy(mode="brma_rule", seed=41)
    buf = HAPPORolloutBuffer(episodes * max_steps, policy.num_agents, 96, 480, 3, [0, 1, 1][:policy.num_agents])
    rows: list[dict] = []
    transition = 0
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=3000 + ep)
            for step in range(max_steps):
                adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
                    for rid in env.red_ids
                ]).astype(np.float32)
                critic = np.asarray(adapted["critic_state"], dtype=np.float32)
                with torch.no_grad():
                    value = float(policy.value(torch.as_tensor(critic, dtype=torch.float32, device=device)).cpu().numpy()[0])
                actions, logp, _dist_rows = _policy_distribution_rows(policy, actor_obs, device, deterministic=False)
                env_actions = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
                env_actions.update(opponent.act(obs, env.blue_ids, env=env))
                next_obs, rewards, terminated, truncated, next_info = env.step(env_actions)
                next_adapted = adapter.adapt_all(next_obs, info=next_info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                with torch.no_grad():
                    next_value = float(policy.value(torch.as_tensor(np.asarray(next_adapted["critic_state"], dtype=np.float32), device=device)).cpu().numpy()[0])
                active = _alive_mask(env)
                done = float(all(terminated.values()) or all(truncated.values()))
                rew = np.asarray([float(rewards.get(rid, 0.0)) for rid in env.red_ids], dtype=np.float32)
                buf.store(actor_obs, critic, actions, logp, rew, np.full(policy.num_agents, done, dtype=np.float32),
                          value, active, next_value=next_value, env_id=0)
                rows.append({
                    "transition_id": transition,
                    "episode": ep,
                    "step": step,
                    "old_log_prob_mean": float(np.mean(logp)),
                    "value": value,
                    "next_value": next_value,
                    "team_done": done,
                    "active_count": float(active.sum()),
                    "team_reward": float((rew * active).sum() / max(active.sum(), 1.0)),
                })
                transition += 1
                obs, info = next_obs, next_info
                if done:
                    break
    finally:
        if hasattr(env, "close"):
            env.close()
    return buf, rows


def _collect_snapshot_buffer(checkpoint: Path, config: str, device: torch.device,
                             samples: int) -> tuple[HAPPORolloutBuffer, list[dict]]:
    policy, _meta = _load_policy(checkpoint, device)
    buf = HAPPORolloutBuffer(samples, policy.num_agents, 96, 480, 3, [0, 1, 1][:policy.num_agents])
    rows: list[dict] = []
    rng = np.random.default_rng(3000)
    for t in range(samples):
        actor_obs = rng.normal(0.0, 0.5, size=(policy.num_agents, 96)).astype(np.float32)
        critic = rng.normal(0.0, 0.5, size=(480,)).astype(np.float32)
        with torch.no_grad():
            value = float(policy.value(torch.as_tensor(critic, dtype=torch.float32, device=device)).cpu().numpy()[0])
        actions, logp, _dist_rows = _policy_distribution_rows(policy, actor_obs, device, deterministic=False)
        rew = rng.normal(0.0, 0.05, size=(policy.num_agents,)).astype(np.float32)
        active = np.ones(policy.num_agents, dtype=np.float32)
        buf.store(actor_obs, critic, actions, logp, rew, np.zeros(policy.num_agents, dtype=np.float32),
                  value, active, next_value=value, env_id=0)
        rows.append({
            "transition_id": t,
            "episode": 0,
            "step": t,
            "old_log_prob_mean": float(np.mean(logp)),
            "value": value,
            "next_value": value,
            "team_done": 0.0,
            "active_count": float(active.sum()),
            "team_reward": float(np.mean(rew)),
            "note": "synthetic_finite_buffer_no_env_step",
        })
    return buf, rows


def _flatten_params(policy) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])


def _norm_delta(before: torch.Tensor, after: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(after - before).item())


def _tensor_stats(name: str, x: torch.Tensor) -> dict:
    x = x.detach().float().flatten()
    return {
        "name": name,
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False).item()) if x.numel() else 0.0,
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
    }


def _audit(run_dir: Path, checkpoint: Path, output_dir: Path, device: torch.device,
           episodes: int, max_steps: int, live_rollout: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / "cloned_update_progress.txt"
    progress.write_text("start\n", encoding="utf-8")
    meta = json.loads((checkpoint.parent / "meta.json").read_text(encoding="utf-8")) if (checkpoint.parent / "meta.json").exists() else {}
    config = _default_config(run_dir, meta)
    progress.write_text(progress.read_text(encoding="utf-8") + "meta\n", encoding="utf-8")
    policy_before, _ = _load_policy(checkpoint, device)
    progress.write_text(progress.read_text(encoding="utf-8") + "policy_before\n", encoding="utf-8")
    policy_clone = copy.deepcopy(policy_before).to(device)
    progress.write_text(progress.read_text(encoding="utf-8") + "policy_clone\n", encoding="utf-8")

    if live_rollout:
        buffer, collection_rows = _collect_buffer(run_dir, checkpoint, config, device, episodes, max_steps)
        collection_mode = "live_rollout"
    else:
        buffer, collection_rows = _collect_snapshot_buffer(checkpoint, config, device, max(8, episodes * max_steps))
        collection_mode = "synthetic_finite_buffer_no_env_step"
    progress.write_text(progress.read_text(encoding="utf-8") + "buffer\n", encoding="utf-8")
    data = buffer.get(device)
    progress.write_text(progress.read_text(encoding="utf-8") + "data\n", encoding="utf-8")
    with torch.no_grad():
        replay_lp, _ent, replay_values, _means = policy_before.evaluate_actions(
            data["actor_obs"], data["critic_state"], data["actions"])
    progress.write_text(progress.read_text(encoding="utf-8") + "replay\n", encoding="utf-8")
    logprob_diff = replay_lp - data["old_log_probs"]
    progress.write_text(progress.read_text(encoding="utf-8") + "logprob_diff\n", encoding="utf-8")
    team_reward = (data["rewards"] * data["active_masks"]).sum(dim=-1) / data["active_masks"].sum(dim=-1).clamp(min=1)
    progress.write_text(progress.read_text(encoding="utf-8") + "team_reward\n", encoding="utf-8")
    team_dones = data["dones"][:, 0].float()
    advantages, returns = _compute_grouped_gae(
        team_reward, data["values"], data["next_values"], team_dones,
        data["env_ids"], gamma=0.99, lam=0.95)
    progress.write_text(progress.read_text(encoding="utf-8") + "gae\n", encoding="utf-8")
    ev = explained_variance(data["values"].detach().cpu().numpy(), returns.detach().cpu().numpy())
    progress.write_text(progress.read_text(encoding="utf-8") + "ev\n", encoding="utf-8")
    corr = pearson_corr(data["values"].detach().cpu().numpy().tolist(), returns.detach().cpu().numpy().tolist())
    progress.write_text(progress.read_text(encoding="utf-8") + "corr\n", encoding="utf-8")

    before_params = _flatten_params(policy_clone)
    progress.write_text(progress.read_text(encoding="utf-8") + "before_params\n", encoding="utf-8")
    trainer = PureHAPPOTrainer(policy_clone, ppo_epochs=1, seed=123)
    metrics = trainer.update(buffer)
    progress.write_text(progress.read_text(encoding="utf-8") + "update\n", encoding="utf-8")
    after_params = _flatten_params(policy_clone)
    progress.write_text(progress.read_text(encoding="utf-8") + "after_params\n", encoding="utf-8")

    with torch.no_grad():
        after_lp, _after_ent, after_values, _after_means = policy_clone.evaluate_actions(
            data["actor_obs"], data["critic_state"], data["actions"])
    progress.write_text(progress.read_text(encoding="utf-8") + "after_eval\n", encoding="utf-8")
    ratio_before = torch.exp(replay_lp - data["old_log_probs"])
    ratio_after = torch.exp(after_lp - data["old_log_probs"])
    clip_fraction = ((ratio_after - 1.0).abs() > 0.2).float()
    approx_kl = (data["old_log_probs"] - after_lp).detach()

    alignment_rows = []
    for idx, row in enumerate(collection_rows):
        alignment_rows.append({
            **row,
            "max_abs_logprob_replay_diff": float(logprob_diff[idx].abs().max().item()),
            "mean_abs_logprob_replay_diff": float(logprob_diff[idx].abs().mean().item()),
            "ratio_before_mean": float(ratio_before[idx].mean().item()),
            "ratio_after_mean": float(ratio_after[idx].mean().item()),
            "clip_fraction_mean": float(clip_fraction[idx].mean().item()),
        })
    write_csv_rows(output_dir / "ppo_buffer_alignment.csv", alignment_rows)
    write_csv_rows(output_dir / "ppo_advantage_diagnostics.csv", [
        _tensor_stats("team_reward", team_reward),
        _tensor_stats("advantage", advantages),
        _tensor_stats("return", returns),
        _tensor_stats("value", data["values"]),
    ])
    write_csv_rows(output_dir / "ppo_ratio_kl_clip_diagnostics.csv", [
        _tensor_stats("ratio_before", ratio_before),
        _tensor_stats("ratio_after", ratio_after),
        _tensor_stats("approx_kl", approx_kl),
        _tensor_stats("clip_fraction", clip_fraction),
    ])
    write_csv_rows(output_dir / "ppo_gradient_update_diagnostics.csv", [{
        "actor_loss_mean": metrics.get("actor_loss_mean", 0.0),
        "critic_loss": metrics.get("critic_loss", 0.0),
        "actor_param_update_norm": _norm_delta(before_params, after_params),
        "action_log_std_mean": metrics.get("action_log_std_mean", 0.0),
        "mav_active_sample_count": metrics.get("mav_active_sample_count", 0),
        "uav_active_sample_count": metrics.get("uav_active_sample_count", 0),
    }])
    write_csv_rows(output_dir / "critic_health_diagnostics.csv", [{
        "explained_variance": ev,
        "value_return_corr": corr if math.isfinite(corr) else 0.0,
        "value_mean_before": float(data["values"].mean().item()),
        "value_mean_after": float(after_values.mean().item()),
        "return_mean": float(returns.mean().item()),
    }])

    replay_max = float(logprob_diff.abs().max().item()) if logprob_diff.numel() else 0.0
    clip_mean = float(clip_fraction.mean().item()) if clip_fraction.numel() else 0.0
    update_norm = _norm_delta(before_params, after_params)
    risks = []
    if replay_max > 1e-5:
        risks.append("CONFIRMED_BUG: old_log_prob replay diff > 1e-5")
    if clip_mean <= 1e-6 and abs(float(ratio_after.mean().item()) - 1.0) < 1e-3 and update_norm < 1e-6:
        risks.append("UPDATE_TOO_WEAK_RISK")
    if clip_mean > 0.5 or abs(float(approx_kl.mean().item())) > 0.2:
        risks.append("UPDATE_UNSTABLE_RISK")
    if ev <= 0 or abs(corr) < 0.1:
        risks.append("CRITIC_UNRELIABLE_RISK")
    if not risks:
        risks.append("no confirmed numerical bug in cloned update")
    _write_md(output_dir / "cloned_update_audit.md", "Cloned PPO Update Audit", "\n".join([
        f"- transitions: {len(buffer)}",
        f"- collection_mode: {collection_mode}",
        f"- max_abs_logprob_replay_diff: {replay_max:.8g}",
        f"- ratio_after_mean: {float(ratio_after.mean().item()):.6g}",
        f"- clip_fraction_mean: {clip_mean:.6g}",
        f"- actor_param_update_norm: {update_norm:.6g}",
        f"- critic_explained_variance: {ev:.6g}",
        f"- value_return_corr: {corr if math.isfinite(corr) else 0.0:.6g}",
        "",
        "## Classification",
        *[f"- {r}" for r in risks],
    ]))
    return {"risks": risks, "transitions": len(buffer), "replay_max": replay_max}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--live-rollout", action="store_true",
                        help="Attempt env.step rollout. Default uses reset snapshots because this audit may run where JSBSim step native-crashes.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir()
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "latest" / "model.pt"
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    _audit(run_dir, checkpoint, Path(args.output_dir), device, args.episodes, args.max_steps, args.live_rollout)


if __name__ == "__main__":
    main()
