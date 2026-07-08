"""Pure-HAPPO update invariant diagnostics.

This script is intentionally synthetic and short-running.  It does not create
an environment or train a policy; it checks PPO accounting invariants that can
explain staged degradation when violated.
"""
from __future__ import annotations

import argparse
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
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.trainer import _compute_grouped_gae

DEFAULT_OUT = ROOT / "outputs" / "audits" / "pure_happo_update_invariants"


def _flatten_params(policy: PureHAPPOPolicy) -> torch.Tensor:
    return torch.cat([p.detach().cpu().flatten() for p in policy.parameters()])


def _finite_float(value: float) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _build_buffer(policy: PureHAPPOPolicy, device: torch.device,
                  steps: int = 12, inactive_agent: int | None = None) -> HAPPORolloutBuffer:
    rng = np.random.default_rng(3107)
    torch.manual_seed(3107)
    buf = HAPPORolloutBuffer(steps, policy.num_agents, 96, 480, 3, [0, 1, 1])
    for t in range(steps):
        actor_obs = rng.normal(0.0, 0.4, size=(policy.num_agents, 96)).astype(np.float32)
        critic_state = rng.normal(0.0, 0.4, size=(480,)).astype(np.float32)
        with torch.no_grad():
            out = policy.act(
                torch.as_tensor(actor_obs, device=device),
                critic_state=torch.as_tensor(critic_state, device=device),
                deterministic=False,
            )
            next_value = float(policy.value(torch.as_tensor(critic_state, device=device))[0].item())
        rewards = rng.normal(0.0, 0.05, size=(policy.num_agents,)).astype(np.float32)
        dones = np.zeros(policy.num_agents, dtype=np.float32)
        if t in {5, steps - 1}:
            dones[:] = 1.0
        active = np.ones(policy.num_agents, dtype=np.float32)
        if inactive_agent is not None:
            active[inactive_agent] = 0.0
        buf.store(
            actor_obs,
            critic_state,
            out["action"].detach().cpu().numpy(),
            out["log_prob"].detach().cpu().numpy(),
            rewards,
            dones,
            float(out["value"].detach().cpu().numpy()[0]),
            active,
            next_value=next_value,
            env_id=t % 2,
        )
    return buf


def _check_grouped_gae() -> tuple[bool, dict]:
    rewards = torch.tensor([1.0, 10.0, 1.0, 10.0])
    values = torch.zeros(4)
    next_values = torch.zeros(4)
    dones = torch.tensor([0.0, 0.0, 1.0, 1.0])
    env_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    advantages, returns = _compute_grouped_gae(
        rewards, values, next_values, dones, env_ids, gamma=1.0, lam=1.0)
    expected = torch.tensor([2.0, 20.0, 1.0, 10.0])
    ok = torch.allclose(advantages, expected) and torch.allclose(returns, expected)
    return bool(ok), {
        "advantages": [float(v) for v in advantages.tolist()],
        "returns": [float(v) for v in returns.tolist()],
        "expected": [float(v) for v in expected.tolist()],
    }


def run_invariants(device: str = "cpu", output_dir: str | Path | None = None) -> dict:
    dev = torch.device(device)
    torch.manual_seed(3107)
    policy = PureHAPPOPolicy(num_agents=3).to(dev)
    buffer = _build_buffer(policy, dev, steps=12)
    data = buffer.get(dev)

    with torch.no_grad():
        replay_lp, entropy, values, _means = policy.evaluate_actions(
            data["actor_obs"], data["critic_state"], data["actions"])
    logprob_diff = (replay_lp - data["old_log_probs"]).abs()
    logprob_replay_pass = bool(float(logprob_diff.max().item()) < 1e-5)
    tanh_action_bounds_pass = bool(torch.isfinite(data["actions"]).all()
                                   and (data["actions"].abs() <= 1.0 + 1e-6).all())

    near_boundary_actions = torch.tensor(
        [[[0.999999, -0.999999, 0.0],
          [0.999, -0.999, 0.5],
          [-0.999999, 0.999999, -0.5]]],
        dtype=torch.float32,
        device=dev,
    )
    with torch.no_grad():
        boundary_lp, boundary_ent, _boundary_v, _ = policy.evaluate_actions(
            data["actor_obs"][:1], data["critic_state"][:1], near_boundary_actions)
    boundary_logprob_finite_pass = bool(torch.isfinite(boundary_lp).all()
                                        and torch.isfinite(boundary_ent).all())

    zero_lr_policy = PureHAPPOPolicy(num_agents=3).to(dev)
    zero_lr_policy.load_state_dict(policy.state_dict())
    before = _flatten_params(zero_lr_policy)
    zero_lr_trainer = PureHAPPOTrainer(
        zero_lr_policy, actor_lr=0.0, critic_lr=0.0, ppo_epochs=1, seed=77)
    zero_lr_metrics = zero_lr_trainer.update(buffer)
    after = _flatten_params(zero_lr_policy)
    zero_lr_delta = float(torch.linalg.vector_norm(after - before).item())
    zero_lr_noop_pass = bool(zero_lr_delta == 0.0)

    inactive_policy = PureHAPPOPolicy(num_agents=3).to(dev)
    inactive_buffer = _build_buffer(inactive_policy, dev, steps=8, inactive_agent=1)
    inactive_metrics = PureHAPPOTrainer(
        inactive_policy, ppo_epochs=1, seed=88).update(inactive_buffer)
    inactive_values = [
        v for k, v in inactive_metrics.items()
        if isinstance(v, (float, int)) and k not in {"last_update_order"}
    ]
    inactive_mask_no_nan_pass = bool(
        inactive_metrics["valid_sample_count_per_agent"][1] == 0
        and inactive_metrics["ratio_after_mean_per_agent"][1] == 0.0
        and all(_finite_float(float(v)) for v in inactive_values)
    )

    grouped_gae_by_env_id_pass, gae_detail = _check_grouped_gae()
    update_metric_keys = sorted(str(k) for k in zero_lr_metrics.keys())
    overall_pass = all([
        logprob_replay_pass,
        tanh_action_bounds_pass,
        boundary_logprob_finite_pass,
        zero_lr_noop_pass,
        inactive_mask_no_nan_pass,
        grouped_gae_by_env_id_pass,
    ])

    summary = {
        "overall_pass": bool(overall_pass),
        "logprob_replay_pass": logprob_replay_pass,
        "max_abs_logprob_replay_diff": float(logprob_diff.max().item()),
        "mean_abs_logprob_replay_diff": float(logprob_diff.mean().item()),
        "tanh_action_bounds_pass": tanh_action_bounds_pass,
        "action_abs_max": float(data["actions"].abs().max().item()),
        "boundary_logprob_finite_pass": boundary_logprob_finite_pass,
        "zero_lr_noop_pass": zero_lr_noop_pass,
        "zero_lr_param_delta_l2": zero_lr_delta,
        "inactive_mask_no_nan_pass": inactive_mask_no_nan_pass,
        "inactive_valid_sample_count_per_agent": inactive_metrics["valid_sample_count_per_agent"],
        "grouped_gae_by_env_id_pass": grouped_gae_by_env_id_pass,
        "grouped_gae_detail": gae_detail,
        "update_metric_keys": update_metric_keys,
        "notes": [
            "Synthetic invariant audit only; no environment rollout and no training run.",
            "A failure here indicates implementation/accounting risk, not reward or scenario difficulty.",
        ],
    }
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = run_invariants(device=args.device, output_dir=args.output_dir)
    status = "PASS" if summary["overall_pass"] else "FAIL"
    print(f"[pure_happo_update_invariants] {status} wrote {Path(args.output_dir) / 'summary.json'}")
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
