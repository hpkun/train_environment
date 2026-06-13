"""Check whether the oracle-pretrained UAV actor matches oracle actions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy


DEFAULT_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"
DEFAULT_CKPT = "outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt"
DEFAULT_JSON = "outputs/oracle_pretrain/action_match/action_match.json"
DEFAULT_MD = "outputs/oracle_pretrain/action_match/action_match.md"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_meta(checkpoint: Path) -> dict:
    meta = checkpoint.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _minmax(arr: np.ndarray) -> list[float]:
    return [float(np.min(arr)), float(np.max(arr))]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-8
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.sum(a[valid] * b[valid], axis=1) / denom[valid]))


def _wrapped_heading_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.remainder(pred - target + 1.0, 2.0) - 1.0


def _oracle_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = pred - target
    err = err.copy()
    err[:, 1] = _wrapped_heading_error(pred[:, 1], target[:, 1])
    return err


def evaluate(args) -> dict:
    dataset = _rel(args.dataset)
    checkpoint = _rel(args.checkpoint)
    data = np.load(dataset, allow_pickle=True)
    actor_obs = np.asarray(data["actor_obs"], dtype=np.float32)
    oracle_action = np.asarray(data["oracle_action"], dtype=np.float32)
    if args.max_samples > 0 and actor_obs.shape[0] > args.max_samples:
        actor_obs = actor_obs[:args.max_samples]
        oracle_action = oracle_action[:args.max_samples]

    device = torch.device(args.device)
    torch.manual_seed(0)
    before = HAPPOReferencePolicy(96, 480).to(device)
    before_state = {k: v.detach().cpu().clone() for k, v in before.state_dict().items()}
    policy = HAPPOReferencePolicy(96, 480).to(device)
    raw_state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(raw_state)
    policy.eval()

    obs_t = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
    roles = torch.ones(obs_t.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        mean, std = policy._means_and_stds(obs_t, roles)
        dist = torch.distributions.Normal(mean, std)
        sampled = torch.clamp(dist.sample(), -1.0, 1.0)
    mean_np = mean.detach().cpu().numpy()
    sampled_np = sampled.detach().cpu().numpy()
    err = mean_np - oracle_action
    wrapped_err = _oracle_error(mean_np, oracle_action)

    uav_delta = 0.0
    mav_delta = 0.0
    critic_delta = 0.0
    for key, value in policy.state_dict().items():
        delta = float(torch.norm(value.detach().cpu() - before_state[key]).item())
        if key.startswith("uav_actor") or key == "action_log_std_uav":
            uav_delta += delta
        elif key.startswith("mav_actor") or key == "action_log_std_mav":
            mav_delta += delta
        elif key.startswith("critic"):
            critic_delta += delta

    meta = _load_meta(checkpoint)
    result = {
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "num_samples": int(actor_obs.shape[0]),
        "mse_mean_action_vs_oracle": float(np.mean(err ** 2)),
        "mae_mean_action_vs_oracle": float(np.mean(np.abs(err))),
        "wrapped_mse_mean_action_vs_oracle": float(np.mean(wrapped_err ** 2)),
        "wrapped_mae_mean_action_vs_oracle": float(np.mean(np.abs(wrapped_err))),
        "cosine_similarity": _cosine(mean_np, oracle_action),
        "action_dim_mean_error": np.mean(err, axis=0).astype(float).tolist(),
        "action_dim_std_error": np.std(err, axis=0).astype(float).tolist(),
        "wrapped_action_dim_mean_error": np.mean(wrapped_err, axis=0).astype(float).tolist(),
        "wrapped_action_dim_std_error": np.std(wrapped_err, axis=0).astype(float).tolist(),
        "policy_mean_min_max": _minmax(mean_np),
        "policy_log_std": policy.action_log_std_uav.detach().cpu().numpy().astype(float).tolist(),
        "sampled_action_min_max": _minmax(sampled_np),
        "dataset_action_min_max": _minmax(oracle_action),
        "checkpoint_has_uav_actor": any(k.startswith("uav_actor") for k in raw_state),
        "checkpoint_has_mav_actor": any(k.startswith("mav_actor") for k in raw_state),
        "checkpoint_has_critic": any(k.startswith("critic") for k in raw_state),
        "uav_actor_parameter_delta_after_load": uav_delta,
        "mav_actor_parameter_delta_after_load": mav_delta,
        "critic_parameter_delta_after_load": critic_delta,
        "mav_actor_frozen_during_pretrain": bool(meta.get("frozen_mav_actor", False)),
        "critic_frozen_during_pretrain": bool(meta.get("frozen_critic", False)),
        "actor_obs_dim": int(actor_obs.shape[1]),
        "action_dim": int(oracle_action.shape[1]),
        "load_ok": True,
        "action_match_ok": bool(np.mean(wrapped_err ** 2) < args.max_mse),
    }
    return result


def _write_md(path: Path, result: dict) -> None:
    lines = [
        "# Oracle Pretrain Action Match",
        "",
        f"- checkpoint_has_uav_actor: {result['checkpoint_has_uav_actor']}",
        f"- uav_actor_parameter_delta_after_load: {result['uav_actor_parameter_delta_after_load']:.6f}",
        f"- mse_mean_action_vs_oracle: {result['mse_mean_action_vs_oracle']:.6f}",
        f"- wrapped_mse_mean_action_vs_oracle: {result['wrapped_mse_mean_action_vs_oracle']:.6f}",
        f"- mae_mean_action_vs_oracle: {result['mae_mean_action_vs_oracle']:.6f}",
        f"- cosine_similarity: {result['cosine_similarity']:.6f}",
        f"- policy_log_std: {result['policy_log_std']}",
        f"- action_match_ok: {result['action_match_ok']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pretrained UAV actor output with oracle actions")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory alias that writes action_match.json and action_match.md.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--max-mse", type=float, default=0.2)
    args = parser.parse_args()

    result = evaluate(args)
    if args.output_dir:
        output_dir = _rel(args.output_dir)
        out_json = output_dir / "action_match.json"
        out_md = output_dir / "action_match.md"
    else:
        out_json = _rel(args.output_json)
        out_md = _rel(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(out_md, result)
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")
    print(f"mse_mean_action_vs_oracle: {result['mse_mean_action_vs_oracle']:.6f}")
    print(f"wrapped_mse_mean_action_vs_oracle: {result['wrapped_mse_mean_action_vs_oracle']:.6f}")
    print(f"cosine_similarity: {result['cosine_similarity']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
