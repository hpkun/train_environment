"""Behavior-clone the shared UAV actor from direct-chase oracle samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy


DEFAULT_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"
DEFAULT_INIT = "outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/best/model.pt"
DEFAULT_MODEL = "outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt"
DEFAULT_META = "outputs/oracle_pretrain/uav_actor_oracle_pretrained/meta.json"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    obs = np.asarray(data["actor_obs"], dtype=np.float32)
    act = np.asarray(data["oracle_action"], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 96:
        raise ValueError(f"actor_obs must have shape [N,96], got {obs.shape}")
    if act.ndim != 2 or act.shape[1] != 3:
        raise ValueError(f"oracle_action must have shape [N,3], got {act.shape}")
    return obs, np.clip(act, -1.0, 1.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain HAPPO shared UAV actor from oracle data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--output-checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output-meta", default=DEFAULT_META)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = _rel(args.dataset)
    init_checkpoint = _rel(args.init_checkpoint)
    obs_np, act_np = _load_dataset(dataset)
    device = torch.device(args.device)
    policy = HAPPOReferencePolicy(96, 480).to(device)
    policy.load(init_checkpoint, map_location=device)

    for p in policy.parameters():
        p.requires_grad = False
    for p in policy.uav_actor.parameters():
        p.requires_grad = True
    policy.action_log_std_uav.requires_grad = True

    optimizer = torch.optim.Adam(
        list(policy.uav_actor.parameters()) + [policy.action_log_std_uav],
        lr=args.lr,
    )
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    action = torch.as_tensor(act_np, dtype=torch.float32, device=device)
    n = obs.shape[0]
    split = max(1, int(n * 0.9))
    indices = np.arange(n)
    final_train_loss = final_val_loss = 0.0
    for _epoch in range(args.epochs):
        np.random.shuffle(indices)
        train_idx = indices[:split]
        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[start:start + args.batch_size]
            pred = torch.clamp(policy.uav_actor(obs[batch_idx]), -0.999, 0.999)
            loss = F.mse_loss(pred, action[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_train_loss = float(loss.item())
        with torch.no_grad():
            val_idx = indices[split:] if split < n else indices[:split]
            val_pred = torch.clamp(policy.uav_actor(obs[val_idx]), -0.999, 0.999)
            final_val_loss = float(F.mse_loss(val_pred, action[val_idx]).item())

    out = _rel(args.output_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    meta = {
        "pretrained_from_oracle": True,
        "frozen_mav_actor": True,
        "frozen_critic": True,
        "dataset": str(dataset),
        "init_checkpoint": str(init_checkpoint),
        "num_samples": int(n),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "actor_obs_dim": 96,
        "critic_state_dim": 480,
        "attention": False,
        "recurrent": False,
    }
    meta_path = _rel(args.output_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"output_checkpoint: {out}")
    print(f"output_meta: {meta_path}")
    print(f"final_train_loss: {final_train_loss:.6f}")
    print(f"final_val_loss: {final_val_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

