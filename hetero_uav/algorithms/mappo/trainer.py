from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .policy import ActorCritic


class MAPPOTrainer:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int,
                 hidden_dim: int, lr: float, clip_param: float,
                 value_coef: float, entropy_coef: float, max_grad_norm: float,
                 device: torch.device):
        self.model = ActorCritic(obs_dim, state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def update(self, batch, epochs: int, minibatch_size: int) -> dict[str, float]:
        n = batch.obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                idx = indices[start:start + minibatch_size]
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch.obs[idx], batch.states[idx], batch.actions[idx])
                ratio = torch.exp(log_probs - batch.old_log_probs[idx])
                surr1 = ratio * batch.advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * batch.advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch.returns[idx])
                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                updates += 1
        return {k: v / max(1, updates) for k, v in stats.items()}

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "obs_dim": self.model.obs_dim,
            "state_dim": self.model.state_dim,
            "action_dim": self.model.action_dim,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device, hidden_dim: int = 128):
        payload = torch.load(path, map_location=device)
        trainer = cls(
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=hidden_dim,
            lr=1e-4,
            clip_param=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=10.0,
            device=device,
        )
        trainer.model.load_state_dict(payload["model_state_dict"])
        return trainer, payload
