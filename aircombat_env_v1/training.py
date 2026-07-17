"""Training utilities shared by the PPO command-line workflow."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainingConfig:
    total_steps: int = 150000
    num_envs: int = 8
    rollout_steps: int = 256
    seed: int = 1
    device: str = "auto"
    eval_interval: int = 10000
    eval_episodes: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03


class PerformanceCurriculum:
    """Two-stage curriculum driven only by consecutive evaluation results."""

    def __init__(self, stage=1, consecutive_passes=0, learnability_passed=False):
        self.stage = int(stage)
        self.consecutive_passes = int(consecutive_passes)
        self.learnability_passed = bool(learnability_passed)

    def update(self, fixed_hit_rate, randomized_hit_rate=None):
        previous_stage = self.stage
        if self.stage == 1:
            passed = float(fixed_hit_rate) >= 0.80
            self.consecutive_passes = (
                self.consecutive_passes + 1 if passed else 0)
            if self.consecutive_passes >= 2:
                self.stage = 2
                self.consecutive_passes = 0
        else:
            passed = (
                float(fixed_hit_rate) >= 0.80
                and randomized_hit_rate is not None
                and float(randomized_hit_rate) >= 0.60)
            self.consecutive_passes = (
                self.consecutive_passes + 1 if passed else 0)
            if self.consecutive_passes >= 2:
                self.learnability_passed = True
        return self.stage != previous_stage

    def state_dict(self):
        return {
            "stage": self.stage,
            "consecutive_passes": self.consecutive_passes,
            "learnability_passed": self.learnability_passed,
        }


def curriculum_stage(global_step=None, total_steps=None):
    """Compatibility helper; performance curriculum always starts at stage 1."""
    del global_step, total_steps
    return 1


def best_fixed_key(result):
    if not result.get("best_eligible", False):
        return None
    return (result["red_hit_rate"], result["mean_return"],
            -result["numerical_invalid"])


def best_randomized_key(result):
    return best_fixed_key(result)


def best_joint_key(fixed, randomized):
    if not fixed.get("best_eligible", False) or not randomized.get(
            "best_eligible", False):
        return None
    return (
        min(fixed["red_hit_rate"], randomized["red_hit_rate"]),
        fixed["red_hit_rate"] + randomized["red_hit_rate"],
        0.5 * (fixed["mean_return"] + randomized["mean_return"]),
        -(fixed["numerical_invalid"] + randomized["numerical_invalid"]),
    )


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_training_state(path, actor, critic, optimizer, global_step,
                        update_index, best_eval_hit_rate, config,
                        best_eval_return=float("-inf"),
                        best_eval_numerical_invalid=10**9,
                        curriculum=None, best_records=None):
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "update_index": int(update_index),
        "best_eval_hit_rate": float(best_eval_hit_rate),
        "best_eval_return": float(best_eval_return),
        "best_eval_numerical_invalid": int(best_eval_numerical_invalid),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__")
        else dict(config),
        "curriculum": curriculum,
        "best_records": best_records,
    }
    torch.save(payload, path)
    return payload


def load_training_state(path, actor, critic, optimizer, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    return payload


def save_model(path, actor, critic, config, global_step):
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__")
        else dict(config),
        "global_step": int(global_step),
    }, path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
