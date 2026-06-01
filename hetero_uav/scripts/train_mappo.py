from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo import MAPPOTrainer, RolloutStorage
from algorithms.mappo.utils import load_yaml
from uav_env import make_env
from uav_env.wrappers import MAPPOEnvWrapper


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    np.random.seed(int(cfg.get("seed", 1)))
    torch.manual_seed(int(cfg.get("seed", 1)))
    device = select_device(str(cfg.get("device", "auto")))
    env = MAPPOEnvWrapper(make_env(cfg["env_config"]))
    obs, state, info = env.reset()

    trainer = MAPPOTrainer(
        obs_dim=env.obs_shape,
        state_dim=env.state_shape,
        action_dim=env.action_shape,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        lr=float(cfg.get("lr", 3e-4)),
        clip_param=float(cfg.get("clip_param", 0.2)),
        value_coef=float(cfg.get("value_coef", 0.5)),
        entropy_coef=float(cfg.get("entropy_coef", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 10.0)),
        device=device,
    )

    total_iterations = int(cfg.get("debug_iterations", 2) if args.debug else cfg.get("total_iterations", 5))
    rollout_steps = int(cfg.get("rollout_steps", 32))
    output_dir = ROOT / str(cfg.get("output_dir", "outputs/mappo"))
    run_dir = output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    model_path = run_dir / "model.pt"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iteration", "average_episode_return", "average_win_rate",
            "average_mav_survival", "average_red_alive", "average_blue_alive",
            "average_episode_length", "policy_loss", "value_loss", "entropy",
        ])
        writer.writeheader()

        episode_returns = []
        episode_wins = []
        episode_mav = []
        episode_red_alive = []
        episode_blue_alive = []
        episode_lengths = []
        current_return = 0.0
        current_length = 0

        for iteration in range(1, total_iterations + 1):
            storage = RolloutStorage(
                rollout_steps=rollout_steps,
                num_agents=env.num_agents,
                obs_dim=env.obs_shape,
                state_dim=env.state_shape,
                action_dim=env.action_shape,
                gamma=float(cfg.get("gamma", 0.99)),
                gae_lambda=float(cfg.get("gae_lambda", 0.95)),
                device=device,
            )

            for _ in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                state_batch = state_t.unsqueeze(0).expand(env.num_agents, -1)
                batch = trainer.model.act(obs_t, state_batch)
                actions = batch.actions.cpu().numpy()
                next_obs, next_state, rewards, dones, info = env.step(actions)
                storage.insert(
                    obs=obs,
                    state=state,
                    actions=actions,
                    log_probs=batch.log_probs.cpu().numpy(),
                    values=batch.values.cpu().numpy(),
                    rewards=rewards,
                    dones=dones,
                )
                current_return += float(np.mean(rewards))
                current_length += 1
                obs, state = next_obs, next_state
                if bool(np.all(dones)):
                    episode_returns.append(current_return)
                    episode_wins.append(1.0 if info.get("winner") == "red" else 0.0)
                    episode_mav.append(float(info.get("mav_survival", 0.0)))
                    episode_red_alive.append(float(info.get("red_alive", 0.0)))
                    episode_blue_alive.append(float(info.get("blue_alive", 0.0)))
                    episode_lengths.append(float(current_length))
                    obs, state, info = env.reset()
                    current_return = 0.0
                    current_length = 0

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                state_batch = state_t.unsqueeze(0).expand(env.num_agents, -1)
                next_value = trainer.model.value(state_batch).cpu().numpy()
            update_stats = trainer.update(
                storage.compute_batch(next_value),
                epochs=int(cfg.get("ppo_epochs", 2)),
                minibatch_size=int(cfg.get("minibatch_size", 64)),
            )

            recent = slice(max(0, len(episode_returns) - 20), len(episode_returns))
            row = {
                "iteration": iteration,
                "average_episode_return": float(np.mean(episode_returns[recent])) if episode_returns else current_return,
                "average_win_rate": float(np.mean(episode_wins[recent])) if episode_wins else 0.0,
                "average_mav_survival": float(np.mean(episode_mav[recent])) if episode_mav else float(info.get("mav_survival", 0.0)),
                "average_red_alive": float(np.mean(episode_red_alive[recent])) if episode_red_alive else float(info.get("red_alive", 0.0)),
                "average_blue_alive": float(np.mean(episode_blue_alive[recent])) if episode_blue_alive else float(info.get("blue_alive", 0.0)),
                "average_episode_length": float(np.mean(episode_lengths[recent])) if episode_lengths else current_length,
                **update_stats,
            }
            writer.writerow(row)
            f.flush()
            print(
                "iteration={iteration} average_episode_return={average_episode_return:.3f} "
                "average_win_rate={average_win_rate:.3f} average_mav_survival={average_mav_survival:.3f} "
                "average_red_alive={average_red_alive:.3f} average_blue_alive={average_blue_alive:.3f} "
                "average_episode_length={average_episode_length:.1f}".format(**row),
                flush=True,
            )
            if iteration % int(cfg.get("save_interval", 1)) == 0:
                trainer.save(model_path, extra={"config": cfg})

    trainer.save(model_path, extra={"config": cfg})
    print(f"saved_model={model_path}")
    env.close()


if __name__ == "__main__":
    main()
