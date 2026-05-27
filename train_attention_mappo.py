"""MAPPO-Attention baseline training script.

This script is intentionally separate from ``train_vanilla_mappo.py``.  It uses
an actor-side EntityObservationEncoder while keeping the vanilla centralized
critic over flattened red observations.  It does not implement BRMA-MAPPO,
MaskVectorGenerator, or biased random masks.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F

from attention_models import AttentionActor, CentralizedAttentionCritic
from my_uav_env.alignment.entity_obs import build_entity_observation
from my_uav_env.alignment.global_state import (
    build_strict_team_global_state,
    describe_strict_global_state_layout,
    infer_strict_team_global_state_dim,
)
from my_uav_env.alignment.normalization import normalize_strict_entities
from my_uav_env.alignment.obs_adapter import build_paper_entity_observation_from_env_obs
from my_uav_env.alignment.reward_utils import REWARD_VERSION
from rule_based_agent import blue_coordinated_actions
from train_vanilla_mappo import (
    CentralizedCritic,
    Config,
    RolloutBuffer,
    SubprocVecEnv,
    _classify_death_reason,
    _cleanup_rotating_checkpoints,
    _compute_obs_dim,
    _current_entropy_coef,
    _episode_outcome,
    _flatten_obs,
    _grad_has_nan,
    _safe_div,
    _select_device,
    _set_main_process_seed,
    compute_gae,
    make_config_from_args,
    parse_args,
)


class AttentionRolloutBuffer:
    """Rollout storage for attention actor + flattened centralized critic."""

    def __init__(self, num_steps: int, num_envs: int, num_red: int,
                 action_dim: int, rnn_hidden_size: int,
                 global_obs_dim: int | None = None):
        self.num_steps = num_steps
        t_size, e_size, a_size = num_steps, num_envs, num_red
        h_size = rnn_hidden_size

        self.obs: list[list[list[np.ndarray]]] = [
            [[None for _ in range(a_size)] for _ in range(e_size)]
            for _ in range(t_size)
        ]
        self.entities: list[list[list[np.ndarray]]] = [
            [[None for _ in range(a_size)] for _ in range(e_size)]
            for _ in range(t_size)
        ]
        self.entity_masks: list[list[list[np.ndarray]]] = [
            [[None for _ in range(a_size)] for _ in range(e_size)]
            for _ in range(t_size)
        ]
        self.actions = np.zeros((t_size, e_size, a_size, action_dim),
                                dtype=np.float32)
        self.rewards = np.zeros((t_size, e_size, a_size), dtype=np.float32)
        self.values = np.zeros((t_size, e_size, a_size), dtype=np.float32)
        self.log_probs = np.zeros((t_size, e_size, a_size), dtype=np.float32)
        self.dones = np.zeros((t_size, e_size, a_size), dtype=np.float32)
        self.alive = np.zeros((t_size, e_size, a_size), dtype=bool)
        self.rnn_actor_init = np.zeros((e_size, a_size, h_size), dtype=np.float32)
        self.rnn_actor_final = np.zeros((e_size, a_size, h_size), dtype=np.float32)
        self.bootstrap_values = np.zeros((e_size, a_size), dtype=np.float32)

        _god = global_obs_dim or 1
        self.global_obs = np.zeros((t_size, e_size, _god), dtype=np.float32)
        self._global_obs_dim = _god

        self.critic_entities: list[list[None | np.ndarray]] = [
            [None for _ in range(e_size)] for _ in range(t_size)]
        self.critic_entity_masks: list[list[None | np.ndarray]] = [
            [None for _ in range(e_size)] for _ in range(t_size)]

    def store_step(self, step: int, env_idx: int, agent_idx: int,
                   entities_np: np.ndarray, entity_mask_np: np.ndarray,
                   obs_flat: np.ndarray, action: np.ndarray, reward: float,
                   value: float, log_prob: float, done: float, alive: bool):
        self.entities[step][env_idx][agent_idx] = entities_np
        self.entity_masks[step][env_idx][agent_idx] = entity_mask_np
        self.obs[step][env_idx][agent_idx] = obs_flat
        self.actions[step, env_idx, agent_idx] = action
        self.rewards[step, env_idx, agent_idx] = reward
        self.values[step, env_idx, agent_idx] = value
        self.log_probs[step, env_idx, agent_idx] = log_prob
        self.dones[step, env_idx, agent_idx] = done
        self.alive[step, env_idx, agent_idx] = alive

    def store_critic_entities(self, step: int, env_idx: int,
                              team_entities: np.ndarray,
                              team_masks: np.ndarray):
        self.critic_entities[step][env_idx] = team_entities
        self.critic_entity_masks[step][env_idx] = team_masks


def parse_args_attention():
    raw_argv = sys.argv[1:]
    clean_argv = []
    obs_adapter = "current"
    critic_state = "engineering"
    encoder_mode = "current"
    preset_name = None
    list_presets = False

    i = 0
    while i < len(raw_argv):
        item = raw_argv[i]
        if item == "--obs-adapter":
            if i + 1 >= len(raw_argv):
                raise SystemExit(
                    "--obs-adapter requires one of: current, paper-placeholder, strict")
            obs_adapter = raw_argv[i + 1]
            i += 2
            continue
        if item.startswith("--obs-adapter="):
            obs_adapter = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--critic-state":
            if i + 1 >= len(raw_argv):
                raise SystemExit(
                    "--critic-state requires one of: engineering, strict-global, attention-entities")
            critic_state = raw_argv[i + 1]
            i += 2
            continue
        if item.startswith("--critic-state="):
            critic_state = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--encoder-mode":
            if i + 1 >= len(raw_argv):
                raise SystemExit(
                    "--encoder-mode requires one of: current, paper-eq33")
            encoder_mode = raw_argv[i + 1]
            i += 2
            continue
        if item.startswith("--encoder-mode="):
            encoder_mode = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--preset":
            if i + 1 >= len(raw_argv):
                raise SystemExit("--preset requires a preset name")
            preset_name = raw_argv[i + 1]
            i += 2
            continue
        if item.startswith("--preset="):
            preset_name = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--list-presets":
            list_presets = True
            i += 1
            continue
        clean_argv.append(item)
        i += 1

    if list_presets:
        from configs.experiment_presets import list_presets as _lp
        print("Available presets:")
        for name in _lp():
            print(f"  {name}")
        raise SystemExit(0)

    if obs_adapter not in ("current", "paper-placeholder", "strict"):
        raise SystemExit(
            "--obs-adapter must be one of: current, paper-placeholder, strict")
    if critic_state not in ("engineering", "strict-global", "attention-entities"):
        raise SystemExit(
            "--critic-state must be one of: engineering, strict-global, attention-entities")
    if encoder_mode not in ("current", "paper-eq33"):
        raise SystemExit(
            "--encoder-mode must be one of: current, paper-eq33")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + clean_argv
        args = parse_args()
    finally:
        sys.argv = old_argv

    args.obs_adapter = obs_adapter
    args.critic_state = critic_state
    args.encoder_mode = encoder_mode

    if preset_name is not None:
        from configs.experiment_presets import get_preset
        preset = get_preset(preset_name)
        _apply_preset_attention(args, preset, raw_argv)
        if args.obs_adapter not in ("current", "paper-placeholder", "strict"):
            raise SystemExit(
                "--obs-adapter must be one of: current, paper-placeholder, strict")
        if args.critic_state not in ("engineering", "strict-global", "attention-entities"):
            raise SystemExit(
                "--critic-state must be one of: engineering, strict-global, attention-entities")
        if args.encoder_mode not in ("current", "paper-eq33"):
            raise SystemExit(
                "--encoder-mode must be one of: current, paper-eq33")
    else:
        if not _has_cli_option(clean_argv, "--log-file"):
            args.log_file = "attention_training_log.csv"
        if not _has_cli_option(clean_argv, "--results-file"):
            args.results_file = "results/attention_mappo_results.csv"
        if not _has_cli_option(clean_argv, "--checkpoint-dir"):
            args.checkpoint_dir = "checkpoints_attention"
    return args


_ATTENTION_PRESET_CLI_FLAGS = {
    "num_red", "num_blue", "num_envs", "total_env_steps",
    "max_episode_length", "replay_buffer_size", "n_minibatches",
    "actor_lr", "critic_lr", "entropy_coef",
    "enable_blue_gcas", "resume_from_best",
    "log_file", "results_file", "checkpoint_dir", "device",
    "obs_adapter", "critic_state", "encoder_mode",
}


def _apply_preset_attention(args, preset: dict, raw_argv: list[str]):
    """Apply preset values to args for any key not explicitly given on CLI."""
    cli_flags = set()
    for item in raw_argv:
        if item.startswith("--"):
            name = item.lstrip("-").split("=", 1)[0].replace("-", "_")
            cli_flags.add(name)
    for key, value in preset.items():
        if key not in _ATTENTION_PRESET_CLI_FLAGS:
            continue
        if key not in cli_flags:
            setattr(args, key, value)


def _has_cli_option(argv: list[str], name: str) -> bool:
    return name in argv or any(item.startswith(name + "=") for item in argv)


def _build_attention_entities(obs_np: dict, obs_adapter: str):
    if obs_adapter == "current":
        return build_entity_observation(obs_np)
    if obs_adapter == "paper-placeholder":
        return build_paper_entity_observation_from_env_obs(obs_np)
    if obs_adapter == "strict":
        raise ValueError(
            "strict adapter must use env.get_strict_team_observations, not obs_np")
    raise ValueError(f"Unknown obs_adapter: {obs_adapter}")


def _zero_entity_like(obs_np: dict, obs_adapter: str) -> tuple[np.ndarray, np.ndarray]:
    if obs_adapter == "strict":
        ally_states = np.asarray(obs_np["ally_states"])
        enemy_states = np.asarray(obs_np["enemy_states"])
        n_entities = 1 + int(ally_states.shape[0]) + int(enemy_states.shape[0])
        entities = np.zeros((n_entities, 10), dtype=np.float32)
        mask = np.ones((n_entities,), dtype=np.int64)
        return entities, mask
    entities, mask = _build_attention_entities(obs_np, obs_adapter)
    return np.zeros_like(entities, dtype=np.float32), np.ones_like(mask, dtype=np.int64)


def _compute_attention_global_obs_dim(config, obs_dim: int) -> int:
    """Return the critic global_obs_dim for the current config."""
    if config.critic_state == "engineering":
        return obs_dim * config.num_red
    if config.critic_state == "strict-global":
        if config.obs_adapter != "strict":
            raise ValueError(
                "--critic-state strict-global requires --obs-adapter strict, "
                f"got obs_adapter={config.obs_adapter!r}")
        return infer_strict_team_global_state_dim(
            num_red=config.num_red,
            num_blue=config.num_blue,
            entity_dim=10,
            include_masks=True,
        )
    raise ValueError(f"Unknown critic_state: {config.critic_state!r}")


def _build_global_obs_for_env(
    env_obs: dict,
    strict_env_obs: dict | None,
    red_ids: list[str],
    obs_dim: int,
    config,
) -> np.ndarray:
    """Build the per-timestep critic global observation."""
    if config.critic_state == "engineering":
        parts = []
        for rid in red_ids:
            if rid in (env_obs or {}):
                parts.append(_flatten_obs(env_obs[rid]))
            else:
                parts.append(np.zeros(obs_dim, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    if config.critic_state == "strict-global":
        if not strict_env_obs or len(strict_env_obs) < config.num_red:
            god = _compute_attention_global_obs_dim(config, obs_dim)
            return np.zeros(god, dtype=np.float32)
        return build_strict_team_global_state(
            strict_env_obs,
            num_red=config.num_red,
            num_blue=config.num_blue,
            agent_prefix="red",
            include_masks=True,
            normalize=True,
        ).astype(np.float32)

    raise ValueError(f"Unknown critic_state: {config.critic_state!r}")


def _build_attention_critic_entities_for_env(
    strict_env_obs: dict | None,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Build team entities/masks for CentralizedAttentionCritic.

    Returns (team_entities, team_masks) with shapes:
        (num_red, n_entities_per_agent, 10) and (num_red, n_entities_per_agent).
    Missing / dead agents get zeros + all-ones mask.
    """
    n_entities = config.num_red + config.num_blue
    team_entities = np.zeros((config.num_red, n_entities, 10), dtype=np.float32)
    team_masks = np.ones((config.num_red, n_entities), dtype=np.int64)

    if not strict_env_obs or len(strict_env_obs) < config.num_red:
        return team_entities, team_masks

    for i in range(config.num_red):
        rid = f"red_{i}"
        tup = strict_env_obs.get(rid)
        if tup is None:
            continue
        entities, mask, _meta = tup
        entities = normalize_strict_entities(
            np.asarray(entities, dtype=np.float32),
            np.asarray(mask, dtype=np.int64),
        )
        team_entities[i] = entities
        team_masks[i] = np.asarray(mask, dtype=np.int64)
    return team_entities, team_masks


def _fetch_strict_red_team_obs(vec_env, config) -> list[dict]:
    """Fetch strict red-team observations from each worker env."""
    results = vec_env.env_method(
        "get_strict_team_observations", "red", timeout=30.0)
    cleaned = []
    for item in results:
        if item is None or isinstance(item, set) or not item:
            cleaned.append({})
        elif isinstance(item, dict):
            cleaned.append(item)
        else:
            cleaned.append({})
    while len(cleaned) < config.num_envs:
        cleaned.append({})
    return cleaned[:config.num_envs]


def _fetch_blue_own_positions(vec_env, timeout: float = 30.0) -> list[dict]:
    """Fetch blue ownship positions from worker envs for rule-policy patrol."""

    raw = vec_env.env_method("get_blue_own_positions", timeout=timeout)
    return [item if isinstance(item, dict) else {} for item in raw]


def _fetch_blue_own_kinematics(
    vec_env,
    timeout: float = 30.0,
) -> tuple[list[dict], list[dict]]:
    """Fetch blue ownship positions/headings from worker envs."""

    raw = vec_env.env_method("get_blue_own_kinematics", timeout=timeout)
    positions_list: list[dict] = []
    headings_list: list[dict] = []
    for item in raw:
        pos: dict = {}
        hdg: dict = {}
        if isinstance(item, dict):
            for bid, data in item.items():
                if isinstance(data, dict):
                    if "position" in data:
                        pos[bid] = data["position"]
                    if "heading" in data:
                        hdg[bid] = data["heading"]
        positions_list.append(pos)
        headings_list.append(hdg)
    return positions_list, headings_list


def ppo_update_attention(actor, critic, actor_opt, critic_opt,
                         buffer: AttentionRolloutBuffer, config: Config,
                         device: torch.device, total_steps: int = 0):
    num_steps = buffer.num_steps
    num_envs = buffer.rnn_actor_init.shape[0]
    num_red = buffer.rnn_actor_init.shape[1]

    global_obs_by_env = []
    trajectories = []

    for env_idx in range(num_envs):
        global_obs_by_env.append(
            buffer.global_obs[:, env_idx, :].astype(np.float32))

        for agent_idx in range(num_red):
            t_entities = []
            t_masks = []
            t_actions = []
            t_rewards = []
            t_values = []
            t_log_probs = []
            t_dones = []
            alive_steps = []

            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_entities.append(buffer.entities[step][env_idx][agent_idx])
                    t_masks.append(buffer.entity_masks[step][env_idx][agent_idx])
                    t_actions.append(buffer.actions[step, env_idx, agent_idx])
                    t_rewards.append(buffer.rewards[step, env_idx, agent_idx])
                    t_values.append(buffer.values[step, env_idx, agent_idx])
                    t_log_probs.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_dones.append(buffer.dones[step, env_idx, agent_idx])
                    alive_steps.append(step)

            if not t_actions:
                continue

            bootstrap = float(buffer.bootstrap_values[env_idx, agent_idx])
            rewards = torch.tensor(t_rewards, device=device)
            old_values = torch.tensor(t_values + [bootstrap], device=device)
            dones = torch.tensor(t_dones, device=device)
            advantages, returns = compute_gae(rewards, old_values, dones,
                                              config.gamma, config.gae_lambda)

            adv_std = advantages.std() if advantages.numel() > 1 else torch.std(
                advantages, correction=0)
            if adv_std <= 1e-8 or torch.isnan(adv_std):
                adv_std = 1.0
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            trajectories.append({
                "env_idx": env_idx,
                "agent_idx": agent_idx,
                "alive_steps": alive_steps,
                "entities": np.stack(t_entities).astype(np.float32),
                "entity_masks": np.stack(t_masks).astype(np.int64),
                "actions": np.stack(t_actions).astype(np.float32),
                "old_log_probs": np.asarray(t_log_probs, dtype=np.float32),
                "advantages": advantages.detach(),
                "returns": returns.detach(),
            })

    if not trajectories:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    actor_losses_log = []
    critic_losses_log = []
    entropies_log = []
    entropy_coef = _current_entropy_coef(config, total_steps)

    for _epoch in range(config.n_update_epochs):
        order = np.random.permutation(len(trajectories))
        minibatches = np.array_split(order, max(1, min(config.n_minibatches, len(order))))

        for mb in minibatches:
            if len(mb) == 0:
                continue

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            actor_losses = []
            critic_losses = []
            entropies = []

            for traj_idx in mb:
                traj = trajectories[int(traj_idx)]
                env_idx = traj["env_idx"]
                agent_idx = traj["agent_idx"]

                rnn_a = torch.as_tensor(
                    buffer.rnn_actor_init[env_idx, agent_idx],
                    dtype=torch.float32, device=device).unsqueeze(0)
                entities = torch.as_tensor(traj["entities"], dtype=torch.float32,
                                           device=device)
                masks = torch.as_tensor(traj["entity_masks"], dtype=torch.long,
                                        device=device)
                actions = torch.as_tensor(traj["actions"], dtype=torch.float32,
                                          device=device)
                old_lp = torch.as_tensor(traj["old_log_probs"],
                                         dtype=torch.float32, device=device)
                advantages = traj["advantages"].to(device)
                returns = traj["returns"].to(device)

                new_lps = []
                traj_entropies = []
                for t_idx in range(entities.shape[0]):
                    action_dist, rnn_a, _attn = actor(
                        entities[t_idx].unsqueeze(0),
                        masks[t_idx].unsqueeze(0),
                        rnn_a,
                    )
                    new_lps.append(
                        action_dist.log_prob(actions[t_idx].unsqueeze(0)).sum(dim=-1))
                    traj_entropies.append(action_dist.entropy().mean())

                new_lp = torch.cat(new_lps)
                entropy_mean = torch.stack(traj_entropies).mean()
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(policy_loss - entropy_coef * entropy_mean)
                entropies.append(entropy_mean.detach())

                if config.critic_state == "attention-entities":
                    crit_ents = [buffer.critic_entities[s][env_idx]
                                 for s in traj["alive_steps"]]
                    crit_msks = [buffer.critic_entity_masks[s][env_idx]
                                 for s in traj["alive_steps"]]
                    ent_t = torch.as_tensor(np.stack(crit_ents),
                                            dtype=torch.float32, device=device)
                    msk_t = torch.as_tensor(np.stack(crit_msks),
                                            dtype=torch.long, device=device)
                    all_values = critic(ent_t, msk_t)  # (T, num_red)
                    values = all_values[:, agent_idx]
                else:
                    global_obs = torch.as_tensor(
                        global_obs_by_env[env_idx][traj["alive_steps"]],
                        dtype=torch.float32, device=device)
                    values = critic(global_obs).squeeze(-1)
                critic_losses.append(F.mse_loss(values, returns))

            actor_loss = torch.stack(actor_losses).mean()
            critic_loss = torch.stack(critic_losses).mean()
            (actor_loss + critic_loss).backward()

            actor_nan = _grad_has_nan(actor)
            critic_nan = _grad_has_nan(critic)
            if actor_nan or critic_nan:
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                print(f"  [WARN] NaN gradient detected (actor={actor_nan}, "
                      f"critic={critic_nan}) - skipping minibatch", flush=True)
                continue

            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            actor_opt.step()
            critic_opt.step()

            actor_losses_log.append(float(actor_loss.item()))
            critic_losses_log.append(float(critic_loss.item()))
            entropies_log.append(float(torch.stack(entropies).mean().item()))

    if not actor_losses_log:
        return {"actor_loss": float("nan"), "critic_loss": float("nan"), "entropy": 0.0}

    return {
        "actor_loss": float(np.mean(actor_losses_log)),
        "critic_loss": float(np.mean(critic_losses_log)),
        "entropy": float(np.mean(entropies_log)),
    }


def _write_results(results_log: list[dict], results_file: str):
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(results_log[0].keys())
        for row in results_log:
            writer.writerow(row.values())


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args_attention()
    config = make_config_from_args(args)
    config.obs_adapter = args.obs_adapter
    config.critic_state = args.critic_state
    config.encoder_mode = ("paper_eq33" if args.encoder_mode == "paper-eq33"
                           else args.encoder_mode)
    _set_main_process_seed(config.seed)
    device = _select_device(config.device)

    obs_dim = _compute_obs_dim(config.num_red, config.num_blue, is_red=True)
    entity_dim = 11 if config.obs_adapter == "current" else 10
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    _ensure_parent_dir(config.log_file)
    csv_file = open(config.log_file, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Iteration", "Step", "ActorLoss", "CriticLoss", "Entropy",
        "RedMeanReward", "RedWinRate", "RedRewardStd", "WinRateRecent",
        "RedMissiles", "BlueMissiles", "Episodes", "RedWins", "BlueWins",
        "Draws", "RedAliveMean", "BlueAliveMean", "RedDeathsMissile",
        "RedDeathsCrash", "BlueDeathsMissile", "BlueDeathsCrash",
        "RedMissileHits", "BlueMissileHits", "RedMissileHitRate",
        "BlueMissileHitRate", "KD_Red", "RWR", "RewardVersion",
    ])
    csv_file.flush()

    print(f"Device: {device}")
    if config.critic_state == "attention-entities":
        print("Architecture: MAPPO-Attention Actor + CentralizedAttentionCritic")
        print("Critic input: strict per-agent entity tables")
        print(f"Critic encoder_mode: {config.encoder_mode}")
    else:
        print("Architecture: MAPPO-Attention Actor + Vanilla CentralizedCritic")
    print("Current mask status: no biased random mask / no mask generator")
    print("Final config:")
    print(f"  num_red / num_blue: {config.num_red} / {config.num_blue}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  total_env_steps: {config.total_env_steps}")
    print(f"  max_episode_length: {config.max_episode_length}")
    print(f"  replay_buffer_size: {config.replay_buffer_size}")
    print(f"  log_file: {config.log_file}")
    print(f"  results_file: {config.results_file}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  seed: {config.seed}")
    print(f"  obs_adapter: {config.obs_adapter}")
    print(f"  encoder_mode: {config.encoder_mode}")
    print(f"  entity_dim: {entity_dim}")
    print(f"  reward_version: {REWARD_VERSION}")
    if config.checkpoint_dir == "checkpoints_attention":
        if config.obs_adapter == "paper-placeholder":
            print("[WARN] paper-placeholder uses entity_dim=10; use a separate "
                  "checkpoint_dir to avoid mixing with current adapter checkpoints.",
                  flush=True)
        elif config.obs_adapter == "strict":
            print("[WARN] strict uses entity_dim=10 and env-state observations; "
                  "use a separate checkpoint_dir.", flush=True)

    min_episodes_to_eval = 50
    num_steps = config.replay_buffer_size // config.num_envs
    env_kwargs = dict(max_num_blue=config.num_blue, max_num_red=config.num_red,
                      max_steps=config.max_episode_length,
                      enable_gcas_for_blue=config.enable_blue_gcas)
    print(f"Starting {config.num_envs} workers...", flush=True)
    vec_env = SubprocVecEnv(config.num_envs, env_kwargs)

    red_ids = [f"red_{i}" for i in range(config.num_red)]
    blue_ids = [f"blue_{i}" for i in range(config.num_blue)]

    actor = AttentionActor(entity_dim=entity_dim, action_dim=config.action_dim,
                           hidden_size=config.mlp_hidden,
                           rnn_hidden=config.rnn_hidden_size,
                           encoder_mode=config.encoder_mode).to(device)
    if config.critic_state == "attention-entities":
        if config.obs_adapter != "strict":
            raise SystemExit(
                "--critic-state attention-entities requires --obs-adapter strict")
        global_obs_dim = 1  # placeholder, not used
        critic = CentralizedAttentionCritic(
            entity_dim=10, hidden_size=config.mlp_hidden,
            num_heads=4, num_agents=config.num_red,
            encoder_mode=config.encoder_mode,
        ).to(device)
    else:
        global_obs_dim = _compute_attention_global_obs_dim(config, obs_dim)
        critic = CentralizedCritic(global_obs_dim=global_obs_dim,
                                   hidden=config.mlp_hidden).to(device)
    print(f"Actor params:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,} "
          f"(centralized, global_obs_dim={global_obs_dim}, "
          f"critic_state={config.critic_state})")
    if config.critic_state == "strict-global":
        layout = describe_strict_global_state_layout(config.num_red, config.num_blue)
        print(f"Strict critic global state layout: {layout}")
    if (config.critic_state == "strict-global"
            and config.checkpoint_dir == "checkpoints_attention"):
        print("[WARN] strict-global uses different critic input dimension; "
              "use a separate checkpoint_dir.")
    if config.obs_adapter == "strict" or config.critic_state == "strict-global":
        print("Strict observation normalization: enabled")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    actor_best_path = os.path.join(config.checkpoint_dir, "attention_actor_best.pt")
    critic_best_path = os.path.join(config.checkpoint_dir, "centralized_critic_best.pt")
    if (config.resume_from_best and os.path.exists(actor_best_path)
            and os.path.exists(critic_best_path)):
        actor.load_state_dict(torch.load(actor_best_path, map_location=device,
                                         weights_only=True))
        critic.load_state_dict(torch.load(critic_best_path, map_location=device,
                                          weights_only=True))
        print("[OK] Loaded best attention checkpoint")
    else:
        print("[WARN] Best attention checkpoint not loaded; using random init")

    rnn_hidden_actor = np.zeros(
        (config.num_envs, config.num_red, config.rnn_hidden_size), dtype=np.float32)

    print(f"Resetting {config.num_envs} environments...", flush=True)
    t_reset = time.perf_counter()
    raw_obs_list = vec_env.reset(timeout=300.0)
    if config.obs_adapter == "strict":
        strict_obs_list = _fetch_strict_red_team_obs(vec_env, config)
    else:
        strict_obs_list = None
    print(f"Reset done ({time.perf_counter() - t_reset:.0f}s)", flush=True)

    total_steps = 0
    iteration = 1
    total_episodes = 0
    red_wins = 0
    blue_wins = 0
    draws = 0
    death_stats = {"red": Counter(), "blue": Counter()}
    red_missiles_total = 0.0
    blue_missiles_total = 0.0
    best_reward_value = -float("inf")
    best_reward_win_rate = 0.0
    best_winrate_value = -float("inf")
    best_winrate_reward = -float("inf")

    recent_ep_rewards_red = deque(maxlen=50)
    comp_keys = ["r_pitch", "r_roll", "r_alt", "r_bound",
                 "r_vel", "r_adv", "r_end", "r_death"]
    recent_ep_comps_red: deque[dict] = deque(maxlen=50)
    recent_ep_missiles_red = deque(maxlen=50)
    recent_ep_missiles_blue = deque(maxlen=50)
    recent_ep_red_alive = deque(maxlen=50)
    recent_ep_blue_alive = deque(maxlen=50)
    current_ep_reward_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_comp_red = {k: np.zeros(config.num_envs, dtype=np.float64)
                           for k in comp_keys}
    current_ep_missiles_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_missiles_blue = np.zeros(config.num_envs, dtype=np.float32)
    results_log: list[dict] = []

    while total_steps < config.total_env_steps:
        t_start = time.perf_counter()
        buffer = AttentionRolloutBuffer(
            num_steps=num_steps, num_envs=config.num_envs,
            num_red=config.num_red, action_dim=config.action_dim,
            rnn_hidden_size=config.rnn_hidden_size,
            global_obs_dim=global_obs_dim,
        )
        buffer.rnn_actor_init = rnn_hidden_actor.copy()
        iter_episodes = 0
        iter_red_wins = 0

        for step in range(num_steps):
            actions_list = []
            engaged_sets = vec_env.env_method("refresh_engaged_targets")
            blue_own_positions_list, blue_own_headings_list = (
                _fetch_blue_own_kinematics(vec_env))

            for env_idx in range(config.num_envs):
                env_obs = raw_obs_list[env_idx]
                env_actions = {}
                if not env_obs or len(env_obs) == 0:
                    zero_flat = np.zeros(obs_dim, dtype=np.float32)
                    zero_entities = np.zeros((1 + config.num_red - 1 + config.num_blue,
                                              entity_dim),
                                             dtype=np.float32)
                    one_mask = np.ones((zero_entities.shape[0],), dtype=np.int64)
                    for i, rid in enumerate(red_ids):
                        env_actions[rid] = np.zeros(config.action_dim, dtype=np.float32)
                        buffer.store_step(
                            step, env_idx, i, zero_entities, one_mask, zero_flat,
                            np.zeros(config.action_dim, dtype=np.float32),
                            0.0, 0.0, 0.0, 1.0, alive=False)
                    for bid in blue_ids:
                        env_actions[bid] = np.zeros(config.action_dim, dtype=np.float32)
                    actions_list.append(env_actions)
                    continue

                blue_obs_dict = {bid: env_obs[bid] for bid in blue_ids}
                env_actions.update(blue_coordinated_actions(
                    blue_obs_dict, config.num_blue, config.num_red,
                    engaged_targets=engaged_sets[env_idx],
                    own_positions=blue_own_positions_list[env_idx],
                    own_headings=blue_own_headings_list[env_idx]))

                red_obs_flat_all = []
                entity_batch = []
                mask_batch = []
                alive_red_indices = []
                alive_obs_flat = []

                for i, rid in enumerate(red_ids):
                    obs_np = env_obs[rid]
                    obs_flat = _flatten_obs(obs_np)
                    red_obs_flat_all.append(obs_flat)
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        if config.obs_adapter == "strict":
                            strict_env_obs = (
                                strict_obs_list[env_idx] if strict_obs_list else {})
                            strict_tuple = strict_env_obs.get(rid)
                            if strict_tuple is not None:
                                entities_np, entity_mask_np, _meta = strict_tuple
                                entities_np = np.asarray(
                                    entities_np, dtype=np.float32)
                                entity_mask_np = np.asarray(
                                    entity_mask_np, dtype=np.int64)
                                entities_np = normalize_strict_entities(
                                    entities_np, entity_mask_np)
                            else:
                                entities_np, entity_mask_np = _zero_entity_like(
                                    obs_np, config.obs_adapter)
                        else:
                            entities_np, entity_mask_np = _build_attention_entities(
                                obs_np, config.obs_adapter)
                        entity_batch.append(entities_np)
                        mask_batch.append(entity_mask_np)
                        alive_obs_flat.append(obs_flat)
                        alive_red_indices.append(i)
                    else:
                        entities_np, entity_mask_np = _zero_entity_like(
                            obs_np, config.obs_adapter)
                        env_actions[rid] = np.zeros(config.action_dim, dtype=np.float32)
                        buffer.store_step(
                            step, env_idx, i, entities_np, entity_mask_np, obs_flat,
                            np.zeros(config.action_dim, dtype=np.float32),
                            0.0, 0.0, 0.0, 1.0, alive=False)

                if alive_red_indices:
                    entities_t = torch.as_tensor(np.stack(entity_batch),
                                                 dtype=torch.float32, device=device)
                    mask_t = torch.as_tensor(np.stack(mask_batch),
                                             dtype=torch.long, device=device)
                    rnn_t = torch.as_tensor(
                        rnn_hidden_actor[env_idx, alive_red_indices],
                        dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action_dist, new_rnn_a, _attn = actor(entities_t, mask_t, rnn_t)
                        action_raw = action_dist.sample()
                        action = action_raw.clamp(-0.999, 0.999)
                        log_prob = action_dist.log_prob(action).sum(dim=-1)

                    for k, i in enumerate(alive_red_indices):
                        rid = red_ids[i]
                        env_actions[rid] = action[k].cpu().numpy()
                        rnn_hidden_actor[env_idx, i] = new_rnn_a[k].cpu().numpy()
                        buffer.store_step(
                            step, env_idx, i,
                            entity_batch[k], mask_batch[k], alive_obs_flat[k],
                            action[k].cpu().numpy(),
                            0.0, 0.0, log_prob[k].item(), 0.0, alive=True)

                if config.critic_state == "attention-entities":
                    team_ent, team_msk = _build_attention_critic_entities_for_env(
                        strict_obs_list[env_idx] if strict_obs_list else None,
                        config,
                    )
                    buffer.store_critic_entities(step, env_idx, team_ent, team_msk)
                    ent_t = torch.as_tensor(team_ent, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    msk_t = torch.as_tensor(team_msk, dtype=torch.long,
                                            device=device).unsqueeze(0)
                    with torch.no_grad():
                        v_per_agent = critic(ent_t, msk_t).squeeze(0)
                    for i in range(config.num_red):
                        if buffer.alive[step, env_idx, i]:
                            buffer.values[step, env_idx, i] = float(v_per_agent[i].item())
                else:
                    global_obs_np = _build_global_obs_for_env(
                        env_obs,
                        strict_obs_list[env_idx] if strict_obs_list else None,
                        red_ids, obs_dim, config,
                    )
                    buffer.global_obs[step, env_idx] = global_obs_np
                    global_obs_t = torch.as_tensor(
                        global_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        v_global = critic(global_obs_t).item()
                    for i in range(config.num_red):
                        if buffer.alive[step, env_idx, i]:
                            buffer.values[step, env_idx, i] = v_global

                actions_list.append(env_actions)

            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(actions_list)

            for env_idx in range(config.num_envs):
                rew = rewards_list[env_idx]
                dones = dones_list[env_idx]
                info = infos_list[env_idx]

                for i, rid in enumerate(red_ids):
                    if buffer.alive[step, env_idx, i]:
                        buffer.rewards[step, env_idx, i] = float(rew.get(rid, 0.0))
                        buffer.dones[step, env_idx, i] = float(dones.get(rid, False))
                    if dones.get(rid, False):
                        rnn_hidden_actor[env_idx, i] = np.zeros(
                            config.rnn_hidden_size, dtype=np.float32)
                    current_ep_reward_red[env_idx] += rew.get(rid, 0.0)
                    rcinfo = info.get(rid, {})
                    for key in comp_keys:
                        current_ep_comp_red[key][env_idx] += rcinfo.get(key, 0.0)

                for rid in red_ids:
                    fired = info.get(rid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_red[env_idx] += fired
                    red_missiles_total += fired
                for bid in blue_ids:
                    fired = info.get(bid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_blue[env_idx] += fired
                    blue_missiles_total += fired

                if all(dones.values()):
                    total_episodes += 1
                    iter_episodes += 1
                    blue_alive = sum(1 for bid in blue_ids
                                     if info.get(bid, {}).get("alive", False))
                    red_alive = sum(1 for rid in red_ids
                                    if info.get(rid, {}).get("alive", False))
                    outcome = _episode_outcome(red_alive, blue_alive)
                    if outcome == "red":
                        red_wins += 1
                        iter_red_wins += 1
                    elif outcome == "blue":
                        blue_wins += 1
                    else:
                        draws += 1

                    for bid in blue_ids:
                        death_reason = info.get(bid, {}).get("death_reason")
                        if death_reason:
                            death_stats["blue"][death_reason] += 1
                    for rid in red_ids:
                        death_reason = info.get(rid, {}).get("death_reason")
                        if death_reason:
                            death_stats["red"][death_reason] += 1

                    recent_ep_rewards_red.append(float(current_ep_reward_red[env_idx]))
                    recent_ep_comps_red.append(
                        {k: float(current_ep_comp_red[k][env_idx]) for k in comp_keys})
                    recent_ep_missiles_red.append(float(current_ep_missiles_red[env_idx]))
                    recent_ep_missiles_blue.append(float(current_ep_missiles_blue[env_idx]))
                    recent_ep_red_alive.append(float(red_alive))
                    recent_ep_blue_alive.append(float(blue_alive))
                    current_ep_reward_red[env_idx] = 0.0
                    for key in comp_keys:
                        current_ep_comp_red[key][env_idx] = 0.0
                    current_ep_missiles_red[env_idx] = 0.0
                    current_ep_missiles_blue[env_idx] = 0.0

            raw_obs_list = next_obs_list
            if config.obs_adapter == "strict":
                strict_obs_list = _fetch_strict_red_team_obs(vec_env, config)
            total_steps += config.num_envs

        buffer.rnn_actor_final = rnn_hidden_actor.copy()

        for env_idx in range(config.num_envs):
            env_obs = raw_obs_list[env_idx]
            if not env_obs or len(env_obs) == 0:
                continue
            if config.critic_state == "attention-entities":
                team_ent, team_msk = _build_attention_critic_entities_for_env(
                    strict_obs_list[env_idx] if strict_obs_list else None,
                    config,
                )
                ent_t = torch.as_tensor(team_ent, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                msk_t = torch.as_tensor(team_msk, dtype=torch.long,
                                        device=device).unsqueeze(0)
                with torch.no_grad():
                    v_per_agent = critic(ent_t, msk_t).squeeze(0)
                for i in range(config.num_red):
                    buffer.bootstrap_values[env_idx, i] = float(v_per_agent[i].item())
            else:
                global_obs_np = _build_global_obs_for_env(
                    env_obs,
                    strict_obs_list[env_idx] if strict_obs_list else None,
                    red_ids, obs_dim, config,
                )
                global_obs_t = torch.as_tensor(global_obs_np, dtype=torch.float32,
                                               device=device).unsqueeze(0)
                with torch.no_grad():
                    v_bootstrap = critic(global_obs_t).item()
                for i in range(config.num_red):
                    buffer.bootstrap_values[env_idx, i] = v_bootstrap

        stats = ppo_update_attention(actor, critic, actor_opt, critic_opt,
                                     buffer, config, device,
                                     total_steps=total_steps)

        t_elapsed = time.perf_counter() - t_start
        avg_r_red = np.mean(recent_ep_rewards_red) if recent_ep_rewards_red else 0.0
        avg_m_red = np.mean(recent_ep_missiles_red) if recent_ep_missiles_red else 0.0
        avg_m_blue = np.mean(recent_ep_missiles_blue) if recent_ep_missiles_blue else 0.0
        red_win_rate = red_wins / max(total_episodes, 1)
        std_r_red = float(np.std(recent_ep_rewards_red)) if len(recent_ep_rewards_red) > 1 else 0.0
        iter_win_rate = iter_red_wins / max(iter_episodes, 1)
        red_alive_mean = np.mean(recent_ep_red_alive) if recent_ep_red_alive else 0.0
        blue_alive_mean = np.mean(recent_ep_blue_alive) if recent_ep_blue_alive else 0.0

        red_deaths_missile = sum(
            v for k, v in death_stats["red"].items()
            if _classify_death_reason(k) == "missile")
        red_deaths_crash = sum(
            v for k, v in death_stats["red"].items()
            if _classify_death_reason(k) == "crash")
        blue_deaths_missile = sum(
            v for k, v in death_stats["blue"].items()
            if _classify_death_reason(k) == "missile")
        blue_deaths_crash = sum(
            v for k, v in death_stats["blue"].items()
            if _classify_death_reason(k) == "crash")
        red_missile_hits = blue_deaths_missile
        blue_missile_hits = red_deaths_missile
        red_total_deaths = sum(death_stats["red"].values())
        blue_total_deaths = sum(death_stats["blue"].values())
        red_missile_hit_rate = _safe_div(red_missile_hits, red_missiles_total)
        blue_missile_hit_rate = _safe_div(blue_missile_hits, blue_missiles_total)
        kd_red = _safe_div(blue_total_deaths, red_total_deaths)
        rwr = _safe_div(red_wins, total_episodes)

        if recent_ep_comps_red:
            avg_comps = {k: float(np.mean([ep[k] for ep in recent_ep_comps_red]))
                         for k in comp_keys}
        else:
            avg_comps = {k: 0.0 for k in comp_keys}
        comp_str = " ".join(f"{k.replace('r_', '').capitalize()}:{avg_comps[k]:+.1f}"
                            for k in comp_keys)

        csv_writer.writerow([
            iteration, total_steps, f"{stats['actor_loss']:.6f}",
            f"{stats['critic_loss']:.6f}", f"{stats['entropy']:.6f}",
            f"{avg_r_red:.4f}", f"{red_win_rate:.6f}", f"{std_r_red:.4f}",
            f"{iter_win_rate:.6f}", f"{avg_m_red:.1f}", f"{avg_m_blue:.1f}",
            total_episodes, red_wins, blue_wins, draws,
            f"{red_alive_mean:.4f}", f"{blue_alive_mean:.4f}",
            red_deaths_missile, red_deaths_crash,
            blue_deaths_missile, blue_deaths_crash,
            red_missile_hits, blue_missile_hits,
            f"{red_missile_hit_rate:.6f}", f"{blue_missile_hit_rate:.6f}",
            f"{kd_red:.6f}", f"{rwr:.6f}", REWARD_VERSION,
        ])
        csv_file.flush()

        results_log.append({
            "Step": total_steps,
            "Iteration": iteration,
            "RedMeanReward": avg_r_red,
            "RedRewardStd": std_r_red,
            "WinRateRecent": iter_win_rate,
            "WinRateCumul": red_win_rate,
            "RedMissiles": avg_m_red,
            "BlueMissiles": avg_m_blue,
            "Episodes": total_episodes,
            "RedWins": red_wins,
            "BlueWins": blue_wins,
            "Draws": draws,
            "RedAliveMean": red_alive_mean,
            "BlueAliveMean": blue_alive_mean,
            "RedDeathsMissile": red_deaths_missile,
            "RedDeathsCrash": red_deaths_crash,
            "BlueDeathsMissile": blue_deaths_missile,
            "BlueDeathsCrash": blue_deaths_crash,
            "RedMissileHits": red_missile_hits,
            "BlueMissileHits": blue_missile_hits,
            "RedMissileHitRate": red_missile_hit_rate,
            "BlueMissileHitRate": blue_missile_hit_rate,
            "KD_Red": kd_red,
            "RWR": rwr,
            "RewardVersion": REWARD_VERSION,
            "ActorLoss": stats["actor_loss"],
            "CriticLoss": stats["critic_loss"],
            "Entropy": stats["entropy"],
            **{key: avg_comps.get(key, 0.0) for key in comp_keys},
        })
        milestone_cur = total_steps // 1_000_000
        milestone_prev = (total_steps - config.num_envs * num_steps) // 1_000_000
        if milestone_cur > milestone_prev or total_steps >= config.total_env_steps:
            _write_results(results_log, config.results_file)
            print(f"  [Results saved] {config.results_file} "
                  f"({len(results_log)} rows)", flush=True)

        print(f"Iter {iteration:5d} | total_steps={total_steps:9d} | "
              f"t={t_elapsed:5.1f}s | R_red={avg_r_red:+8.1f} [{comp_str}] | "
              f"M_red={avg_m_red:.0f} M_blue={avg_m_blue:.0f} | "
              f"ActorLoss={stats['actor_loss']:+.4f} "
              f"CriticLoss={stats['critic_loss']:+.4f} "
              f"EntCoef={_current_entropy_coef(config, total_steps):.4f} "
              f"Entropy={stats['entropy']:.4f} | "
              f"WinRate_red={red_win_rate:.3f} "
              f"(Ep={total_episodes} W={red_wins}/{blue_wins}/{draws})")

        if iteration % 10 == 0:
            actor_path = os.path.join(
                config.checkpoint_dir, f"attention_actor_latest_{iteration:06d}.pt")
            critic_path = os.path.join(
                config.checkpoint_dir,
                f"centralized_critic_latest_{iteration:06d}.pt")
            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)
            _cleanup_rotating_checkpoints(config.checkpoint_dir,
                                          "attention_actor_latest", keep=5)
            _cleanup_rotating_checkpoints(config.checkpoint_dir,
                                          "centralized_critic_latest", keep=5)

        if total_episodes >= min_episodes_to_eval:
            # best_reward: selects by recent average reward
            # best_winrate: selects by recent iteration win rate, reward as tie-breaker
            # legacy best.pt aliases best_winrate for evaluator compatibility
            if avg_r_red > best_reward_value:
                best_reward_value = avg_r_red
                best_reward_win_rate = red_win_rate
                torch.save(actor.state_dict(),
                           os.path.join(config.checkpoint_dir,
                                        "attention_actor_best_reward.pt"))
                torch.save(critic.state_dict(),
                           os.path.join(config.checkpoint_dir,
                                        "centralized_critic_best_reward.pt"))
                print(f"  *** New Best Reward Model Saved! "
                      f"(Reward={best_reward_value:+.2f}, "
                      f"RecentWinRate={iter_win_rate:.4f}, "
                      f"CumulWinRate={red_win_rate:.4f}) ***")

            winrate_is_better = (
                iter_win_rate > best_winrate_value
                or (abs(iter_win_rate - best_winrate_value) < 1e-6
                    and avg_r_red > best_winrate_reward)
            )
            if winrate_is_better:
                best_winrate_value = iter_win_rate
                best_winrate_reward = avg_r_red
                torch.save(actor.state_dict(),
                           os.path.join(config.checkpoint_dir,
                                        "attention_actor_best_winrate.pt"))
                torch.save(critic.state_dict(),
                           os.path.join(config.checkpoint_dir,
                                        "centralized_critic_best_winrate.pt"))
                torch.save(actor.state_dict(), actor_best_path)
                torch.save(critic.state_dict(), critic_best_path)
                print(f"  *** New Best WinRate Model Saved! "
                      f"(RecentWinRate={best_winrate_value:.4f}, "
                      f"Reward={best_winrate_reward:+.2f}, "
                      f"CumulWinRate={red_win_rate:.4f}) ***")

        iteration += 1

    torch.save(actor.state_dict(),
               os.path.join(config.checkpoint_dir, "attention_actor_final.pt"))
    torch.save(critic.state_dict(),
               os.path.join(config.checkpoint_dir, "centralized_critic_final.pt"))
    print("=" * 70)
    print(f"Final models saved to {config.checkpoint_dir}/")
    print(f"Results saved to {config.results_file} ({len(results_log)} rows)")
    print(f"Total episodes: {total_episodes}  Red wins: {red_wins}  "
          f"Blue wins: {blue_wins}  Draws: {draws}  "
          f"Red win rate: {red_win_rate:.4f}")
    csv_file.close()
    vec_env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
