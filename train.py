"""
train.py —— 最小可运行训练循环 (MAPPO + my_uav_env)

将 my_uav_env 环境与 algorithm 中的 MAPPO Actor/Critic 拼接，
验证全流程连通性。蓝方用 PPO 训练，红方用随机策略。

用法：  python train.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_uav_env import UavCombatEnv
from algorithm.mappo_nets import Actor, Critic


# ==============================================================================
#  超参数
# ==============================================================================
MAX_NUM_BLUE = 6
MAX_NUM_RED = 6
MAX_STEPS = 150               # 单集最大步数
NUM_ITERATIONS = 20           # 训练迭代次数
ACTION_DIM = 3

# PPO
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01           # 动作分布熵系数（鼓励探索）
BETA = 0.01                   # 掩码信息瓶颈系数
LR = 3e-4
GRAD_CLIP = 5.0

# Mask
NUM_ENEMY_DROP = 2            # Actor 每步屏蔽的敌机数
NUM_ALLY_DROP = 1             # Actor 每步随机屏蔽的友机数

DEVICE = torch.device("cpu")


# ==============================================================================
#  辅助函数
# ==============================================================================
def _obs_to_tensor(per_agent_obs: dict, device: torch.device):
    """将单个 agent 的 numpy 观测转为 float32 Tensor（加 batch 维）。"""
    return {
        "ego_state":    torch.as_tensor(per_agent_obs["ego_state"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "ally_states":  torch.as_tensor(per_agent_obs["ally_states"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "enemy_states": torch.as_tensor(per_agent_obs["enemy_states"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "death_mask":   torch.as_tensor(per_agent_obs["death_mask"],
                                        dtype=torch.int64, device=device).unsqueeze(0),
    }


def _stack_obs(obs_list: list[dict[str, torch.Tensor]]):
    """将多个 agent 的观测 Tensor 沿 batch 维拼接。"""
    return {
        "ego_state":    torch.cat([o["ego_state"] for o in obs_list], dim=0),
        "ally_states":  torch.cat([o["ally_states"] for o in obs_list], dim=0),
        "enemy_states": torch.cat([o["enemy_states"] for o in obs_list], dim=0),
        "death_mask":   torch.cat([o["death_mask"] for o in obs_list], dim=0),
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float, lam: float):
    """广义优势估计 (GAE)。

    Args:
        rewards: (T,)  每一步的 reward
        values:  (T+1,) 包含最后一步的 bootstrap value
        dones:   (T,)  1 = episode 终止
    Returns:
        advantages: (T,)
        returns:    (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


# ==============================================================================
#  PPO 更新（逐 agent 梯度累积）
# ==============================================================================
def ppo_update(actor, critic, actor_opt, critic_opt, traj, blue_ids):
    """
    逐 agent 做 GRU 时序展开 + PPO Clip Loss，梯度累积后统一 step。

    每个 agent 的轨迹独立通过 GRU 展开，计算 loss 后立即 backward
    （释放该 agent 的 computation graph），梯度在 optimizer 中累积，
    最后统一 clip + step。
    """
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_mask_ent = 0.0
    total_action_ent = 0.0
    n_agents = 0

    actor_opt.zero_grad()
    critic_opt.zero_grad()

    for aid in blue_ids:
        t = traj[aid]
        T = len(t["act"])
        if T == 0:
            continue
        # 跳过从未存活的 agent
        if all(d > 0.5 for d in t["done"]):
            continue

        # ---- GAE ----
        rewards = torch.tensor(t["rew"], device=DEVICE)         # (T,)
        old_values = torch.cat(t["val"])                         # (T+1,)
        dones = torch.tensor(t["done"], device=DEVICE)           # (T,)

        advantages, returns = compute_gae(rewards, old_values, dones, GAMMA, GAE_LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 逐步 GRU 展开，重新计算 new_log_prob、value、entropy ----
        rnn_a = torch.zeros(1, 128, device=DEVICE)
        rnn_c = torch.zeros(1, 128, device=DEVICE)

        new_lps = []
        new_vals = []
        mask_ents = []
        action_ents = []

        for step_t in range(T):
            obs_t = t["obs"][step_t]       # dict of (1, ...) tensors
            act_t = t["act"][step_t]       # (3,) detached tensor

            dist_t, rnn_a, ment_t = actor(obs_t, NUM_ENEMY_DROP, NUM_ALLY_DROP, rnn_a)
            val_t, rnn_c = critic(obs_t, rnn_c)

            new_lp = dist_t.log_prob(act_t.unsqueeze(0)).sum(dim=-1)   # (1,)
            new_lps.append(new_lp)
            new_vals.append(val_t.squeeze(-1))                          # (1,) → scalar
            mask_ents.append(ment_t)
            action_ents.append(dist_t.entropy().mean())

        new_lp = torch.cat(new_lps)                      # (T,)
        old_lp = torch.stack(t["lp"]).squeeze(-1)          # (T,)
        new_vals = torch.cat(new_vals)                     # (T,)

        mask_ent_avg = torch.stack(mask_ents).mean()
        action_ent_avg = torch.stack(action_ents).mean()

        # ---- PPO Clip Loss (这个 agent 的贡献) ----
        ratio = torch.exp(new_lp - old_lp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        agent_actor_loss = policy_loss - ENTROPY_COEF * action_ent_avg \
                           - BETA * mask_ent_avg
        agent_critic_loss = F.mse_loss(new_vals, returns)

        # ---- 梯度累积（Actor 和 Critic 参数独立，无需 retain_graph） ----
        agent_actor_loss.backward()
        agent_critic_loss.backward()

        total_actor_loss += agent_actor_loss.item()
        total_critic_loss += agent_critic_loss.item()
        total_mask_ent += mask_ent_avg.item()
        total_action_ent += action_ent_avg.item()
        n_agents += 1

    if n_agents == 0:
        return {"actor_loss": 0.0, "critic_loss": 0.0,
                "mask_ent": 0.0, "action_ent": 0.0}

    # ---- 统一 clip grad + optimizer step ----
    nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)

    actor_opt.step()
    critic_opt.step()

    return {
        "actor_loss": total_actor_loss / n_agents,
        "critic_loss": total_critic_loss / n_agents,
        "mask_ent": total_mask_ent / n_agents,
        "action_ent": total_action_ent / n_agents,
    }


# ==============================================================================
#  主训练循环
# ==============================================================================
def run():
    # ---- 1. 环境初始化 ----
    env = UavCombatEnv(max_num_blue=MAX_NUM_BLUE, max_num_red=MAX_NUM_RED,
                       max_steps=MAX_STEPS)
    blue_ids = env.blue_ids
    red_ids = env.red_ids
    N_blue = len(blue_ids)
    N_red = len(red_ids)

    print(f"环境: {N_blue}v{N_red},  max_steps={MAX_STEPS}")
    print(f"蓝方 ID: {blue_ids}")
    print(f"红方 ID: {red_ids}")

    # ---- 2. 网络 + 优化器 ----
    actor = Actor(feature_dim=10, hidden_size=128, num_heads=4,
                  action_dim=ACTION_DIM, rnn_hidden_size=128).to(DEVICE)
    critic = Critic(feature_dim=10, hidden_size=128, num_heads=4,
                    rnn_hidden_size=128).to(DEVICE)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    print(f"Actor  params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    print("=" * 60)

    # ---- 3. 训练循环 ----
    for iteration in range(1, NUM_ITERATIONS + 1):
        obs_dict, _ = env.reset()

        # 初始化 RNN 隐藏状态（每个蓝方 agent 一份）
        rnn_hidden_actor = {aid: torch.zeros(1, 128, device=DEVICE)
                            for aid in blue_ids}
        rnn_hidden_critic = {aid: torch.zeros(1, 128, device=DEVICE)
                             for aid in blue_ids}

        # 轨迹缓存
        traj = {aid: {"obs": [], "act": [], "lp": [], "val": [],
                      "rew": [], "done": []}
                for aid in blue_ids}

        episode_reward_blue = 0.0
        episode_reward_red = 0.0

        # ------------------------------------------------------------------
        #  3a. Rollout: 采集一集数据
        # ------------------------------------------------------------------
        for step in range(1, MAX_STEPS + 1):
            actions = {}

            # ---- 蓝方：Actor 采样 ----
            blue_obs_tensors = []
            alive_blue_ids = []
            for aid in blue_ids:
                sim = env.blue_planes.get(aid)
                alive = sim is not None and sim.is_alive
                obs_tensor = _obs_to_tensor(obs_dict[aid], DEVICE)
                if alive:
                    blue_obs_tensors.append(obs_tensor)
                    alive_blue_ids.append(aid)
                else:
                    # 死亡 agent：记录占位数据
                    traj[aid]["obs"].append(obs_tensor)
                    traj[aid]["act"].append(torch.zeros(ACTION_DIM, device=DEVICE))
                    traj[aid]["lp"].append(torch.zeros(1, device=DEVICE))
                    traj[aid]["val"].append(torch.zeros(1, device=DEVICE))
                    traj[aid]["rew"].append(0.0)
                    traj[aid]["done"].append(1.0)
                    actions[aid] = np.zeros(ACTION_DIM, dtype=np.float32)

            if alive_blue_ids:
                batch_obs = _stack_obs(blue_obs_tensors)
                batch_rnn_a = torch.cat([rnn_hidden_actor[aid]
                                         for aid in alive_blue_ids], dim=0)

                with torch.no_grad():
                    action_dist, new_rnn_a, _mask_ent = actor(
                        batch_obs, NUM_ENEMY_DROP, NUM_ALLY_DROP, batch_rnn_a,
                    )
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)  # (N_alive,)

                    batch_rnn_c = torch.cat([rnn_hidden_critic[aid]
                                             for aid in alive_blue_ids], dim=0)
                    value, new_rnn_c = critic(batch_obs, batch_rnn_c)

                # 拆分回各 agent
                for i, aid in enumerate(alive_blue_ids):
                    rnn_hidden_actor[aid] = new_rnn_a[i:i + 1]
                    rnn_hidden_critic[aid] = new_rnn_c[i:i + 1]
                    traj[aid]["obs"].append(blue_obs_tensors[i])
                    traj[aid]["act"].append(action[i].detach())
                    traj[aid]["lp"].append(log_prob[i].detach().unsqueeze(0))
                    traj[aid]["val"].append(value[i].detach())
                    traj[aid]["rew"].append(0.0)   # 占位，step 后填充
                    traj[aid]["done"].append(0.0)
                    actions[aid] = action[i].cpu().numpy()

            # ---- 红方：随机动作 ----
            for aid in red_ids:
                actions[aid] = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)

            # ---- 环境步进 ----
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # 记录 reward 和 done
            for aid in blue_ids:
                if traj[aid]["rew"]:   # 有占位则填充
                    traj[aid]["rew"][-1] = float(rewards[aid])
                episode_reward_blue += rewards[aid]
            for aid in red_ids:
                episode_reward_red += rewards[aid]

            # 标记终止
            for aid in blue_ids:
                d = bool(terminated.get(aid, False) or truncated.get(aid, False))
                if d and traj[aid]["done"]:
                    traj[aid]["done"][-1] = 1.0

            if step == MAX_STEPS or all(terminated.values()) or all(truncated.values()):
                break

        episode_len = step

        # ---- Bootstrap：最后一步 Critic 估值（用于 GAE） ----
        with torch.no_grad():
            for aid in blue_ids:
                sim = env.blue_planes.get(aid)
                if sim is not None and sim.is_alive:
                    obs_t = _obs_to_tensor(obs_dict[aid], DEVICE)
                    val_last, _ = critic(obs_t, rnn_hidden_critic[aid])
                    traj[aid]["val"].append(val_last.squeeze(0).detach())
                else:
                    traj[aid]["val"].append(torch.zeros(1, device=DEVICE))

        # ------------------------------------------------------------------
        #  3b. PPO 更新
        # ------------------------------------------------------------------
        stats = ppo_update(actor, critic, actor_opt, critic_opt, traj, blue_ids)

        blue_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
        red_alive = sum(1 for s in env.red_planes.values() if s.is_alive)

        print(f"Iter {iteration:3d} | steps={episode_len:3d} | "
              f"blue_alive={blue_alive}/{N_blue} red_alive={red_alive}/{N_red} | "
              f"R_blue={episode_reward_blue:+7.1f} R_red={episode_reward_red:+7.1f} | "
              f"ActorLoss={stats['actor_loss']:+.4f} CriticLoss={stats['critic_loss']:+.4f} "
              f"MaskEnt={stats['mask_ent']:.4f} ActEnt={stats['action_ent']:.4f}")

    env.close()
    print("=" * 60)
    print("训练循环验证通过！")


if __name__ == "__main__":
    run()
