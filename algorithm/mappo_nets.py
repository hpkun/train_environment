"""
MAPPO Actor-Critic 网络 —— 均以 EntityObservationEncoder 为前置特征提取器。

- Actor:  特征提取 → GRU → MLP → Normal(mu, sigma) + 可学习 log_std
- Critic: 特征提取（无掩码、全观测）→ GRU → MLP → V(s)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import MaskVectorGenerator, EntityObservationEncoder


# ==============================================================================
#  1. Actor —— 策略网络
# ==============================================================================
class Actor(nn.Module):
    """
    随机策略 π(a|s, h)，输出连续动作空间上的高斯分布。

    结构：EntityObservationEncoder → GRUCell → MLP → mu (+ 可学习 sigma)
    """

    def __init__(self, feature_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, action_dim: int = 3,
                 rnn_hidden_size: int = 128, temperature: float = 0.1):
        super().__init__()

        # 偏置随机掩码生成器（仅 Actor 使用，选择性忽略低威胁敌机）
        self.mask_gen = MaskVectorGenerator(
            feature_dim=feature_dim, hidden_dim=64, temperature=temperature,
        )

        # 实体观测编码器（态势 → 定长特征向量）
        self.obs_encoder = EntityObservationEncoder(
            feature_dim=feature_dim, hidden_size=hidden_size, num_heads=num_heads,
        )

        # 序列记忆：GRU 单步单元
        self.rnn = nn.GRUCell(input_size=hidden_size, hidden_size=rnn_hidden_size)

        # 动作均值输出 MLP：rnn_hidden → 64 → action_dim，输出经 tanh 映射到 [-1, 1]
        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        # 可学习的独立对数标准差（与状态无关的探索噪声）
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs_dict: dict[str, torch.Tensor],
                num_enemy_drop: int, num_ally_drop: int,
                rnn_hidden: torch.Tensor):
        """
        Args:
            obs_dict:       {"ego_state": (B,10), "ally_states": (B,N_ally,10),
                             "enemy_states": (B,N_enemy,10), "death_mask": (B,1+N_ally+N_enemy)}
            num_enemy_drop: 屏蔽敌机数量（传给 MaskVectorGenerator）
            num_ally_drop:  随机屏蔽友机数量
            rnn_hidden:     GRU 隐藏状态 (B, rnn_hidden_size)
        Returns:
            action_dist:  torch.distributions.Normal
            rnn_hidden:   更新后的 GRU 隐藏状态 (B, rnn_hidden_size)
            entropy:      掩码分布的熵（标量，供信息瓶颈损失）
        """
        # ---- 1. 偏置随机掩码 ----
        enemy_mask, entropy = self.mask_gen(
            enemy_states=obs_dict["enemy_states"], num_drop=num_enemy_drop,
        )

        # ---- 2. 态势特征提取 ----
        features = self.obs_encoder(obs_dict, enemy_mask=enemy_mask,
                                    num_ally_drop=num_ally_drop)  # (B, hidden_size)

        # ---- 3. GRU 时序更新 ----
        rnn_hidden_new = self.rnn(features, rnn_hidden)  # (B, rnn_hidden_size)

        # ---- 4. 动作均值 ----
        mu = self.action_head(rnn_hidden_new)  # (B, action_dim)  值域 [-1, 1]

        # ---- 5. 动作标准差 ----
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)  # (action_dim,)
        sigma = sigma.unsqueeze(0).expand_as(mu)                 # (B, action_dim)

        action_dist = torch.distributions.Normal(mu, sigma)
        return action_dist, rnn_hidden_new, entropy


# ==============================================================================
#  2. Critic —— 价值网络
# ==============================================================================
class Critic(nn.Module):
    """
    中心化价值函数 V(s)，评估全局态势的期望回报。

    结构：EntityObservationEncoder（无掩码） → GRUCell → MLP → 标量 V

    CTDE 设计要点：Critic 看到全部观测（不屏蔽任何敌机/友机），
    以提供最准确的价值估计用于 Actor 更新。
    """

    def __init__(self, feature_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, rnn_hidden_size: int = 128):
        super().__init__()

        # 独立的特征提取器（与 Actor 不共享参数）
        self.obs_encoder = EntityObservationEncoder(
            feature_dim=feature_dim, hidden_size=hidden_size, num_heads=num_heads,
        )

        # 序列记忆
        self.rnn = nn.GRUCell(input_size=hidden_size, hidden_size=rnn_hidden_size)

        # 价值输出 MLP：rnn_hidden → 64 → 1
        self.value_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs_dict: dict[str, torch.Tensor],
                rnn_hidden: torch.Tensor):
        """
        Args:
            obs_dict:   (同上)
            rnn_hidden: GRU 隐藏状态 (B, rnn_hidden_size)
        Returns:
            value:      (B, 1) 状态价值
            rnn_hidden: 更新后的 GRU 隐藏状态 (B, rnn_hidden_size)
        """
        B = obs_dict["ego_state"].shape[0]
        N_enemy = obs_dict["enemy_states"].shape[1]
        device = obs_dict["ego_state"].device

        # ---- Critic 看到全部信息：全零掩码，不丢弃任何实体 ----
        enemy_mask = torch.zeros(B, N_enemy, device=device)   # 全 0 → 不屏蔽
        num_ally_drop = 0

        # ---- 态势特征提取 ----
        features = self.obs_encoder(obs_dict, enemy_mask=enemy_mask,
                                    num_ally_drop=num_ally_drop)  # (B, hidden_size)

        # ---- GRU 时序更新 ----
        rnn_hidden_new = self.rnn(features, rnn_hidden)  # (B, rnn_hidden_size)

        # ---- 价值估计 ----
        value = self.value_head(rnn_hidden_new)  # (B, 1)

        return value, rnn_hidden_new


# ==============================================================================
#  单元测试
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MAPPO Actor-Critic 网络测试")
    print("=" * 60)

    # ---- 模拟 10v10 环境输入 ----
    B = 4
    N_ally = 9
    N_enemy = 10
    FEAT_DIM = 11
    ACTION_DIM = 3
    HIDDEN = 128
    RNN_HIDDEN = 128

    torch.manual_seed(42)

    obs_dict = {
        "ego_state":    torch.randn(B, FEAT_DIM),
        "ally_states":  torch.randn(B, N_ally, FEAT_DIM),
        "enemy_states": torch.randn(B, N_enemy, FEAT_DIM),
        "death_mask":   torch.ones(B, 1 + N_ally + N_enemy),
    }
    obs_dict["death_mask"][0, 15] = 0
    obs_dict["death_mask"][2, 5]  = 0

    print(f"Batch: {B},  Ally: {N_ally},  Enemy: {N_enemy},  "
          f"Feat: {FEAT_DIM},  Action: {ACTION_DIM}")
    print("-" * 60)

    # ---- 实例化 ----
    actor = Actor(feature_dim=FEAT_DIM, hidden_size=HIDDEN, num_heads=4,
                  action_dim=ACTION_DIM, rnn_hidden_size=RNN_HIDDEN)
    critic = Critic(feature_dim=FEAT_DIM, hidden_size=HIDDEN, num_heads=4,
                    rnn_hidden_size=RNN_HIDDEN)

    print(f"Actor  parameters:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters:  {sum(p.numel() for p in critic.parameters()):,}")
    print(f"(Encoder 不共享，各自独立)")
    print("-" * 60)

    # ---- 初始 GRU 隐藏状态 ----
    rnn_hidden_actor = torch.zeros(B, RNN_HIDDEN)
    rnn_hidden_critic = torch.zeros(B, RNN_HIDDEN)

    # ---- Actor 前向传播 ----
    num_enemy_drop = 3
    num_ally_drop = 2
    action_dist, rnn_actor_out, entropy = actor(
        obs_dict, num_enemy_drop, num_ally_drop, rnn_hidden_actor,
    )

    print(f"[Actor]")
    print(f"  mu shape:         {action_dist.mean.shape}     (期望: ({B}, {ACTION_DIM}))")
    print(f"  sigma shape:      {action_dist.stddev.shape}   (期望: ({B}, {ACTION_DIM}))")
    print(f"  mu 值域:          [{action_dist.mean.min().item():+.3f}, "
          f"{action_dist.mean.max().item():+.3f}]  (应在 [-1, 1])")
    print(f"  entropy:          {entropy.item():.4f}")
    print(f"  rnn_hidden shape: {rnn_actor_out.shape}     (期望: ({B}, {RNN_HIDDEN}))")

    # 采样动作
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action).sum(dim=-1)
    print(f"  sampled action:   {action[0].detach().tolist()}")
    print(f"  log_prob shape:   {log_prob.shape}          (期望: ({B},))")
    print()

    # ---- Critic 前向传播 ----
    value, rnn_critic_out = critic(obs_dict, rnn_hidden_critic)

    print(f"[Critic]")
    print(f"  value shape:      {value.shape}      (期望: ({B}, 1))")
    print(f"  value range:      [{value.min().item():+.3f}, "
          f"{value.max().item():+.3f}]")
    print(f"  rnn_hidden shape: {rnn_critic_out.shape}     (期望: ({B}, {RNN_HIDDEN}))")
    print()

    # ---- 验证梯度 ----
    loss = (action_dist.mean.sum() + value.sum() + entropy * 0.01)
    loss.backward()

    actor_grad = sum(p.grad.norm().item() ** 2 for p in actor.parameters()
                     if p.grad is not None) ** 0.5
    critic_grad = sum(p.grad.norm().item() ** 2 for p in critic.parameters()
                      if p.grad is not None) ** 0.5
    print(f"Actor  grad norm:  {actor_grad:.2f}  (应 > 0)")
    print(f"Critic grad norm:  {critic_grad:.2f}  (应 > 0)")
    print()

    # ---- 多步 RNN 滚动测试 ----
    print("-" * 60)
    print("多步 RNN 滚动测试 (3 步):")
    rnn_a = torch.zeros(B, RNN_HIDDEN)
    rnn_c = torch.zeros(B, RNN_HIDDEN)
    for step in range(1, 4):
        # 模拟 env 返回的新观测（随机）
        new_obs = {
            "ego_state":    torch.randn(B, FEAT_DIM),
            "ally_states":  torch.randn(B, N_ally, FEAT_DIM),
            "enemy_states": torch.randn(B, N_enemy, FEAT_DIM),
            "death_mask":   torch.ones(B, 1 + N_ally + N_enemy),
        }
        dist, rnn_a, _ = actor(new_obs, num_enemy_drop, num_ally_drop, rnn_a)
        val, rnn_c = critic(new_obs, rnn_c)

        act = dist.sample()
        print(f"  Step {step}:  rnn_a.sum={rnn_a.sum().item():.2f},  "
              f"rnn_c.sum={rnn_c.sum().item():.2f},  "
              f"value.mean={val.mean().item():+.3f},  "
              f"act[0]={act[0].detach().tolist()}")

    print()
    print("=" * 60)
    print("全部测试通过!")
