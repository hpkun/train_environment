"""
核心前端网络架构 —— 论文《Biased Random Masked Attention MAPPO》复现。

提供两个核心类：
  1. MaskVectorGenerator  — Gumbel-Softmax Top-K 偏置随机掩码生成器
  2. EntityObservationEncoder — 实体观测编码器（多头注意力 + 掩码融合）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
#  1. MaskVectorGenerator —— 偏置随机掩码生成器
# ==============================================================================
class MaskVectorGenerator(nn.Module):
    """
    利用 Gumbel-Softmax Top-K 技巧，从敌机状态中学习"哪些敌机可以安全地忽略"，
    输出一个可微的二值掩码向量 m_B。

    输入：
        enemy_states  — (Batch, N_enemy, Feature_Dim=10)
        num_drop      — 期望屏蔽的敌机数量

    输出：
        mask          — (Batch, N_enemy)  1=屏蔽该敌机, 0=保留
        entropy       — 标量，Gumbel-Softmax 分布的信息熵（用于信息瓶颈损失）
    """

    def __init__(self, feature_dim: int = 11, hidden_dim: int = 64,
                 temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

        # 双层 MLP：10 → 64 → 1，输出每个敌机的"保留对数概率"
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, enemy_states: torch.Tensor, num_drop: int):
        """
        Args:
            enemy_states: (B, N_enemy, 10)
            num_drop:     每个样本需要屏蔽的敌机数量
        Returns:
            mask:    (B, N_enemy) 离散 0/1 掩码 (STE)
            entropy: 标量，分布平均熵
        """
        B, N_enemy, _ = enemy_states.shape

        # ---- 1. MLP 输出每个敌机的保留 Logits (B, N_enemy, 1) → squeeze → (B, N_enemy) ----
        logits = self.mlp(enemy_states).squeeze(-1)  # (B, N_enemy)

        # ---- 2. Gumbel 噪声 ----
        # Gumbel(0, 1) = -log(-log(U))  where U ~ Uniform(0, 1)
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u.clamp(1e-12, 1.0 - 1e-12)))

        # ---- 3. Gumbel-Softmax: 添加噪声后过温度参数 Softmax ----
        noisy_logits = (logits + gumbel_noise) / self.temperature
        soft_probs = F.softmax(noisy_logits, dim=-1)  # (B, N_enemy) 每行和为 1

        # ---- 4. 熵 (用于信息瓶颈正则) ----
        log_probs = F.log_softmax(noisy_logits, dim=-1)
        entropy = -torch.sum(soft_probs * log_probs, dim=-1).mean()  # 标量

        # ---- 5. Top-K 选取：概率最小的 num_drop 个实体 → 屏蔽 (mask=1) ----
        # topk(-soft_probs, k) = 最小的 k 个概率的索引
        # 安全钳：k 不能超过实体总数 (2v2 场景下实体极少)
        k_drop = min(num_drop, N_enemy)
        if k_drop == 0:
            return torch.zeros_like(soft_probs), entropy
        _, bottom_k_idx = torch.topk(-soft_probs, k_drop, dim=-1)

        discrete_mask = torch.zeros_like(soft_probs)          # (B, N_enemy) 全 0
        discrete_mask.scatter_(-1, bottom_k_idx, 1.0)        # 屏蔽位置置 1

        # ---- 6. Straight-Through Estimator ----
        # 前向：discrete_mask (0/1)；反向：梯度流经 soft_probs
        mask = discrete_mask + soft_probs - soft_probs.detach()

        return mask, entropy


# ==============================================================================
#  2. EntityObservationEncoder —— 实体观测编码器
# ==============================================================================
class EntityObservationEncoder(nn.Module):
    """
    处理变长实体序列（ego + allies + enemies），通过随机掩码 + 多头注意力
    提取全局态势特征。

    输入：
        obs_dict       — {"ego_state": (B,10), "ally_states": (B,N_ally,10),
                           "enemy_states": (B,N_enemy,10), "death_mask": (B,1+N_ally+N_enemy)}
        enemy_mask     — (B, N_enemy) 来自 MaskVectorGenerator，1=主动屏蔽
        num_ally_drop  — 随机屏蔽的友机数量

    输出：
        features       — (B, hidden_size=128) 定长态势特征向量
    """

    def __init__(self, feature_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        # 实体特征映射：10 → hidden_size
        self.fc_embed = nn.Linear(feature_dim, hidden_size)

        # 残差投影（ego 本身是 10 维，映射到 hidden_size）
        self.res_proj = nn.Linear(feature_dim, hidden_size)

        # 多头注意力（batch_first=True 适配 (B, Seq, Dim) 输入）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.hidden_size = hidden_size

    def forward(self, obs_dict: dict[str, torch.Tensor],
                enemy_mask: torch.Tensor,
                num_ally_drop: int):
        """
        Returns:
            features: (B, hidden_size)
        """
        ego_state   = obs_dict["ego_state"]    # (B, 10)
        ally_states = obs_dict["ally_states"]  # (B, N_ally, 10)
        enemy_states = obs_dict["enemy_states"] # (B, N_enemy, 10)
        death_mask  = obs_dict["death_mask"]    # (B, 1+N_ally+N_enemy)  0=死/不存在, 1=活

        B = ego_state.shape[0]
        N_ally = ally_states.shape[1]
        N_enemy = enemy_states.shape[1]
        device = ego_state.device

        # ------------------------------------------------------------------
        #  (a) 生成随机友军掩码 m_R  (B, N_ally)  — 1=屏蔽
        # ------------------------------------------------------------------
        if num_ally_drop > 0 and N_ally > 0:
            # torch.topk 随机选择 num_ally_drop 个位置
            rand_noise = torch.rand(B, N_ally, device=device)
            _, drop_idx = torch.topk(rand_noise, min(num_ally_drop, N_ally), dim=-1)
            ally_mask = torch.zeros(B, N_ally, device=device)
            ally_mask.scatter_(-1, drop_idx, 1.0)
        else:
            ally_mask = torch.zeros(B, N_ally, device=device)

        # ------------------------------------------------------------------
        #  (b) 拼接主动掩码：ego(永远不屏蔽) + ally_mask + enemy_mask
        # ------------------------------------------------------------------
        ego_active = torch.zeros(B, 1, device=device)           # (B, 1)  全 0
        active_mask = torch.cat([ego_active, ally_mask, enemy_mask], dim=-1)  # (B, 1+N_ally+N_enemy)
        # active_mask: 1 = 主动屏蔽该实体（算法决定不关注）

        # ------------------------------------------------------------------
        #  (c) 合并环境死亡掩码 → 最终 key_padding_mask
        #     key_padding_mask: True = 注意力忽略该位置
        #     忽略条件：实体已死亡 (death_mask==0) 或 被主动屏蔽 (active_mask==1)
        # ------------------------------------------------------------------
        dead = (death_mask == 0)               # (B, seq_len) True=已死
        key_padding_mask = dead | (active_mask > 0.5)  # (B, seq_len) True=忽略

        # ------------------------------------------------------------------
        #  (d) 拼接实体序列 + FC 嵌入
        # ------------------------------------------------------------------
        ego_expanded = ego_state.unsqueeze(1)                  # (B, 1, 10)
        entity_seq = torch.cat([ego_expanded, ally_states, enemy_states], dim=1)  # (B, 1+N_ally+N_enemy, 10)

        X = self.fc_embed(entity_seq)  # (B, Seq_Len, hidden_size)

        # ------------------------------------------------------------------
        #  (e) 多头注意力（Q=K=V=X，key_padding_mask 屏蔽无关实体）
        # ------------------------------------------------------------------
        attn_out, _ = self.attention(X, X, X, key_padding_mask=key_padding_mask)
        # attn_out: (B, Seq_Len, hidden_size)

        # ------------------------------------------------------------------
        #  (f) 取 ego 位置 (index 0) 的输出 + 残差连接 → 定长特征
        # ------------------------------------------------------------------
        ego_feat = attn_out[:, 0, :]                  # (B, hidden_size)
        residual = self.res_proj(ego_state)           # (B, hidden_size)
        features = ego_feat + residual                # (B, hidden_size)

        return features


# ==============================================================================
#  单元测试
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Biased Random Masked Attention MAPPO — 前端网络测试")
    print("=" * 60)

    # ---- 模拟 10v10 环境输入 ----
    B = 4
    N_ally = 9       # 除自己外 9 架友机
    N_enemy = 10      # 10 架敌机
    FEAT_DIM = 10     # 特征维度

    torch.manual_seed(42)

    # 构造 obs_dict
    obs_dict = {
        "ego_state":    torch.randn(B, FEAT_DIM),
        "ally_states":  torch.randn(B, N_ally, FEAT_DIM),
        "enemy_states": torch.randn(B, N_enemy, FEAT_DIM),
        "death_mask":   torch.ones(B, 1 + N_ally + N_enemy),  # 全部存活
    }
    # 随机标记一些死亡实体
    obs_dict["death_mask"][0, 15] = 0  # batch 0 的第 15 号实体已死
    obs_dict["death_mask"][1, 3]  = 0
    obs_dict["death_mask"][2, 18] = 0

    print(f"Batch Size:       {B}")
    print(f"ego_state:        {obs_dict['ego_state'].shape}")
    print(f"ally_states:      {obs_dict['ally_states'].shape}")
    print(f"enemy_states:     {obs_dict['enemy_states'].shape}")
    print(f"death_mask:       {obs_dict['death_mask'].shape}")
    print(f"  death_mask[0]:  {obs_dict['death_mask'][0].int().tolist()}")
    print("-" * 60)

    # ---- 实例化模块 ----
    mask_gen = MaskVectorGenerator(feature_dim=FEAT_DIM, hidden_dim=64, temperature=0.5)
    encoder  = EntityObservationEncoder(feature_dim=FEAT_DIM, hidden_size=128, num_heads=4)

    print(f"MaskVectorGenerator parameters: {sum(p.numel() for p in mask_gen.parameters()):,}")
    print(f"EntityObservationEncoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print("-" * 60)

    # ---- 前向传播 ----
    num_enemy_drop = 3
    num_ally_drop  = 2

    enemy_mask, entropy = mask_gen(enemy_states=obs_dict["enemy_states"],
                                   num_drop=num_enemy_drop)
    print(f"[MaskVectorGenerator]")
    print(f"  enemy_mask shape: {enemy_mask.shape}  (期望: ({B}, {N_enemy}))")
    print(f"  enemy_mask[0]:    {enemy_mask[0].detach().round().int().tolist()}")
    print(f"  masked count[0]:  {int(enemy_mask[0].sum().item())} / {N_enemy}  (期望屏蔽 {num_enemy_drop})")
    print(f"  entropy:          {entropy.item():.4f}")
    print(f"  mask requires_grad: {enemy_mask.requires_grad}")
    print()

    features = encoder(obs_dict=obs_dict, enemy_mask=enemy_mask,
                       num_ally_drop=num_ally_drop)
    print(f"[EntityObservationEncoder]")
    print(f"  features shape:   {features.shape}  (期望: ({B}, 128))")

    # ---- 验证梯度回传 ----
    loss = features.sum() + entropy * 0.01
    loss.backward()

    grad_norm = 0.0
    for p in mask_gen.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    for p in encoder.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    print(f"  gradient norm:    {grad_norm ** 0.5:.4f}  (应 > 0)")
    print()

    # ---- 规模泛化测试：6v6 → 10v10 ----
    print("-" * 60)
    print("规模泛化测试: 6v6 输入 → 前向传播")
    N_ally_small = 5
    N_enemy_small = 6
    obs_small = {
        "ego_state":    torch.randn(B, FEAT_DIM),
        "ally_states":  torch.randn(B, N_ally_small, FEAT_DIM),
        "enemy_states": torch.randn(B, N_enemy_small, FEAT_DIM),
        "death_mask":   torch.ones(B, 1 + N_ally_small + N_enemy_small),
    }
    enemy_mask_small, _ = mask_gen(obs_small["enemy_states"], num_drop=2)
    features_small = encoder(obs_small, enemy_mask_small, num_ally_drop=1)
    print(f"  6v6 features:     {features_small.shape}  (期望: ({B}, 128))")

    print("=" * 60)
    print("全部测试通过!")
