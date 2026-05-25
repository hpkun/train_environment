"""
train_ppo.py —— 工业级多进程并行 MAPPO 训练脚本

严格复现论文《Biased Random Masked Attention MAPPO》中的训练循环与损失函数。

架构:
- 32 个并行环境 (SubprocVecEnv + multiprocessing)
- 红方 (6 架) 由 Actor/Critic 控制，蓝方 (6 架) 规则策略
- GAE + PPO Clip + 信息瓶颈掩码损失 (KL 散度)
- 三个独立优化器: Actor / Critic / Mask Generator

用法:  python train_ppo.py
"""
from __future__ import annotations

import os
import sys

# ---- 多进程性能：禁止底层库的线程池竞争 ----
# 在 import numpy / torch 前设置环境变量，在 import torch 后锁死线程数。
# 防止 SubprocVecEnv 多进程 + PyTorch 多线程引发 CPU 线程颠簸。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import glob
import multiprocessing as mp
import time
from collections import Counter, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass

# NB: UavCombatEnv is imported inside _worker() AFTER stdout is silenced,
# so that the JSBSim C++ startup banner is suppressed in every worker.
from algorithm.mappo_nets import Actor, Critic
from rule_based_agent import blue_coordinated_actions


def _cleanup_rotating_checkpoints(directory: str, prefix: str, keep: int = 5):
    """删除超出保留数量的旧轮转 checkpoint 文件。"""
    pattern = os.path.join(directory, f"{prefix}_*.pt")
    files = sorted(glob.glob(pattern))
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass


# ==============================================================================
#  配置 (严格对齐论文参数)
# ==============================================================================
class Config:
    # ---- 环境 (对标论文 6v6) ----
    num_envs: int = 32          # 论文原版 Rollout threads
    num_red: int = 6            # 6v6 训练场景
    num_blue: int = 6
    max_episode_length: int = 1400  # 论文一致
    action_dim: int = 3

    # ---- PPO (适配大规模数据) ----
    replay_buffer_size: int = 2000  # 论文原版 Replay buffer size
    n_update_epochs: int = 10        # 每轮数据的 PPO epoch 数
    n_minibatches: int = 4           # minibatch 切分数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 5.0

    # ---- 学习率 (三个独立优化器) ----
    actor_lr: float = 0.0002
    critic_lr: float = 0.0005
    mask_lr: float = 0.0005

    # ---- 损失系数 ----
    entropy_coef: float = 0.05       # β (动作熵系数)
    temperature: float = 0.1         # τ (Gumbel-Softmax 温度)

    # ---- 网络 ----
    feature_dim: int = 11  # 11-dim entity vec: Δn,Δe,Δu,AO,TA,R,V_tgt,sinφ,cosφ,sinθ,cosθ
    hidden_size: int = 128
    rnn_hidden_size: int = 128
    num_heads: int = 4

    # ---- 训练总步数 ----
    total_env_steps: int = 10_000_000


# ==============================================================================
#  SubprocVecEnv —— 多进程并行环境
# ==============================================================================
def _worker(remote: mp.connection.Connection,
            parent_remote: mp.connection.Connection,
            env_kwargs: dict):
    """子进程入口：自行构造 UavCombatEnv 并循环响应指令。"""
    parent_remote.close()

    # Worker 进程也锁定单线程：NumPy 操作（obs 拼接等）不使用 OpenMP 线程池
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ---- Permanently silence all C / C++ / Win32 stdout/stderr ----
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        import ctypes
        _crt = ctypes.CDLL("msvcrt")
        _crt._dup2.restype = ctypes.c_int
        _nul_fd = os.open(os.devnull, os.O_WRONLY)
        _crt._dup2(_nul_fd, 1)
        _crt._dup2(_nul_fd, 2)
        os.close(_nul_fd)
        _krn = ctypes.WinDLL("kernel32", use_last_error=True)
        _h_nul = _krn.CreateFileW("NUL", 0x40000000, 3, None, 3, 0x80, None)
        if _h_nul not in (-1, None):
            _krn.SetStdHandle(-11, _h_nul)
            _krn.SetStdHandle(-12, _h_nul)
    else:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 1)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    # Lazy import after silence — JSBSim C++ banner is suppressed
    from my_uav_env import UavCombatEnv
    env = UavCombatEnv(**env_kwargs)
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, rewards, terminated, truncated, info = env.step(data)
            dones = {}
            for aid in env.agent_ids:
                dones[aid] = bool(terminated.get(aid, False) or truncated.get(aid, False))
            if all(dones.values()):
                obs, _ = env.reset()
            remote.send((obs, rewards, dones, info))

        elif cmd == "reset":
            obs, info = env.reset()
            remote.send(obs)

        elif cmd == "call":
            # Generic method call: data = (method_name, args, kwargs)
            method_name, args, kwargs = data
            result = getattr(env, method_name)(*args, **kwargs)
            remote.send(result)

        elif cmd == "close":
            remote.close()
            break


class SubprocVecEnv:
    """多进程向量化环境封装 (Windows spawn 兼容)。

    启动进程间加入小延迟以避免 JSBSim 并发的 I/O 争用。
    """

    def __init__(self, num_envs: int, env_kwargs: dict, startup_delay: float = 0.5):
        self.n_envs = num_envs
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []

        for i in range(num_envs):
            p = ctx.Process(
                target=_worker,
                args=(self.work_remotes[i], self.remotes[i], env_kwargs),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            self.work_remotes[i].close()
            if i < num_envs - 1:
                time.sleep(startup_delay)  # 错峰启动，减轻磁盘 I/O 峰值

    def reset(self, timeout: float = 300.0) -> list[dict]:
        """发送 reset 指令到所有 worker，带超时等待。"""
        for remote in self.remotes:
            remote.send(("reset", None))
        results = []
        for i, remote in enumerate(self.remotes):
            if remote.poll(timeout):
                results.append(remote.recv())
            else:
                raise TimeoutError(
                    f"Worker {i} (PID {self.processes[i].pid}) did not "
                    f"respond to 'reset' within {timeout:.0f}s — "
                    f"likely JSBSim init hang or crash."
                )
        return results

    def step(self, actions_list: list[dict], timeout: float = 60.0) -> tuple:
        """发送动作列表 (每环境一个 dict)，返回 (obs, rewards, dones, infos) 的 tuple-of-lists。"""
        for remote, actions in zip(self.remotes, actions_list):
            remote.send(("step", actions))
        results = []
        for i, remote in enumerate(self.remotes):
            if remote.poll(timeout):
                results.append(remote.recv())
            else:
                raise TimeoutError(
                    f"Worker {i} (PID {self.processes[i].pid}) did not "
                    f"respond to 'step' within {timeout:.0f}s."
                )
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def env_method(self, method_name: str, *args, **kwargs):
        """Call a method on every remote env and return the list of results."""
        for remote in self.remotes:
            remote.send(("call", (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _fetch_blue_own_positions(vec_env) -> list[dict]:
    """Fetch blue ownship positions from worker envs for rule-policy patrol."""

    raw = vec_env.env_method("get_blue_own_positions")
    return [item if isinstance(item, dict) else {} for item in raw]


# ==============================================================================
#  辅助函数
# ==============================================================================
def _obs_np_to_tensor(obs_np: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """单个 agent 的 numpy 观测 → float32 Tensor (添加 batch 维)。"""
    return {
        "ego_state":    torch.as_tensor(obs_np["ego_state"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "ally_states":  torch.as_tensor(obs_np["ally_states"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "enemy_states": torch.as_tensor(obs_np["enemy_states"],
                                        dtype=torch.float32, device=device).unsqueeze(0),
        "death_mask":   torch.as_tensor(obs_np["death_mask"],
                                        dtype=torch.int64, device=device).unsqueeze(0),
    }


def _stack_obs(obs_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """将多个 agent 的观测 Tensor 沿 batch 维拼接。"""
    return {
        "ego_state":    torch.cat([o["ego_state"] for o in obs_list], dim=0),
        "ally_states":  torch.cat([o["ally_states"] for o in obs_list], dim=0),
        "enemy_states": torch.cat([o["enemy_states"] for o in obs_list], dim=0),
        "death_mask":   torch.cat([o["death_mask"] for o in obs_list], dim=0),
    }


def _obs_tensor_slice(obs: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    """从 batch-obs 中取出第 idx 个样本 (保留 batch 维)。"""
    return {
        "ego_state":    obs["ego_state"][idx:idx + 1],
        "ally_states":  obs["ally_states"][idx:idx + 1],
        "enemy_states": obs["enemy_states"][idx:idx + 1],
        "death_mask":   obs["death_mask"][idx:idx + 1],
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float, lam: float):
    """广义优势估计 (GAE)。

    Args:
        rewards: (T,) 每一步的 reward
        values:  (T+1,) 最后一步为 bootstrap value
        dones:   (T,) 1=终止, 0=继续
    Returns:
        advantages: (T,), returns: (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


def gaussian_kl(mu_full: torch.Tensor, sigma_full: torch.Tensor,
                mu_masked: torch.Tensor, sigma_masked: torch.Tensor) -> torch.Tensor:
    r"""两个对角高斯分布间的 KL 散度，逐样本计算。

    论文公式 (44):
      D_{KL}(p_full || p_masked) = 0.5 * Σ_i [ σ1_i²/σ2_i²
        + (μ2_i - μ1_i)² / σ2_i² - 1 + 2·log(σ2_i / σ1_i) ]

    Args:
        mu_full:    (B, D) 无掩码均值
        sigma_full: (B, D) 无掩码标准差
        mu_masked:  (B, D) 有掩码均值
        sigma_masked:(B, D) 有掩码标准差

    Returns:
        kl: (B,) 每个样本的 KL 散度 (对 D 维求和)
    """
    var1 = sigma_full ** 2       # σ1²
    var2 = sigma_masked ** 2     # σ2²

    kl = 0.5 * (
        var1 / var2
        + (mu_masked - mu_full) ** 2 / var2
        - 1.0
        + 2.0 * (torch.log(sigma_masked) - torch.log(sigma_full))
    )
    return kl.sum(dim=-1)  # (B,) 在动作维度求和


# ==============================================================================
#  经验缓冲区
# ==============================================================================
class RolloutBuffer:
    """存储一次 PPO rollout 的轨迹数据。

    维度约定:
      S = 每环境收集步数 (num_steps)
      E = 并行环境数 (num_envs)
      A = 红方 agent 数 (num_red)
    """

    def __init__(self, num_steps: int, num_envs: int, num_red: int, action_dim: int,
                 rnn_hidden_size: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_red = num_red

        # 观测以 list-of-dicts 存储 (可变实体数，无法预分配为固定数组)
        self.obs: list[list[list[dict | None]]] = [
            [[None for _ in range(num_red)] for _ in range(num_envs)]
            for _ in range(num_steps)
        ]

        # 动作 / 奖励 / 价值 / 对数概率 / done 标志 → 预分配 numpy 数组
        shape_sa = (num_steps, num_envs, num_red, action_dim)
        shape_s = (num_steps, num_envs, num_red)

        self.actions = np.zeros(shape_sa, dtype=np.float32)
        self.rewards = np.zeros(shape_s, dtype=np.float32)
        self.values = np.zeros(shape_s, dtype=np.float32)
        self.log_probs = np.zeros(shape_s, dtype=np.float32)
        self.dones = np.zeros(shape_s, dtype=np.float32)
        self.alive = np.zeros(shape_s, dtype=bool)

        # 掩码信息 (每 env 每步)
        self.mask_entropy = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.num_enemy_drop = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.num_ally_drop = np.zeros((num_steps, num_envs), dtype=np.int32)

        # 初始 RNN 隐藏状态 (用于 update 时初始化 GRU 时序展开)
        self.rnn_actor_init = np.zeros(
            (num_envs, num_red, rnn_hidden_size), dtype=np.float32,
        )
        self.rnn_critic_init = np.zeros(
            (num_envs, num_red, rnn_hidden_size), dtype=np.float32,
        )
        # 最终 RNN 隐藏状态 (用于 bootstrap 估值)
        self.rnn_actor_final = np.zeros(
            (num_envs, num_red, rnn_hidden_size), dtype=np.float32,
        )
        self.rnn_critic_final = np.zeros(
            (num_envs, num_red, rnn_hidden_size), dtype=np.float32,
        )

    def store_step(self, step: int, env_idx: int, agent_idx: int,
                   obs_np: dict, action: np.ndarray, reward: float,
                   value: float, log_prob: float, done: float, alive: bool):
        """存储单个 agent 在一个时间步的数据。"""
        self.obs[step][env_idx][agent_idx] = obs_np
        self.actions[step, env_idx, agent_idx] = action
        self.rewards[step, env_idx, agent_idx] = reward
        self.values[step, env_idx, agent_idx] = value
        self.log_probs[step, env_idx, agent_idx] = log_prob
        self.dones[step, env_idx, agent_idx] = done
        self.alive[step, env_idx, agent_idx] = alive

    def store_env_meta(self, step: int, env_idx: int,
                       mask_ent: float, num_ed: int, num_ad: int):
        """存储每步 / 每环境的掩码元信息。"""
        self.mask_entropy[step, env_idx] = mask_ent
        self.num_enemy_drop[step, env_idx] = num_ed
        self.num_ally_drop[step, env_idx] = num_ad

    def flatten_for_update(self, device: torch.device) -> dict:
        """将缓冲区数据展平为训练用的 Tensor 字典 (仅保留 alive 时间步)。"""
        alive_mask = self.alive  # (S, E, A)

        # 展平所有维度，仅取 alive=True 的条目
        idx = np.where(alive_mask)
        n_total = len(idx[0])
        if n_total == 0:
            return {"n_total": 0}

        # obs: 逐个索引复制 (无法直接 numpy 切片，因为是 dict)
        obs_flat: list[dict[str, np.ndarray]] = []
        for s, e, a in zip(*idx):
            obs_flat.append(self.obs[s][e][a])

        return {
            "n_total": n_total,
            "obs": obs_flat,  # list of numpy dicts
            "actions": torch.as_tensor(self.actions[alive_mask], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[alive_mask], dtype=torch.float32, device=device),
            "values": torch.as_tensor(self.values[alive_mask], dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(self.log_probs[alive_mask], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[alive_mask], dtype=torch.float32, device=device),
            # 元信息 (按 step+env，不按 agent 重复)
            "mask_entropy": torch.as_tensor(self.mask_entropy, dtype=torch.float32, device=device),
            "num_enemy_drop": torch.as_tensor(self.num_enemy_drop, dtype=torch.int64, device=device),
            "num_ally_drop": torch.as_tensor(self.num_ally_drop, dtype=torch.int64, device=device),
            # 索引映射: step 和 env 对应每个 alive 条目的位置
            "step_idx": torch.as_tensor(idx[0], dtype=torch.int64, device=device),
            "env_idx": torch.as_tensor(idx[1], dtype=torch.int64, device=device),
            "agent_idx": torch.as_tensor(idx[2], dtype=torch.int64, device=device),
            # 初始 RNN 状态
            "rnn_actor_init": torch.as_tensor(self.rnn_actor_init, dtype=torch.float32, device=device),
            "rnn_critic_init": torch.as_tensor(self.rnn_critic_init, dtype=torch.float32, device=device),
        }


# ==============================================================================
#  PPO 更新 (论文核心: 三个 Loss)
# ==============================================================================
def ppo_update(config: Config, actor: Actor, critic: Critic,
               actor_opt: torch.optim.Adam,
               critic_opt: torch.optim.Adam,
               mask_opt: torch.optim.Adam,
               buffer: RolloutBuffer, device: torch.device):
    """
    对一次 rollout 数据执行多 epoch PPO 更新。

    包含三个损失:
      A. Critic Loss  — MSE(V(s), Return)
      B. Actor Loss   — PPO Clip + 动作熵
      C. Mask Loss    — KL(π_full || π_masked) - β·H(mask)  (信息瓶颈)
    """
    data = buffer.flatten_for_update(device)
    n_total = data["n_total"]
    if n_total == 0:
        return {}

    # ---- 将展平数据重组为 per-env-per-agent 轨迹 ----
    # 遍历每个 (env, agent) 组合，构建有序的时序轨迹
    num_steps = buffer.num_steps
    num_envs = buffer.num_envs
    num_red = buffer.num_red

    # 预计算每个 (env, agent) 的 GAE advantages 和 returns
    # 存储为 per-env-per-agent 的 list-of-tensors
    traj_data: list[list[dict | None]] = [
        [None for _ in range(num_red)] for _ in range(num_envs)
    ]

    for env_idx in range(num_envs):
        for agent_idx in range(num_red):
            # 收集该 agent 在整个 rollout 中的所有 alive 时间步
            t_obs: list[dict[str, np.ndarray]] = []
            t_act: list[np.ndarray] = []
            t_rew: list[float] = []
            t_val: list[float] = []
            t_lp: list[float] = []
            t_done: list[float] = []
            t_steps: list[int] = []  # 对应 rollout 中的原始步索引

            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_obs.append(buffer.obs[step][env_idx][agent_idx])
                    t_act.append(buffer.actions[step, env_idx, agent_idx])
                    t_rew.append(buffer.rewards[step, env_idx, agent_idx])
                    t_val.append(buffer.values[step, env_idx, agent_idx])
                    t_lp.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_done.append(buffer.dones[step, env_idx, agent_idx])
                    t_steps.append(step)

            if len(t_act) == 0:
                continue

            # 计算该 agent 的 bootstrap value
            # (如果最后一个时间步未终止，需要用 Critic 再估值一次作为 V_{T+1})
            T = len(t_act)
            vals_with_bootstrap = np.array(t_val + [0.0], dtype=np.float32)

            # 如果 agent 在最后步仍然存活 → 需要 bootstrap
            if t_done[-1] == 0.0:
                # 使用 rollout 最后一步存储的 obs 做 Critic 估值
                last_obs_np = t_obs[-1]
                last_obs_t = _obs_np_to_tensor(last_obs_np, device)
                rnn_c_init = torch.as_tensor(
                    buffer.rnn_critic_init[env_idx, agent_idx], device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    val_bootstrap, _ = critic(last_obs_t, rnn_c_init)
                vals_with_bootstrap[T] = val_bootstrap.item()

            # GAE
            rew_t = torch.tensor(t_rew, dtype=torch.float32, device=device)
            val_t = torch.tensor(vals_with_bootstrap, dtype=torch.float32, device=device)
            don_t = torch.tensor(t_done, dtype=torch.float32, device=device)

            advantages, returns = compute_gae(rew_t, val_t, don_t,
                                              config.gamma, config.gae_lambda)
            if advantages.numel() > 1:
                adv_std = advantages.std()
            else:
                adv_std = torch.std(advantages, correction=0)
            if adv_std <= 1e-8 or torch.isnan(adv_std):
                adv_std = 1.0
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            traj_data[env_idx][agent_idx] = {
                "obs": t_obs,
                "act": torch.as_tensor(np.stack(t_act), dtype=torch.float32, device=device),
                "old_lp": torch.as_tensor(np.array(t_lp), dtype=torch.float32, device=device),
                "advantages": advantages,
                "returns": returns,
                "steps": t_steps,  # rollout 中的原始步索引 (用于查询 drop count)
            }

    # ---- PPO Epochs ----
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_mask_loss = 0.0
    total_kl = 0.0
    total_ment = 0.0
    n_updates = 0

    # 收集所有有效 agent 及其轨迹长度，用于 minibatch 采样
    agent_keys = []
    for env_idx in range(num_envs):
        for agent_idx in range(num_red):
            td = traj_data[env_idx][agent_idx]
            if td is not None and len(td["act"]) > 0:
                agent_keys.append((env_idx, agent_idx, len(td["act"])))

    if not agent_keys:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "mask_loss": 0.0,
                "kl": 0.0, "mask_ent": 0.0}

    # 将 agent_keys 按轨迹长度分组 (相同长度可拼 batch)
    # 为简单及正确性：逐 agent 做 GRU 展开 → 逐 agent backward → 梯度累积
    for epoch in range(config.n_update_epochs):
        # 随机打乱 agent 顺序
        np.random.shuffle(agent_keys)

        # 切分为 minibatches
        n_agents = len(agent_keys)
        batch_size = max(1, n_agents // config.n_minibatches)
        for mb_start in range(0, n_agents, batch_size):
            mb_keys = agent_keys[mb_start:mb_start + batch_size]

            # 对 minibatch 中的每个 agent 独立处理，累积梯度
            mb_actor_loss = 0.0
            mb_critic_loss = 0.0
            mb_mask_loss = 0.0
            mb_kl = 0.0
            mb_ment = 0.0
            mb_count = 0

            for env_idx, agent_idx, T in mb_keys:
                td = traj_data[env_idx][agent_idx]
                if td is None:
                    continue

                # ---- 1. 预计算全观测轨迹 (无梯度，作为 KL 的固定目标) ----
                rnn_a_init = torch.as_tensor(
                    buffer.rnn_actor_init[env_idx, agent_idx], device=device,
                ).unsqueeze(0)

                dists_full: list[tuple[torch.Tensor, torch.Tensor]] = []
                with torch.no_grad():
                    rnn_full = rnn_a_init.clone()
                    for t in range(T):
                        obs_t = _obs_np_to_tensor(td["obs"][t], device)
                        dist_f, rnn_full, _ = actor(obs_t, num_enemy_drop=0,
                                                    num_ally_drop=0,
                                                    rnn_hidden=rnn_full)
                        dists_full.append((dist_f.mean.clone(), dist_f.stddev.clone()))

                # ---- 2. 有掩码轨迹展开 (跟踪梯度: Actor + Mask + Critic) ----
                rnn_a = rnn_a_init.clone()
                rnn_c = torch.as_tensor(
                    buffer.rnn_critic_init[env_idx, agent_idx], device=device,
                ).unsqueeze(0)

                new_lps = []
                new_vals = []
                mask_ents = []
                action_ents = []
                kl_divs = []

                t_steps = td["steps"]  # rollout 原始步索引列表

                for i, t in enumerate(range(T)):
                    obs_t = _obs_np_to_tensor(td["obs"][t], device)
                    act_t = td["act"][t:t + 1]  # (1, 3)
                    rollout_step = t_steps[t]   # 该轨迹步在 rollout 中的真实步索引
                    num_ed = int(buffer.num_enemy_drop[rollout_step, env_idx])
                    num_ad = int(buffer.num_ally_drop[rollout_step, env_idx])

                    # ---- 有掩码前向 (Masked Obs) ----
                    dist_m, rnn_a, ment_t = actor(obs_t, num_enemy_drop=num_ed,
                                                  num_ally_drop=num_ad,
                                                  rnn_hidden=rnn_a)

                    # PPO 对数概率
                    new_lp = dist_m.log_prob(act_t).sum(dim=-1)  # (1,)
                    new_lps.append(new_lp)
                    mask_ents.append(ment_t)
                    action_ents.append(dist_m.entropy().mean())

                    # ---- KL 散度: D_KL(π_full || π_masked) 仅梯度流经 π_masked ----
                    mu_f, sigma_f = dists_full[t]
                    kl_t = gaussian_kl(mu_f, sigma_f, dist_m.mean, dist_m.stddev).mean()
                    kl_divs.append(kl_t)

                    # ---- Critic 前向 (无掩码，全局观测) ----
                    val_t, rnn_c = critic(obs_t, rnn_c)
                    new_vals.append(val_t.squeeze(-1))  # (1,) → scalar

                # ---- 拼接当前 agent 的各时间步 ----
                new_lp = torch.cat(new_lps)                    # (T,)
                old_lp = td["old_lp"]                           # (T,)
                new_vals = torch.cat(new_vals)                  # (T,)
                advantages = td["advantages"]                   # (T,)
                returns = td["returns"]                         # (T,)
                mask_ent_avg = torch.stack(mask_ents).mean()
                action_ent_avg = torch.stack(action_ents).mean()
                kl_avg = torch.stack(kl_divs).mean()

                # ---- A. Actor Loss (PPO Clip) ----
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                agent_actor_loss = policy_loss - config.entropy_coef * action_ent_avg

                # ---- B. Critic Loss (MSE) ----
                agent_critic_loss = F.mse_loss(new_vals, returns)

                # ---- C. Mask Loss (信息瓶颈) ----
                agent_mask_loss = kl_avg - config.entropy_coef * mask_ent_avg

                # ---- 梯度累积 ----
                agent_actor_loss.backward(retain_graph=True)
                agent_critic_loss.backward(retain_graph=True)
                agent_mask_loss.backward()

                mb_actor_loss += agent_actor_loss.item()
                mb_critic_loss += agent_critic_loss.item()
                mb_mask_loss += agent_mask_loss.item()
                mb_kl += kl_avg.item()
                mb_ment += mask_ent_avg.item()
                mb_count += 1

            if mb_count == 0:
                continue

            # ---- 梯度裁剪 + 优化器步进 ----
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)

            actor_opt.step()
            critic_opt.step()
            mask_opt.step()

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            mask_opt.zero_grad()

            total_actor_loss += mb_actor_loss / mb_count
            total_critic_loss += mb_critic_loss / mb_count
            total_mask_loss += mb_mask_loss / mb_count
            total_kl += mb_kl / mb_count
            total_ment += mb_ment / mb_count
            n_updates += 1

    return {
        "actor_loss": total_actor_loss / max(1, n_updates),
        "critic_loss": total_critic_loss / max(1, n_updates),
        "mask_loss": total_mask_loss / max(1, n_updates),
        "kl": total_kl / max(1, n_updates),
        "mask_ent": total_ment / max(1, n_updates),
    }


# ==============================================================================
#  主训练循环
# ==============================================================================
def main():
    config = Config()

    # ---- 设备选择 ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _cuda_name = torch.cuda.get_device_name(0)
        print(f"设备: CUDA ({_cuda_name})")
    else:
        device = torch.device("cpu")
        print(f"设备: cpu (CUDA 不可用 — PyTorch 未安装 CUDA 版本或驱动缺失)")
        print(f"      安装 GPU PyTorch: pip install torch --index-url "
              f"https://download.pytorch.org/whl/cu121")

    print(f"并行环境: {config.num_envs}")
    print(f"场景: {config.num_red}v{config.num_blue} (红方 RL, 蓝方规则)")
    print(f"缓冲区步数: {config.replay_buffer_size}")
    # 掩码安全范围诊断
    _n_ally = config.num_red - 1
    _n_enemy = config.num_blue
    print(f"掩码采样 (enemy): U(0, {min(2, max(0, _n_enemy - 1))})  "
          f"掩码采样 (ally):  U(0, {min(2, _n_ally)})")

    # ---- 1. 创建并行环境 (32 进程错峰启动) ----
    num_steps = config.replay_buffer_size // config.num_envs
    env_kwargs = dict(max_num_blue=config.num_blue, max_num_red=config.num_red,
                      max_steps=config.max_episode_length)
    print(f"正在启动 {config.num_envs} 个 worker 进程 (错峰 0.5s, 预计 "
          f"{config.num_envs * 0.5:.0f}s)...", flush=True)
    vec_env = SubprocVecEnv(config.num_envs, env_kwargs)

    red_ids = [f"red_{i}" for i in range(config.num_red)]
    blue_ids = [f"blue_{i}" for i in range(config.num_blue)]

    # ---- 2. 初始化网络 ----
    actor = Actor(
        feature_dim=config.feature_dim, hidden_size=config.hidden_size,
        num_heads=config.num_heads, action_dim=config.action_dim,
        rnn_hidden_size=config.rnn_hidden_size, temperature=config.temperature,
    ).to(device)
    critic = Critic(
        feature_dim=config.feature_dim, hidden_size=config.hidden_size,
        num_heads=config.num_heads, rnn_hidden_size=config.rnn_hidden_size,
    ).to(device)

    print(f"Actor  params:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params:  {sum(p.numel() for p in critic.parameters()):,}")

    # ---- 3. 三个独立优化器 ----
    # Actor (不含 mask_gen)
    actor_params = [p for n, p in actor.named_parameters() if "mask_gen" not in n]
    actor_opt = torch.optim.Adam(actor_params, lr=config.actor_lr)

    # Critic
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # Mask Generator
    mask_opt = torch.optim.Adam(actor.mask_gen.parameters(), lr=config.mask_lr)

    # ---- 4. 初始 RNN 状态 (per env, per red agent) ----
    rnn_hidden_actor = np.zeros(
        (config.num_envs, config.num_red, config.rnn_hidden_size), dtype=np.float32,
    )
    rnn_hidden_critic = np.zeros(
        (config.num_envs, config.num_red, config.rnn_hidden_size), dtype=np.float32,
    )

    # ---- 5. 重置所有环境 (384 个 JSBSim 实例初始化, 预计 2-5 分钟) ----
    print(f"正在重置 {config.num_envs} 个环境 (每环境 {config.num_red + config.num_blue} 架 F-16)...", flush=True)
    t_reset = time.perf_counter()
    raw_obs_list = vec_env.reset(timeout=300.0)  # 5 分钟超时
    print(f"重置完成 ({time.perf_counter() - t_reset:.0f}s)", flush=True)
    print("=" * 70)

    # ---- 持久化：CSV 日志 ----
    csv_file = open("brmappo_training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Iteration", "Step", "ActorLoss", "CriticLoss",
                         "MaskLoss", "KL", "MaskEnt",
                         "RedMeanReward", "BlueMeanReward", "RedWinRate",
                         "RedMissiles", "BlueMissiles"])
    csv_file.flush()

    # ---- 持久化：checkpoint 目录 ----
    os.makedirs("checkpoints", exist_ok=True)
    print(f"模型存档: checkpoints/ (每 10 iter, 保留最新 5 个)")

    # ---- 评估准入准则 ----
    MIN_EPISODES_TO_EVAL = 50  # 最少完成 50 局后才允许覆盖 best 模型

    # ---- 6. 训练循环 ----
    total_steps = 0
    iteration = 1
    total_episodes = 0
    red_wins = 0
    blue_wins = 0
    draws = 0
    death_stats = {"red": Counter(), "blue": Counter()}
    best_win_rate = 0.0
    best_reward = -float("inf")

    # Episodic reward trackers — only fully-completed episodes contribute
    recent_ep_rewards_red = deque(maxlen=50)
    recent_ep_rewards_blue = deque(maxlen=50)
    # Per-component episodic trackers for red team diagnostics
    COMP_KEYS = ["r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv", "r_end"]
    recent_ep_comps_red: deque[dict] = deque(maxlen=50)

    while total_steps < config.total_env_steps:
        t_start = time.perf_counter()

        # 初始化缓冲区
        buffer = RolloutBuffer(
            num_steps=num_steps, num_envs=config.num_envs,
            num_red=config.num_red, action_dim=config.action_dim,
            rnn_hidden_size=config.rnn_hidden_size,
        )

        # 记录初始 RNN 状态
        buffer.rnn_actor_init = rnn_hidden_actor.copy()
        buffer.rnn_critic_init = rnn_hidden_critic.copy()

        current_ep_reward_red = np.zeros(config.num_envs, dtype=np.float32)
        current_ep_reward_blue = np.zeros(config.num_envs, dtype=np.float32)
        current_ep_comp_red = {k: np.zeros(config.num_envs, dtype=np.float64)
                               for k in COMP_KEYS}
        episode_red_missiles = np.zeros(config.num_envs, dtype=np.float32)
        episode_blue_missiles = np.zeros(config.num_envs, dtype=np.float32)

        # ------------------------------------------------------------------
        #  6a. 经验收集 (Rollout)
        # ------------------------------------------------------------------
        for step in range(num_steps):
            actions_list = []  # 每个 env 一个 action dict

            # Fetch engaged-targets sets from all envs (one pipe round-trip)
            engaged_sets = vec_env.env_method("refresh_engaged_targets")
            blue_own_positions_list = _fetch_blue_own_positions(vec_env)

            for env_idx in range(config.num_envs):
                env_obs = raw_obs_list[env_idx]  # dict[agent_id → obs_dict]
                env_actions = {}
                alive_mask = np.ones(config.num_red, dtype=bool)

                # ---- 蓝方：协同目标分配 + 引导律 (GCAS / 导弹规避在 env 层自动保护) ----
                blue_obs_dict = {bid: env_obs[bid] for bid in blue_ids}
                env_actions.update(blue_coordinated_actions(
                    blue_obs_dict, config.num_blue, config.num_red,
                    engaged_targets=engaged_sets[env_idx],
                    own_positions=blue_own_positions_list[env_idx]))

                # ---- 红方：Actor 采样 ----
                red_obs_tensors = []
                alive_red_indices = []
                for i, rid in enumerate(red_ids):
                    sim_key = rid  # env 用 aid 索引 plane
                    # 检查是否存活 (通过 obs 中的 death_mask 判断)
                    death_mask = env_obs[rid]["death_mask"]
                    # red_i 的 ego 在 death_mask 中位置是 1 + (num_red - 1) + i = num_red + i
                    # 实际上不需要这么复杂，直接看 ego_state 特征是否全零
                    ego_state = env_obs[rid]["ego_state"]
                    alive = not np.allclose(ego_state, 0.0)  # 零向量 = 已死
                    if alive:
                        red_obs_tensors.append(env_obs[rid])
                        alive_red_indices.append(i)
                        alive_mask[i] = True
                    else:
                        env_actions[rid] = np.zeros(config.action_dim, dtype=np.float32)
                        alive_mask[i] = False
                        # 存储占位数据 (死 agent 跳过)
                        buffer.store_step(step, env_idx, i, env_obs[rid],
                                          np.zeros(config.action_dim, dtype=np.float32),
                                          0.0, 0.0, 0.0, 1.0, alive=False)

                if alive_red_indices:
                    # 随机掩码数量 (论文 Algorithm 1: U(0, 2))
                    # 安全钳：采样上限 = min(2, 实体数)，且至少保留 1 个敌机
                    #         防止 Actor 完全失去态势感知导致梯度崩溃。
                    N_ally  = config.num_red - 1   # 红方视角的友机数 (不含自己)
                    N_enemy = config.num_blue       # 红方视角的敌机数
                    max_enemy_drop = min(2, max(0, N_enemy - 1))
                    max_ally_drop  = min(2, N_ally)
                    num_enemy_drop = np.random.randint(0, max_enemy_drop + 1)
                    num_ally_drop  = np.random.randint(0, max_ally_drop + 1)

                    # 批量前向
                    obs_t_list = [_obs_np_to_tensor(o, device) for o in red_obs_tensors]
                    batch_obs = _stack_obs(obs_t_list)

                    batch_rnn_a = torch.as_tensor(
                        rnn_hidden_actor[env_idx, alive_red_indices], device=device,
                    )
                    batch_rnn_c = torch.as_tensor(
                        rnn_hidden_critic[env_idx, alive_red_indices], device=device,
                    )

                    with torch.no_grad():
                        action_dist, new_rnn_a, mask_ent = actor(
                            batch_obs,
                            num_enemy_drop=num_enemy_drop,
                            num_ally_drop=num_ally_drop,
                            rnn_hidden=batch_rnn_a,
                        )
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action).sum(dim=-1)  # (N_alive,)

                        value, new_rnn_c = critic(batch_obs, batch_rnn_c)

                    # 拆分回各 agent
                    for k, i in enumerate(alive_red_indices):
                        rid = red_ids[i]
                        env_actions[rid] = action[k].cpu().numpy()
                        rnn_hidden_actor[env_idx, i] = new_rnn_a[k].cpu().numpy()
                        rnn_hidden_critic[env_idx, i] = new_rnn_c[k].cpu().numpy()

                        buffer.store_step(
                            step, env_idx, i,
                            obs_np=red_obs_tensors[k],
                            action=action[k].cpu().numpy(),
                            reward=0.0,  # 占位，step 后填充
                            value=value[k].item(),
                            log_prob=log_prob[k].item(),
                            done=0.0,
                            alive=True,
                        )

                    buffer.store_env_meta(step, env_idx,
                                          mask_ent.item(), num_enemy_drop, num_ally_drop)
                else:
                    buffer.store_env_meta(step, env_idx, 0.0, 0, 0)

                actions_list.append(env_actions)

            # ---- 环境步进 ----
            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(actions_list)

            # 更新观测、奖励、done
            for env_idx in range(config.num_envs):
                rew = rewards_list[env_idx]
                don = dones_list[env_idx]
                info = infos_list[env_idx]

                for i, rid in enumerate(red_ids):
                    # 填充最近一次 store_step 的 reward 和 done
                    if buffer.alive[step, env_idx, i]:
                        buffer.rewards[step, env_idx, i] = float(rew.get(rid, 0.0))
                        buffer.dones[step, env_idx, i] = float(don.get(rid, False))

                    # 若 agent 死亡，重置其 RNN 状态
                    if don.get(rid, False):
                        rnn_hidden_actor[env_idx, i] = np.zeros(
                            config.rnn_hidden_size, dtype=np.float32,
                        )
                        rnn_hidden_critic[env_idx, i] = np.zeros(
                            config.rnn_hidden_size, dtype=np.float32,
                        )

                # ---- accumulate step rewards FIRST (incl. terminal r_end for dead agents) ----
                for rid in red_ids:
                    current_ep_reward_red[env_idx] += rew.get(rid, 0.0)
                    # Accumulate per-component diagnostics
                    rcinfo = info.get(rid, {})
                    for k in COMP_KEYS:
                        current_ep_comp_red[k][env_idx] += rcinfo.get(k, 0.0)
                for bid in blue_ids:
                    current_ep_reward_blue[env_idx] += rew.get(bid, 0.0)

                # ---- episodic settlement AFTER accumulation (terminal r_end is included) ----
                # 检测整局结束 (所有 agent 同时终止)
                if all(don.values()):
                    total_episodes += 1
                    blue_alive = sum(
                        1 for bid in blue_ids
                        if info.get(bid, {}).get("alive", False))
                    red_alive = sum(
                        1 for rid in red_ids
                        if info.get(rid, {}).get("alive", False))
                    if blue_alive == 0 and red_alive > 0:
                        red_wins += 1
                    elif red_alive == 0 and blue_alive > 0:
                        blue_wins += 1
                    else:
                        draws += 1
                    # 累积死亡原因
                    for bid in blue_ids:
                        dr = info.get(bid, {}).get("death_reason")
                        if dr:
                            death_stats["blue"][dr] += 1
                    for rid in red_ids:
                        dr = info.get(rid, {}).get("death_reason")
                        if dr:
                            death_stats["red"][dr] += 1

                    recent_ep_rewards_red.append(float(current_ep_reward_red[env_idx]))
                    recent_ep_rewards_blue.append(float(current_ep_reward_blue[env_idx]))
                    # Persist component breakdown
                    recent_ep_comps_red.append(
                        {k: float(current_ep_comp_red[k][env_idx]) for k in COMP_KEYS})
                    current_ep_reward_red[env_idx] = 0.0
                    current_ep_reward_blue[env_idx] = 0.0
                    for k in COMP_KEYS:
                        current_ep_comp_red[k][env_idx] = 0.0

                # Track per-episode missile launches
                for rid in red_ids:
                    episode_red_missiles[env_idx] += info.get(rid, {}).get("missiles_fired_this_step", 0)
                for bid in blue_ids:
                    episode_blue_missiles[env_idx] += info.get(bid, {}).get("missiles_fired_this_step", 0)

            raw_obs_list = next_obs_list
            total_steps += config.num_envs

        # 保存 rollout 结束后的 RNN 状态 (用于 GAE bootstrap)
        buffer.rnn_actor_final = rnn_hidden_actor.copy()
        buffer.rnn_critic_final = rnn_hidden_critic.copy()

        # ------------------------------------------------------------------
        #  6b. PPO 更新
        # ------------------------------------------------------------------
        stats = ppo_update(
            config, actor, critic, actor_opt, critic_opt, mask_opt, buffer, device,
        )

        t_elapsed = time.perf_counter() - t_start

        # ---- 统计汇总 ----
        avg_r_red = np.mean(recent_ep_rewards_red) if recent_ep_rewards_red else 0.0
        avg_r_blue = np.mean(recent_ep_rewards_blue) if recent_ep_rewards_blue else 0.0
        avg_m_red = np.mean(episode_red_missiles)
        avg_m_blue = np.mean(episode_blue_missiles)
        red_win_rate = red_wins / max(total_episodes, 1)

        # Average per-component breakdown across completed episodes
        if recent_ep_comps_red:
            avg_comps = {k: float(np.mean([ep[k] for ep in recent_ep_comps_red]))
                         for k in COMP_KEYS}
        else:
            avg_comps = {k: 0.0 for k in COMP_KEYS}

        # Build breakdown string: [Alt:+12.3 Pitch:-0.5 Roll:0.0 Vel:-0.3 Adv:+0.0 End:-180.0]
        comp_str = " ".join(
            f"{k.replace('r_','').capitalize()}:{avg_comps[k]:+.1f}"
            for k in COMP_KEYS)

        # ---- 持久化：CSV 写入 + flush ----
        csv_writer.writerow([iteration, total_steps,
                             f"{stats.get('actor_loss', 0):.6f}",
                             f"{stats.get('critic_loss', 0):.6f}",
                             f"{stats.get('mask_loss', 0):.6f}",
                             f"{stats.get('kl', 0):.6f}",
                             f"{stats.get('mask_ent', 0):.6f}",
                             f"{avg_r_red:.4f}",
                             f"{avg_r_blue:.4f}",
                             f"{red_win_rate:.6f}",
                             f"{avg_m_red:.1f}",
                             f"{avg_m_blue:.1f}"])
        csv_file.flush()

        def _fmt_death(counter: Counter) -> str:
            order = ["Missile_Kill", "Crash_LowAlt", "Crash_HighAlt",
                     "Crash_OverG", "Crash_Extreme"]
            parts = []
            for k in order:
                parts.append(f"{k.replace('Crash_','').replace('Missile_','')}:{counter.get(k, 0)}")
            for k in sorted(counter.keys()):
                if k not in order:
                    parts.append(f"{k}:{counter[k]}")
            return ", ".join(parts)

        print(f"Iter {iteration:5d} | "
              f"total_steps={total_steps:9d} | "
              f"t={t_elapsed:5.1f}s | "
              f"R_red={avg_r_red:+8.1f} [{comp_str}] "
              f"R_blue={avg_r_blue:+8.1f} | "
              f"M_red={avg_m_red:.0f} M_blue={avg_m_blue:.0f} | "
              f"ActorLoss={stats.get('actor_loss', 0):+.4f} "
              f"CriticLoss={stats.get('critic_loss', 0):+.4f} "
              f"MaskLoss={stats.get('mask_loss', 0):+.4f} | "
              f"KL={stats.get('kl', 0):.4f} "
              f"MaskEnt={stats.get('mask_ent', 0):.4f} | "
              f"WinRate_red={red_win_rate:.3f} "
              f"(Ep={total_episodes} W={red_wins}/{blue_wins}/{draws}) | "
              f"Deaths: Red[{_fmt_death(death_stats['red'])}] "
              f"Blue[{_fmt_death(death_stats['blue'])}]")

        # ---- 持久化：高频轮转存档 (每 10 iter, 保留最新 5 个) ----
        if iteration % 10 == 0:
            actor_path = f"checkpoints/brmappo_actor_latest_{iteration:06d}.pt"
            critic_path = f"checkpoints/brmappo_critic_latest_{iteration:06d}.pt"
            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)
            _cleanup_rotating_checkpoints("checkpoints", "brmappo_actor_latest", keep=5)
            _cleanup_rotating_checkpoints("checkpoints", "brmappo_critic_latest", keep=5)

        # ---- 持久化：最佳模型拦截 (需满足评估准入准则) ----
        if total_episodes >= MIN_EPISODES_TO_EVAL:
            is_better = (red_win_rate > best_win_rate) or (
                abs(red_win_rate - best_win_rate) < 1e-9 and avg_r_red > best_reward)
            if is_better:
                best_win_rate = red_win_rate
                best_reward = avg_r_red
                torch.save(actor.state_dict(),
                           "checkpoints/brmappo_actor_best.pt")
                torch.save(critic.state_dict(),
                           "checkpoints/brmappo_critic_best.pt")
                print(f"  *** New Best Model Saved! "
                      f"(WinRate={best_win_rate:.4f}, Reward={best_reward:+.2f}) ***")

        iteration += 1

    # ---- 7. 持久化：最终模型存档 ----
    torch.save(actor.state_dict(), "checkpoints/brmappo_actor_final.pt")
    torch.save(critic.state_dict(), "checkpoints/brmappo_critic_final.pt")
    print("=" * 70)
    print(f"最终模型已保存至 checkpoints/brmappo_actor_final.pt / "
          f"checkpoints/brmappo_critic_final.pt")
    print(f"总 Episodes: {total_episodes}  "
          f"红方胜: {red_wins}  蓝方胜: {blue_wins}  平局: {draws}  "
          f"红方胜率: {red_wins / max(total_episodes, 1):.4f}")
    csv_file.close()

    # ---- 8. 清理 ----
    vec_env.close()
    print("训练完成！")


if __name__ == "__main__":
    mp.freeze_support()
    main()
