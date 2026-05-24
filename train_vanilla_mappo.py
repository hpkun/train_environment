"""
train_vanilla_mappo.py —— 纯 MLP MAPPO 基线训练脚本

剥离了论文的 EntityObservationEncoder 和 MaskVectorGenerator，
使用展平观测 → GRU → MLP 的最简架构，仅保留 PPO Clip + MSE + Entropy。

用途：验证 my_uav_env 环境连通性 (物理 / 奖励 / 开火 / 终止)。
用法：python train_vanilla_mappo.py
"""
from __future__ import annotations

import csv
import os

# ---- 多进程性能：禁止底层库的线程池竞争 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Allow Intel + LLVM OpenMP runtimes to coexist (JSBSim vs numpy/torch).
# Without this, ``FGFDMExec()`` aborts the process with OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter, deque
import multiprocessing as mp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass

# NB: UavCombatEnv is imported inside _worker() lazily.
# JSBSim C++ banners are *not* suppressed — suppress_jsbsim_output defaults to False.
# Tracer files (_jsbsim_tracer_*.txt, _worker_reset_tracer_*.txt) are left behind
# if a worker crashes during JSBSim construction, aiding diagnosis.

# ==============================================================================
#  配置 (2v2 快速验证)
# ==============================================================================
# class Config:
#     # ---- 环境 (对标论文 6v6) ----
#     num_envs: int = 8           # Reduced from 32 to avoid JSBSim C++ resource exhaustion
#     num_red: int = 6            # 6v6 训练场景
#     num_blue: int = 6
#     max_episode_length: int = 1400  # 论文一致
#     action_dim: int = 3

#     # ---- PPO (适配大规模数据) ----
#     replay_buffer_size: int = 500   # 500 steps / 8 envs = 62.5 rollout steps
#     n_update_epochs: int = 10       # 保持不变
#     n_minibatches: int = 4          # 保持不变
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     clip_epsilon: float = 0.2
#     max_grad_norm: float = 5.0      # 硬件防爆器，保持不变

#     # ---- 学习率 (对标论文求稳) ----
#     actor_lr: float = 0.0002        # 论文原版 Actor learning rate
#     critic_lr: float = 0.0005       # 论文原版 Critic learning rate

#     # ---- 损失系数 (高探索度) ----
#     entropy_coef: float = 0.05      # 论文原版 Entropy loss coefficient

#     # ---- 网络 ----
#     mlp_hidden: int = 128           # 论文一致
#     rnn_hidden_size: int = 128      # 论文一致

#     # ---- 训练总量 ----
#     total_env_steps: int = 10_000_000  # 1000 万步 (1e7)
class Config:
    # ---- 环境 (从局部冲突开始，降低协同难度) ----
    num_envs: int = 8           
    num_red: int = 2            # 💡 强烈建议先从 2v2 或 1v1 开始训练！
    num_blue: int = 2           # 等模型在 2v2 收敛后，再将权重加载到 6v6 中微调
    max_episode_length: int = 1400  
    max_episode_length: int = 1400
    enable_blue_gcas: bool = False
    resume_from_best: bool = False
    action_dim: int = 3

    # ---- PPO (论文 Table 3) ----
    replay_buffer_size: int = 2000  
    n_update_epochs: int = 10       
    n_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 5.0      

    # ---- 学习率 (论文 Table 3) ----
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4

    # ---- 损失系数 (论文 Table 3: constant 0.05, no decay) ----
    entropy_coef: float = 0.05      

    # ---- 网络 (论文 Table 3: [128, 128]) ----
    mlp_hidden: int = 128
    rnn_hidden_size: int = 128      

    # ---- 训练总量 ----
    total_env_steps: int = 10_000_000


# ==============================================================================
#  纯 MLP Actor / Critic (展平观测 → GRU → MLP)
# ==============================================================================

def _compute_obs_dim(num_red: int, num_blue: int, is_red: bool) -> int:
    """计算展平后的观测向量维度。"""
    if is_red:
        n_ally = num_red - 1
        n_enemy = num_blue
    else:
        n_ally = num_blue - 1
        n_enemy = num_red
    total_entities = 1 + max(n_ally, 0) + n_enemy
    return 11 * total_entities + (num_red + num_blue) + 1 + 1 + 3  # entities + death_mask + missile_warning + altitude + velocity


class VanillaActor(nn.Module):
    """纯 MLP 策略网络：展平 obs → GRU → MLP → Gaussian(mu, learnable sigma)."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden: int = 128, rnn_hidden: int = 128):
        super().__init__()
        self.fc_in = nn.Linear(obs_dim, hidden)
        self.rnn = nn.GRUCell(hidden, rnn_hidden)
        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -1.204))  # ln(0.3)

    def forward(self, obs_flat: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        Args:
            obs_flat:   (B, obs_dim)  展平观测
            rnn_hidden: (B, rnn_hidden)  GRU 隐藏状态
        Returns:
            action_dist: Normal
            rnn_hidden:  (B, rnn_hidden)
        """
        x = F.relu(self.fc_in(obs_flat))          # (B, hidden)
        rnn_hidden_new = self.rnn(x, rnn_hidden)  # (B, rnn_hidden)
        mu = self.action_head(rnn_hidden_new)      # (B, action_dim) in [-1, 1]
        # Guard: if weights were corrupted by a previous NaN backward pass,
        # replace NaN mu with zeros so Normal() doesn't raise.
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu = mu.clamp(-0.999, 0.999)  # keep mu strictly inside [-1, 1]
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mu)
        return torch.distributions.Normal(mu, sigma), rnn_hidden_new


class CentralizedCritic(nn.Module):
    """MAPPO centralized critic: global state (all red agents' obs concat) → V(s).

    Paper §3.4: the critic sees the joint global state so it can learn a
    value function with full battlefield awareness.  No RNN — the global
    state already contains all information needed for credit assignment.
    """

    def __init__(self, global_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor):
        """global_obs: (B, global_obs_dim) → value: (B, 1)"""
        return self.net(global_obs)


# ==============================================================================
#  观测展平工具
# ==============================================================================
def _flatten_obs(obs_np: dict) -> np.ndarray:
    """将 Dict 观测展平为一维向量。"""
    return np.concatenate([
        obs_np["ego_state"].ravel(),
        obs_np["ally_states"].ravel(),
        obs_np["enemy_states"].ravel(),
        obs_np["death_mask"].astype(np.float32).ravel(),
        obs_np["missile_warning"].ravel(),
        (obs_np["altitude"].ravel() / 10000.0).astype(np.float32),
        (obs_np["velocity"].ravel() / 600.0).astype(np.float32),
    ])


def _cleanup_rotating_checkpoints(directory: str, prefix: str, keep: int = 5):
    """删除超出保留数量的旧轮转 checkpoint 文件。

    Args:
        directory: checkpoint 所在目录
        prefix:    文件名前缀 (e.g. "vanilla_actor_latest")
        keep:      保留最新的文件数量 (默认 5)
    """
    import glob as _glob
    pattern = os.path.join(directory, f"{prefix}_*.pt")
    files = sorted(_glob.glob(pattern))
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass


# ==============================================================================
#  蓝方自动驾驶仪 —— 高度预算控制器 (Altitude Budget Controller)
#
#  核心思想: 高度是消耗性资源，机动侵略性必须随高度降低而收敛。
#  高空 = 富有 → 可以大坡度追击。  低空 = 贫穷 → 只求自保，禁止一切战斗。
#
#  动作通道映射 (env.py _parse_actions 中的定义):
#    action[0] = pitch_cmd    → target_pitch        (正=抬头, PID→elevator)
#    action[1] = heading_cmd  → target_heading_delta (正=右转, PID→aileron)
#    action[2] = vel_cmd      → target_velocity      (正=加速, PID→throttle)
# ==============================================================================

from rule_based_agent import blue_coordinated_actions


# ==============================================================================
#  SubprocVecEnv (从 train_ppo.py 精简，保留超时保护)
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
    # Allow Intel + LLVM OpenMP runtimes to coexist (JSBSim vs numpy/torch)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Note: we do NOT apply permanent CRT/Win32 stdout suppression here.
    # The SuppressOutput context manager inside AircraftSimulator.reload()
    # handles JSBSim C++ banner suppression *temporarily* during FGFDMExec
    # construction, with proper save/restore of fds, handles, and Python
    # stdout objects.  Permanent OS-level suppression was causing worker
    # crashes on some Windows configurations.

    # ---- Step-by-step init with per-stage crash diagnostics ----
    env = None
    try:
        from my_uav_env import UavCombatEnv
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_import_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during import:\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    try:
        env = UavCombatEnv(**env_kwargs)
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_env_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during UavCombatEnv():\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    try:
        remote.send(("ready", os.getpid()))
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_ready_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during ready signal:\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, BrokenPipeError, OSError):
            break
        try:
            if cmd == "step":
                obs, rewards, terminated, truncated, info = env.step(data)
                dones = {}
                for aid in env.agent_ids:
                    dones[aid] = bool(terminated.get(aid, False) or truncated.get(aid, False))
                if all(dones.values()):
                    obs, _ = env.reset()
                    import gc
                    gc.collect()
                remote.send((obs, rewards, dones, info))
            elif cmd == "reset":
                obs, info = env.reset()
                import gc
                gc.collect()
                remote.send(obs)
            elif cmd == "call":
                method_name, args, kwargs = data
                result = getattr(env, method_name)(*args, **kwargs)
                remote.send(result)
            elif cmd == "close":
                remote.close()
                break
        except Exception:
            import traceback as _tb
            with open(f"_worker_crash_{os.getpid()}.txt", "w") as _f:
                _f.write(f"Worker PID {os.getpid()} crashed on cmd={cmd}:\n")
                _tb.print_exc(file=_f)
            try:
                remote.send(("error", _tb.format_exc()))
            except Exception:
                pass
            break


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, startup_delay: float = 0.5,
                 ready_timeout: float = 600.0):
        self.n_envs = num_envs
        self._dead_workers: set[int] = set()
        self._env_kwargs = env_kwargs  # stored for worker restart
        ctx = mp.get_context("spawn")
        remotes_tup, work_remotes_tup = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.remotes = list(remotes_tup)
        self.work_remotes = list(work_remotes_tup)
        self.processes = []
        for i in range(num_envs):
            p = ctx.Process(target=_worker,
                            args=(self.work_remotes[i], self.remotes[i], env_kwargs),
                            daemon=True)
            p.start()
            self.processes.append(p)
            self.work_remotes[i].close()
            if i < num_envs - 1:
                time.sleep(startup_delay)

        # Wait for every worker to signal readiness (env constructed, import done)
        ready_count = 0
        for i, remote in enumerate(self.remotes):
            try:
                if not remote.poll(ready_timeout):
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not become ready within {ready_timeout:.0f}s")
                msg = remote.recv()
            except (BrokenPipeError, OSError, EOFError):
                raise RuntimeError(
                    f"Worker {i} (PID {self.processes[i].pid}) "
                    f"pipe broken during startup — OS may have killed the process") from None
            if isinstance(msg, tuple) and msg[0] == "ready":
                ready_count += 1
            else:
                raise RuntimeError(
                    f"Worker {i} sent unexpected init message: {str(msg)[:200]}")
        print(f"  All {ready_count} workers ready", flush=True)

    def reset(self, timeout: float = 300.0,
              serial: bool = True) -> list[dict]:
        """Reset all workers.

        When ``serial=True`` (default), each worker is reset one at a time —
        send reset, wait for response, then move to the next worker.  This
        prevents concurrent JSBSim construction (96 FGFDMExec instances) that
        triggers OS-level process kills on Windows.  Subsequent resets that use
        ``reload()`` (fast) can pass ``serial=False``.
        """
        results = [None] * len(self.remotes)
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results[i] = {}
                continue
            # Send reset command
            try:
                remote.send(("reset", None))
            except (BrokenPipeError, OSError):
                self._dead_workers.add(i)
                results[i] = {}
                continue

            if serial:
                # Wait for this worker's response before talking to the next one
                try:
                    ready = remote.poll(timeout)
                except (BrokenPipeError, OSError, EOFError):
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died before reset response", flush=True)
                    self._dead_workers.add(i)
                    results[i] = {}
                    continue
                if ready:
                    try:
                        msg = remote.recv()
                    except EOFError:
                        print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                              f"died during reset", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                        continue
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error during reset: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                    else:
                        results[i] = msg
                else:
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not respond within {timeout:.0f}s")

        # If not serial, collect responses after all sends (original behaviour)
        if not serial:
            for i, remote in enumerate(self.remotes):
                if results[i] is not None:
                    continue  # already handled (dead worker)
                try:
                    ready = remote.poll(timeout)
                except (BrokenPipeError, OSError, EOFError):
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died before reset response", flush=True)
                    self._dead_workers.add(i)
                    results[i] = {}
                    continue
                if ready:
                    try:
                        msg = remote.recv()
                    except EOFError:
                        print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                              f"died during reset", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                        continue
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error during reset: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                    else:
                        results[i] = msg
                else:
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not respond within {timeout:.0f}s")

        # Auto-restart dead workers after reset (retry up to 3 times)
        for i in list(self._dead_workers):
            # Clean up any crash tracer files from the dead worker
            _dead_pid = self.processes[i].pid
            for _pattern in ("_worker_crash_import_", "_worker_crash_env_",
                             "_worker_crash_ready_", "_worker_crash_",
                             "_worker_reset_tracer_", "_jsbsim_tracer_"):
                _tracer_path = f"{_pattern}{_dead_pid}.txt"
                if os.path.exists(_tracer_path):
                    try:
                        os.remove(_tracer_path)
                    except OSError:
                        pass
            restarted = False
            for attempt in range(3):
                try:
                    new_obs = self._restart_worker(i, self._env_kwargs)
                    results[i] = new_obs
                    print(f"  [INFO] Worker {i} restarted after reset (attempt {attempt+1})", flush=True)
                    restarted = True
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  [WARN] Worker {i} restart attempt {attempt+1} failed: {e}", flush=True)
                        time.sleep(5.0)
                    else:
                        print(f"  [ERROR] Worker {i} restart failed after 3 attempts: {e}", flush=True)
            if not restarted:
                raise RuntimeError(
                    f"Worker {i} could not be restarted after 3 attempts. "
                    f"Training cannot continue with a dead environment.")
        return results

    def step(self, actions_list: list[dict], timeout: float = 60.0) -> tuple:
        for i, (remote, actions) in enumerate(zip(self.remotes, actions_list)):
            if i not in self._dead_workers:
                try:
                    remote.send(("step", actions))
                except (BrokenPipeError, OSError):
                    self._dead_workers.add(i)
        results = []
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results.append(({}, {}, {}, {}))
                continue
            try:
                ready = remote.poll(timeout)
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died before step response", flush=True)
                self._dead_workers.add(i)
                results.append(({}, {}, {}, {}))
                continue
            if ready:
                try:
                    msg = remote.recv()
                except EOFError:
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died (EOFError)", flush=True)
                    self._dead_workers.add(i)
                    results.append(({}, {}, {}, {}))
                    continue
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                    print(f"  [WARN] Worker {i} error: {msg[1][:200]}", flush=True)
                    self._dead_workers.add(i)
                    results.append(({}, {}, {}, {}))
                else:
                    results.append(msg)
            else:
                raise TimeoutError(
                    f"Worker {i} (PID {self.processes[i].pid}) "
                    f"did not respond within {timeout:.0f}s")

        # Auto-restart dead workers and fix up dummy results
        for i in list(self._dead_workers):
            try:
                new_obs = self._restart_worker(i, self._env_kwargs)
                results[i] = (new_obs, {}, {aid: True for aid in new_obs}, {})
            except Exception as e:
                print(f"  [WARN] Failed to restart worker {i}: {e}", flush=True)

        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def env_method(self, method_name: str, *args, timeout: float = 30.0, **kwargs):
        """Call a method on every remote env and return the list of results."""
        results = []
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results.append(set())
                continue
            try:
                remote.send(("call", (method_name, args, kwargs)))
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died during env_method send", flush=True)
                self._dead_workers.add(i)
                results.append(set())
                continue
            try:
                if remote.poll(timeout):
                    msg = remote.recv()
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error in {method_name}: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results.append(set())
                    else:
                        results.append(msg)
                else:
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"timed out during env_method {method_name}", flush=True)
                    self._dead_workers.add(i)
                    results.append(set())
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died during env_method recv", flush=True)
                self._dead_workers.add(i)
                results.append(set())
        return results

    def _restart_worker(self, i: int, env_kwargs: dict):
        """Restart a dead worker and return its initial observation."""
        # Clean up old process
        old_p = self.processes[i]
        if old_p.is_alive():
            old_p.terminate()
            old_p.join(timeout=5)
        # Close old remote
        try:
            self.remotes[i].close()
        except Exception:
            pass
        # Create new pipe and process
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_worker,
                        args=(child_conn, parent_conn, env_kwargs),
                        daemon=True)
        p.start()
        child_conn.close()
        self.processes[i] = p
        self.remotes[i] = parent_conn
        # Wait for ready signal (with protection against immediate worker death)
        try:
            if not parent_conn.poll(600.0):
                raise RuntimeError(f"Worker {i} ready signal timed out")
            msg = parent_conn.recv()
            if not (isinstance(msg, tuple) and msg[0] == "ready"):
                raise RuntimeError(f"Worker {i} unexpected ready msg: {str(msg)[:200]}")
            # Send reset command
            parent_conn.send(("reset", None))
            if not parent_conn.poll(300.0):
                raise RuntimeError(f"Worker {i} reset timed out")
            obs = parent_conn.recv()
            if isinstance(obs, tuple) and len(obs) == 2 and obs[0] == "error":
                raise RuntimeError(f"Worker {i} error during reset: {obs[1][:200]}")
            self._dead_workers.discard(i)
            print(f"  [INFO] Worker {i} restarted (PID {p.pid})", flush=True)
            return obs
        except (BrokenPipeError, OSError, EOFError) as e:
            raise RuntimeError(f"Worker {i} pipe broken during restart: {e}") from e

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ==============================================================================
#  Rollout Buffer (无掩码字段，精简版)
# ==============================================================================
class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, num_red: int,
                 action_dim: int, rnn_hidden_size: int):
        self.num_steps = num_steps
        T, E, A = num_steps, num_envs, num_red
        H = rnn_hidden_size

        # 展平观测存储 (可变长度 obs_dim 由实际数据决定)
        self.obs: list[list[list[np.ndarray]]] = [
            [[None for _ in range(A)] for _ in range(E)] for _ in range(T)]
        self.actions = np.zeros((T, E, A, action_dim), dtype=np.float32)
        self.rewards = np.zeros((T, E, A), dtype=np.float32)
        self.values = np.zeros((T, E, A), dtype=np.float32)
        self.log_probs = np.zeros((T, E, A), dtype=np.float32)
        self.dones = np.zeros((T, E, A), dtype=np.float32)
        self.alive = np.zeros((T, E, A), dtype=bool)

        # Actor RNN states only (centralized critic has no RNN)
        self.rnn_actor_init = np.zeros((E, A, H), dtype=np.float32)
        self.rnn_actor_final = np.zeros((E, A, H), dtype=np.float32)

        # GAE bootstrap values: V(s_T) shared by all agents (centralized critic)
        self.bootstrap_values = np.zeros((E, A), dtype=np.float32)

    def store_step(self, step: int, env_idx: int, agent_idx: int,
                   obs_np: np.ndarray, action: np.ndarray,
                   reward: float, value: float, log_prob: float,
                   done: float, alive: bool):
        self.obs[step][env_idx][agent_idx] = obs_np
        self.actions[step, env_idx, agent_idx] = action
        self.rewards[step, env_idx, agent_idx] = reward
        self.values[step, env_idx, agent_idx] = value
        self.log_probs[step, env_idx, agent_idx] = log_prob
        self.dones[step, env_idx, agent_idx] = done
        self.alive[step, env_idx, agent_idx] = alive


# ==============================================================================
#  GAE
# ==============================================================================
def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float, lam: float):
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


# ==============================================================================
#  PPO Update (无掩码)
# ==============================================================================
def _grad_has_nan(module: nn.Module) -> bool:
    """Return True if any parameter gradient contains NaN or Inf."""
    for p in module.parameters():
        if p.grad is not None:
            if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                return True
    return False


def _current_entropy_coef(config, _total_steps: int = 0) -> float:
    """Constant entropy coefficient (paper Table 3)."""
    return config.entropy_coef


def _safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0)


def _classify_death_reason(reason: str | None) -> str:
    if not reason:
        return "other"
    r = str(reason).lower()
    if "shot" in r or "missile" in r or "hit" in r:
        return "missile"
    if "crash" in r or "ground" in r or "altitude" in r:
        return "crash"
    return "other"


def _episode_outcome(red_alive: int, blue_alive: int) -> str:
    if blue_alive == 0 and red_alive > 0:
        return "red"
    if red_alive == 0 and blue_alive > 0:
        return "blue"
    return "draw"


def _ppo_update_legacy(actor, critic, actor_opt, critic_opt, buffer, config, device,
                       total_steps: int = 0):
    """MAPPO CTDE update: centralized critic sees global state (all red obs concat).

    The critic outputs a single V(s_global) shared by all agents.  GAE uses
    per-agent rewards + the shared value to compute per-agent advantages.
    The critic loss averages MSE across all alive agent-timesteps.
    """
    num_steps = buffer.num_steps
    num_envs = buffer.rnn_actor_init.shape[0]
    num_red = buffer.rnn_actor_init.shape[1]

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_agents = 0

    actor_opt.zero_grad()
    critic_opt.zero_grad()

    for env_idx in range(num_envs):
        # ---- Build global obs per timestep (all red agents' obs concat) ----
        global_obs_seq = []
        for t in range(num_steps):
            parts = [buffer.obs[t][env_idx][i] for i in range(num_red)]
            global_obs_seq.append(np.concatenate(parts))

        # ---- Centralized critic: batch forward all timesteps → V(s_global_t) ----
        gobs_batch = torch.as_tensor(np.stack(global_obs_seq), dtype=torch.float32,
                                      device=device)  # (T, global_obs_dim)
        new_vals_global = critic(gobs_batch).squeeze(-1)  # (T,)

        env_critic_loss = 0.0
        env_critic_count = 0

        for agent_idx in range(num_red):
            # ---- Collect trajectory (alive steps only) ----
            t_obs_flat = []
            t_act = []
            t_rew = []
            t_val = []
            t_lp = []
            t_done = []
            alive_steps = []
            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_obs_flat.append(buffer.obs[step][env_idx][agent_idx])
                    t_act.append(buffer.actions[step, env_idx, agent_idx])
                    t_rew.append(buffer.rewards[step, env_idx, agent_idx])
                    t_val.append(buffer.values[step, env_idx, agent_idx])
                    t_lp.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_done.append(buffer.dones[step, env_idx, agent_idx])
                    alive_steps.append(step)

            if len(t_act) == 0:
                continue

            T = len(t_act)

            # ---- GAE (uses centralized values stored during rollout) ----
            bootstrap = float(buffer.bootstrap_values[env_idx, agent_idx])
            rewards = torch.tensor(t_rew, device=device)
            old_values = torch.tensor(t_val + [bootstrap], device=device)
            dones = torch.tensor(t_done, device=device)

            advantages, returns = compute_gae(rewards, old_values, dones,
                                              config.gamma, config.gae_lambda)

            if advantages.numel() > 1:
                adv_std = advantages.std()
            else:
                adv_std = torch.std(advantages, correction=0)
            if adv_std <= 1e-8 or torch.isnan(adv_std):
                adv_std = 1.0
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            # ---- Critic loss: MSE(V(s_global_t), return_i(t)) per alive step ----
            for t_idx, step in enumerate(alive_steps):
                env_critic_loss += F.mse_loss(
                    new_vals_global[step], returns[t_idx].detach(),
                    reduction='none')
                env_critic_count += 1

            # ---- Actor GRU unroll (per-agent, from rollout init state) ----
            rnn_a = torch.as_tensor(
                buffer.rnn_actor_init[env_idx, agent_idx], device=device).unsqueeze(0)

            new_lps = []
            entropies = []

            for t in range(T):
                obs_t = torch.as_tensor(t_obs_flat[t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                act_t = torch.as_tensor(t_act[t], dtype=torch.float32, device=device)

                action_dist, rnn_a = actor(obs_t, rnn_a)

                new_lp = action_dist.log_prob(act_t.unsqueeze(0)).sum(dim=-1)
                new_lps.append(new_lp)
                entropies.append(action_dist.entropy().mean())

            new_lp = torch.cat(new_lps)
            old_lp = torch.tensor(t_lp, device=device)

            ent_avg = torch.stack(entropies).mean()

            # ---- PPO Clip Loss ----
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                1 + config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_coef = _current_entropy_coef(config, total_steps)
            actor_loss = policy_loss - entropy_coef * ent_avg

            actor_loss.backward()

            total_actor_loss += actor_loss.item()
            total_entropy += ent_avg.item()
            n_agents += 1

        # ---- Centralized critic loss (per-env, averaged across agents) ----
        if env_critic_count > 0:
            env_critic_loss = env_critic_loss / env_critic_count
            env_critic_loss.backward()
            total_critic_loss += env_critic_loss.item()

    if n_agents == 0:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # ---- NaN gradient guard ----
    actor_nan = _grad_has_nan(actor)
    critic_nan = _grad_has_nan(critic)
    if actor_nan or critic_nan:
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        print(f"  [WARN] NaN gradient detected (actor={actor_nan}, critic={critic_nan}) — "
              f"skipping optimizer step to preserve weights", flush=True)
        return {"actor_loss": float("nan"), "critic_loss": float("nan"), "entropy": 0.0}

    nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
    actor_opt.step()
    critic_opt.step()

    return {
        "actor_loss": total_actor_loss / n_agents,
        "critic_loss": total_critic_loss / max(num_envs, 1),
        "entropy": total_entropy / n_agents,
    }


# ==============================================================================
#  主训练循环
# ==============================================================================
def ppo_update(actor, critic, actor_opt, critic_opt, buffer, config, device,
               total_steps: int = 0):
    """Trajectory-minibatch MAPPO update used by the training loop."""
    num_steps = buffer.num_steps
    num_envs = buffer.rnn_actor_init.shape[0]
    num_red = buffer.rnn_actor_init.shape[1]

    global_obs_by_env = []
    trajectories = []

    for env_idx in range(num_envs):
        global_obs_seq = []
        for t in range(num_steps):
            parts = [buffer.obs[t][env_idx][i] for i in range(num_red)]
            global_obs_seq.append(np.concatenate(parts))
        global_obs_by_env.append(np.stack(global_obs_seq).astype(np.float32))

        for agent_idx in range(num_red):
            t_obs_flat = []
            t_act = []
            t_rew = []
            t_val = []
            t_lp = []
            t_done = []
            alive_steps = []
            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_obs_flat.append(buffer.obs[step][env_idx][agent_idx])
                    t_act.append(buffer.actions[step, env_idx, agent_idx])
                    t_rew.append(buffer.rewards[step, env_idx, agent_idx])
                    t_val.append(buffer.values[step, env_idx, agent_idx])
                    t_lp.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_done.append(buffer.dones[step, env_idx, agent_idx])
                    alive_steps.append(step)

            if not t_act:
                continue

            bootstrap = float(buffer.bootstrap_values[env_idx, agent_idx])
            rewards = torch.tensor(t_rew, device=device)
            old_values = torch.tensor(t_val + [bootstrap], device=device)
            dones = torch.tensor(t_done, device=device)
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
                "obs": np.stack(t_obs_flat).astype(np.float32),
                "actions": np.stack(t_act).astype(np.float32),
                "old_log_probs": np.asarray(t_lp, dtype=np.float32),
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
                obs = torch.as_tensor(traj["obs"], dtype=torch.float32, device=device)
                acts = torch.as_tensor(traj["actions"], dtype=torch.float32,
                                       device=device)
                old_lp = torch.as_tensor(traj["old_log_probs"],
                                         dtype=torch.float32, device=device)
                adv = traj["advantages"].to(device)
                ret = traj["returns"].to(device)

                new_lps = []
                traj_entropies = []
                for t in range(obs.shape[0]):
                    action_dist, rnn_a = actor(obs[t].unsqueeze(0), rnn_a)
                    new_lps.append(
                        action_dist.log_prob(acts[t].unsqueeze(0)).sum(dim=-1))
                    traj_entropies.append(action_dist.entropy().mean())

                new_lp = torch.cat(new_lps)
                ent_avg = torch.stack(traj_entropies).mean()
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(policy_loss - entropy_coef * ent_avg)
                entropies.append(ent_avg.detach())

                gobs = torch.as_tensor(
                    global_obs_by_env[env_idx][traj["alive_steps"]],
                    dtype=torch.float32, device=device)
                values = critic(gobs).squeeze(-1)
                critic_losses.append(F.mse_loss(values, ret))

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

            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
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


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 计算展平观测维度 (红方视角)
    obs_dim = _compute_obs_dim(config.num_red, config.num_blue, is_red=True)

    # ---- 持久化：创建 checkpoint 目录 ----
    os.makedirs("checkpoints", exist_ok=True)

    # ---- 持久化：CSV 日志 ----
    csv_file = open("vanilla_training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Iteration", "Step", "ActorLoss", "CriticLoss",
                         "Entropy", "RedMeanReward", "RedWinRate",
                         "RedRewardStd", "WinRateRecent",
                         "RedMissiles", "BlueMissiles",
                         "Episodes", "RedWins", "BlueWins", "Draws",
                         "RedAliveMean", "BlueAliveMean",
                         "RedDeathsMissile", "RedDeathsCrash",
                         "BlueDeathsMissile", "BlueDeathsCrash",
                         "RedMissileHits", "BlueMissileHits",
                         "RedMissileHitRate", "BlueMissileHitRate",
                         "KD_Red", "RWR"])
    csv_file.flush()

    print(f"设备: {device}")
    print(f"架构: Vanilla MLP + GRU (无注意力, 无掩码)")
    print(f"场景: {config.num_red}v{config.num_blue} (红方 RL, 蓝方规则)")
    print(f"展平 obs 维度: {obs_dim}")
    print(f"buffer: {config.replay_buffer_size} 步 ({config.num_envs} env × "
          f"{config.replay_buffer_size // config.num_envs} steps)")
    print(f"MLP hidden: {config.mlp_hidden},  RNN hidden: {config.rnn_hidden_size}")
    print(f"CSV 日志: vanilla_training_log.csv")
    print(f"模型存档: checkpoints/ (每 10 iter, 保留最新 5 个)")

    # ---- 评估准入准则 ----
    MIN_EPISODES_TO_EVAL = 50  # 最少完成 50 局后才允许覆盖 best 模型

    # ---- 1. 创建并行环境 ----
    num_steps = config.replay_buffer_size // config.num_envs
    env_kwargs = dict(max_num_blue=config.num_blue, max_num_red=config.num_red,
                      max_steps=config.max_episode_length,
                      enable_gcas_for_blue=config.enable_blue_gcas)
    print(f"正在启动 {config.num_envs} 个 worker 进程...", flush=True)
    vec_env = SubprocVecEnv(config.num_envs, env_kwargs)

    red_ids = [f"red_{i}" for i in range(config.num_red)]
    blue_ids = [f"blue_{i}" for i in range(config.num_blue)]

    # ---- 2. 初始化网络 ----
    actor = VanillaActor(obs_dim=obs_dim, action_dim=config.action_dim,
                         hidden=config.mlp_hidden,
                         rnn_hidden=config.rnn_hidden_size).to(device)
    global_obs_dim = obs_dim * config.num_red
    critic = CentralizedCritic(global_obs_dim=global_obs_dim,
                                hidden=config.mlp_hidden).to(device)

    print(f"Actor  params:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params:  {sum(p.numel() for p in critic.parameters()):,}  "
          f"(centralized, global_obs_dim={global_obs_dim})")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # ---- 3. 从 best checkpoint 恢复训练 (消融 r_ceil 后重新起航) ----
    actor_best_path = "checkpoints/vanilla_actor_best.pt"
    critic_best_path = "checkpoints/centralized_critic_best.pt"
    if (config.resume_from_best and os.path.exists(actor_best_path)
            and os.path.exists(critic_best_path)):
        actor.load_state_dict(torch.load(actor_best_path, map_location=device,
                                         weights_only=True))
        critic.load_state_dict(torch.load(critic_best_path, map_location=device,
                                          weights_only=True))
        print(f"✓ 已加载 best checkpoint 权重 (actor + critic)")
    else:
        print(f"⚠ best checkpoint 不存在，使用随机初始化权重")

    # ---- 4. 初始 RNN 状态 ----
    rnn_hidden_actor = np.zeros(
        (config.num_envs, config.num_red, config.rnn_hidden_size), dtype=np.float32)

    # ---- 5. 重置 ----
    print(f"正在重置 {config.num_envs} 个环境...", flush=True)
    t_reset = time.perf_counter()
    raw_obs_list = vec_env.reset(timeout=300.0)
    print(f"重置完成 ({time.perf_counter() - t_reset:.0f}s)", flush=True)
    print("=" * 70)

    # ---- 6. 训练循环 ----
    total_steps = 0
    iteration = 1
    total_episodes = 0
    red_wins = 0
    blue_wins = 0
    draws = 0
    death_stats = {"red": Counter(), "blue": Counter()}
    red_missiles_total = 0.0
    blue_missiles_total = 0.0
    best_win_rate = 0.0
    best_reward = -float("inf")

    # Episodic reward trackers — only fully-completed episodes contribute (Red only)
    recent_ep_rewards_red = deque(maxlen=50)
    # Per-component episodic trackers for red team diagnostics
    COMP_KEYS = ["r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv", "r_end", "r_death"]
    recent_ep_comps_red: deque[dict] = deque(maxlen=50)
    recent_ep_missiles_red = deque(maxlen=50)
    recent_ep_missiles_blue = deque(maxlen=50)
    recent_ep_red_alive = deque(maxlen=50)
    recent_ep_blue_alive = deque(maxlen=50)
    current_ep_reward_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_comp_red = {k: np.zeros(config.num_envs, dtype=np.float64)
                           for k in COMP_KEYS}
    current_ep_missiles_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_missiles_blue = np.zeros(config.num_envs, dtype=np.float32)
    # Results log for offline plotting — accumulates all key metrics per iteration
    results_log: list[dict] = []

    while total_steps < config.total_env_steps:
        t_start = time.perf_counter()

        buffer = RolloutBuffer(
            num_steps=num_steps, num_envs=config.num_envs,
            num_red=config.num_red, action_dim=config.action_dim,
            rnn_hidden_size=config.rnn_hidden_size,
        )
        buffer.rnn_actor_init = rnn_hidden_actor.copy()

        # Per-iteration episode counters (for sliding-window win rate)
        iter_episodes = 0
        iter_red_wins = 0

        # ---- Rollout ----
        for step in range(num_steps):
            actions_list = []

            # Fetch engaged-targets sets from all envs (one pipe round-trip)
            engaged_sets = vec_env.env_method("refresh_engaged_targets")

            for env_idx in range(config.num_envs):
                env_obs = raw_obs_list[env_idx]
                # Guard against empty obs from a dead worker that couldn't be restarted
                if not env_obs or len(env_obs) == 0:
                    env_actions = {aid: np.zeros(config.action_dim, dtype=np.float32)
                                   for aid in red_ids + blue_ids}
                    actions_list.append(env_actions)
                    continue
                env_actions = {}

                # 蓝方协同目标分配 + 引导律 (GCAS / 导弹规避在 env 层自动保护)
                blue_obs_dict = {bid: env_obs[bid] for bid in blue_ids}
                env_actions.update(blue_coordinated_actions(
                    blue_obs_dict, config.num_blue, config.num_red,
                    engaged_targets=engaged_sets[env_idx]))

                # 红方 Actor 采样 + 全局观测构建
                red_obs_flat_all = []  # all agents (alive/dead) for global state
                red_obs_tensors = []
                alive_red_indices = []
                for i, rid in enumerate(red_ids):
                    obs_np = env_obs[rid]
                    obs_flat = _flatten_obs(obs_np)
                    red_obs_flat_all.append(obs_flat)
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        red_obs_tensors.append(obs_flat)
                        alive_red_indices.append(i)
                    else:
                        env_actions[rid] = np.zeros(config.action_dim, dtype=np.float32)
                        buffer.store_step(step, env_idx, i, obs_flat,
                                          np.zeros(config.action_dim, dtype=np.float32),
                                          0.0, 0.0, 0.0, 1.0, alive=False)

                if alive_red_indices:
                    obs_batch = torch.as_tensor(
                        np.stack(red_obs_tensors), dtype=torch.float32, device=device)
                    batch_rnn_a = torch.as_tensor(
                        rnn_hidden_actor[env_idx, alive_red_indices], device=device)

                    with torch.no_grad():
                        action_dist, new_rnn_a = actor(obs_batch, batch_rnn_a)
                        action_raw = action_dist.sample()
                        # Clamp to [-1, 1] before env / log_prob (Gaussian tail guard)
                        action = action_raw.clamp(-0.999, 0.999)
                        log_prob = action_dist.log_prob(action).sum(dim=-1)

                    for k, i in enumerate(alive_red_indices):
                        rid = red_ids[i]
                        env_actions[rid] = action[k].cpu().numpy()
                        rnn_hidden_actor[env_idx, i] = new_rnn_a[k].cpu().numpy()
                        buffer.store_step(step, env_idx, i,
                                          red_obs_tensors[k],
                                          action[k].cpu().numpy(),
                                          0.0, 0.0,  # value filled below
                                          log_prob[k].item(), 0.0, alive=True)

                # ---- Centralized critic: global state V(s) shared by all agents ----
                global_obs_np = np.concatenate(red_obs_flat_all)  # (global_obs_dim,)
                global_obs_t = torch.as_tensor(
                    global_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    v_global = critic(global_obs_t).item()
                # All red agents (alive and dead) share the centralized value
                for i in range(config.num_red):
                    if buffer.alive[step, env_idx, i]:
                        buffer.values[step, env_idx, i] = v_global

                actions_list.append(env_actions)

            # ---- 环境步进 ----
            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(actions_list)

            # 填充 reward / done + 追踪 episode 结局
            for env_idx in range(config.num_envs):
                rew = rewards_list[env_idx]
                don = dones_list[env_idx]
                info = infos_list[env_idx]

                # ---- accumulate step rewards FIRST (incl. terminal r_end for dead agents) ----
                for i, rid in enumerate(red_ids):
                    if buffer.alive[step, env_idx, i]:
                        buffer.rewards[step, env_idx, i] = float(rew.get(rid, 0.0))
                        buffer.dones[step, env_idx, i] = float(don.get(rid, False))
                    if don.get(rid, False):
                        rnn_hidden_actor[env_idx, i] = np.zeros(
                            config.rnn_hidden_size, dtype=np.float32)
                    current_ep_reward_red[env_idx] += rew.get(rid, 0.0)
                    # Accumulate per-component diagnostics
                    rcinfo = info.get(rid, {})
                    for k in COMP_KEYS:
                        current_ep_comp_red[k][env_idx] += rcinfo.get(k, 0.0)

                inf = infos_list[env_idx]
                for rid in red_ids:
                    fired = inf.get(rid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_red[env_idx] += fired
                    red_missiles_total += fired
                for bid in blue_ids:
                    fired = inf.get(bid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_blue[env_idx] += fired
                    blue_missiles_total += fired

                # ---- episodic settlement AFTER accumulation (terminal r_end is included) ----
                if all(don.values()):
                    total_episodes += 1
                    iter_episodes += 1
                    blue_alive = sum(
                        1 for bid in blue_ids
                        if info.get(bid, {}).get("alive", False))
                    red_alive = sum(
                        1 for rid in red_ids
                        if info.get(rid, {}).get("alive", False))
                    outcome = _episode_outcome(red_alive, blue_alive)
                    if outcome == "red":
                        red_wins += 1
                        iter_red_wins += 1
                    elif outcome == "blue":
                        blue_wins += 1
                    else:
                        draws += 1
                    # Accumulate death reasons
                    for bid in blue_ids:
                        dr = info.get(bid, {}).get("death_reason")
                        if dr:
                            death_stats["blue"][dr] += 1
                    for rid in red_ids:
                        dr = info.get(rid, {}).get("death_reason")
                        if dr:
                            death_stats["red"][dr] += 1

                    recent_ep_rewards_red.append(float(current_ep_reward_red[env_idx]))
                    # Persist component breakdown
                    recent_ep_comps_red.append(
                        {k: float(current_ep_comp_red[k][env_idx]) for k in COMP_KEYS})
                    recent_ep_missiles_red.append(float(current_ep_missiles_red[env_idx]))
                    recent_ep_missiles_blue.append(float(current_ep_missiles_blue[env_idx]))
                    recent_ep_red_alive.append(float(red_alive))
                    recent_ep_blue_alive.append(float(blue_alive))
                    current_ep_reward_red[env_idx] = 0.0
                    for k in COMP_KEYS:
                        current_ep_comp_red[k][env_idx] = 0.0
                    current_ep_missiles_red[env_idx] = 0.0
                    current_ep_missiles_blue[env_idx] = 0.0

            raw_obs_list = next_obs_list
            total_steps += config.num_envs

        # 保存 rollout 结束后的 RNN 状态 (用于 GAE bootstrap)
        buffer.rnn_actor_final = rnn_hidden_actor.copy()

        # 计算 GAE bootstrap 值: centralized V(s_T) shared by all agents
        for env_idx in range(config.num_envs):
            env_obs = raw_obs_list[env_idx]
            if not env_obs or len(env_obs) == 0:
                continue
            # Build global obs from all red agents' final observations
            global_obs_parts = []
            for rid in red_ids:
                if rid in env_obs:
                    global_obs_parts.append(_flatten_obs(env_obs[rid]))
                else:
                    global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
            global_obs_np = np.concatenate(global_obs_parts)
            global_obs_t = torch.as_tensor(global_obs_np, dtype=torch.float32,
                                           device=device).unsqueeze(0)
            with torch.no_grad():
                v_bootstrap = critic(global_obs_t).item()
            for i in range(config.num_red):
                buffer.bootstrap_values[env_idx, i] = v_bootstrap

        # ---- PPO 更新 ----
        stats = ppo_update(actor, critic, actor_opt, critic_opt,
                           buffer, config, device, total_steps=total_steps)

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
                             f"{stats['actor_loss']:.6f}",
                             f"{stats['critic_loss']:.6f}",
                             f"{stats['entropy']:.6f}",
                             f"{avg_r_red:.4f}",
                             f"{red_win_rate:.6f}",
                             f"{std_r_red:.4f}",
                             f"{iter_win_rate:.6f}",
                             f"{avg_m_red:.1f}",
                             f"{avg_m_blue:.1f}",
                             total_episodes, red_wins, blue_wins, draws,
                             f"{red_alive_mean:.4f}",
                             f"{blue_alive_mean:.4f}",
                             red_deaths_missile,
                             red_deaths_crash,
                             blue_deaths_missile,
                             blue_deaths_crash,
                             red_missile_hits,
                             blue_missile_hits,
                             f"{red_missile_hit_rate:.6f}",
                             f"{blue_missile_hit_rate:.6f}",
                             f"{kd_red:.6f}",
                             f"{rwr:.6f}"])
        csv_file.flush()

        # ---- 持久化：results/ 绘图数据 (累计 + 每 1M 步自动保存) ----
        results_log.append({
            "Step":           total_steps,
            "Iteration":      iteration,
            "RedMeanReward":  avg_r_red,
            "RedRewardStd":   std_r_red,
            "WinRateRecent":  iter_win_rate,
            "WinRateCumul":   red_win_rate,
            "RedMissiles":    avg_m_red,
            "BlueMissiles":   avg_m_blue,
            "Episodes":       total_episodes,
            "RedWins":        red_wins,
            "BlueWins":       blue_wins,
            "Draws":          draws,
            "RedAliveMean":   red_alive_mean,
            "BlueAliveMean":  blue_alive_mean,
            "RedDeathsMissile": red_deaths_missile,
            "RedDeathsCrash": red_deaths_crash,
            "BlueDeathsMissile": blue_deaths_missile,
            "BlueDeathsCrash": blue_deaths_crash,
            "RedMissileHits": red_missile_hits,
            "BlueMissileHits": blue_missile_hits,
            "RedMissileHitRate": red_missile_hit_rate,
            "BlueMissileHitRate": blue_missile_hit_rate,
            "KD_Red":         kd_red,
            "RWR":            rwr,
            "ActorLoss":      stats["actor_loss"],
            "CriticLoss":     stats["critic_loss"],
            "Entropy":        stats["entropy"],
            "r_pitch":        avg_comps.get("r_pitch", 0.0),
            "r_roll":         avg_comps.get("r_roll", 0.0),
            "r_alt":          avg_comps.get("r_alt", 0.0),
            "r_bound":        avg_comps.get("r_bound", 0.0),
            "r_vel":          avg_comps.get("r_vel", 0.0),
            "r_adv":          avg_comps.get("r_adv", 0.0),
            "r_end":          avg_comps.get("r_end", 0.0),
            "r_death":        avg_comps.get("r_death", 0.0),
        })
        milestone_cur = total_steps // 1_000_000
        milestone_prev = (total_steps - config.num_envs * num_steps) // 1_000_000
        if milestone_cur > milestone_prev or total_steps >= config.total_env_steps:
            os.makedirs("results", exist_ok=True)
            with open("results/vanilla_mappo_results.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(results_log[0].keys())
                for row in results_log:
                    w.writerow(row.values())
            print(f"  [Results saved] results/vanilla_mappo_results.csv "
                  f"({len(results_log)} rows)", flush=True)

        # ---- 终端打印 ----
        def _fmt_death(counter: Counter) -> str:
            order = ["Missile_Kill", "Crash_LowAlt", "Crash_HighAlt",
                     "Crash_OverG", "Crash_Extreme"]
            parts = []
            for k in order:
                parts.append(f"{k.replace('Crash_','').replace('Missile_','')}:{counter.get(k, 0)}")
            # Catch any unexpected reason
            for k in sorted(counter.keys()):
                if k not in order:
                    parts.append(f"{k}:{counter[k]}")
            return ", ".join(parts)

        print(f"Iter {iteration:5d} | "
              f"total_steps={total_steps:9d} | "
              f"t={t_elapsed:5.1f}s | "
              f"R_red={avg_r_red:+8.1f} [{comp_str}] | "
              f"M_red={avg_m_red:.0f} M_blue={avg_m_blue:.0f} | "
              f"ActorLoss={stats['actor_loss']:+.4f} "
              f"CriticLoss={stats['critic_loss']:+.4f} "
              f"EntCoef={_current_entropy_coef(config, total_steps):.4f} "
              f"Entropy={stats['entropy']:.4f} | "
              f"WinRate_red={red_win_rate:.3f} "
              f"(Ep={total_episodes} W={red_wins}/{blue_wins}/{draws}) | "
              f"Deaths: Red[{_fmt_death(death_stats['red'])}] "
              f"Blue[{_fmt_death(death_stats['blue'])}]")

        # ---- 持久化：高频轮转存档 (每 10 iter, 保留最新 5 个) ----
        if iteration % 10 == 0:
            actor_path = f"checkpoints/vanilla_actor_latest_{iteration:06d}.pt"
            critic_path = f"checkpoints/centralized_critic_latest_{iteration:06d}.pt"
            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)
            # 轮转清理：删除超出保留数量的旧 checkpoint
            _cleanup_rotating_checkpoints("checkpoints", "vanilla_actor_latest", keep=5)
            _cleanup_rotating_checkpoints("checkpoints", "centralized_critic_latest", keep=5)

        # ---- 持久化：最佳模型拦截 (需满足评估准入准则) ----
        # 以近期奖励为主指标（反映当前模型真实表现），累计胜率为 tiebreaker
        if total_episodes >= MIN_EPISODES_TO_EVAL:
            is_better = (red_win_rate > best_win_rate) or (
                abs(red_win_rate - best_win_rate) < 1e-6 and avg_r_red > best_reward)
            if is_better:
                best_win_rate = red_win_rate
                best_reward = avg_r_red
                torch.save(actor.state_dict(),
                           "checkpoints/vanilla_actor_best.pt")
                torch.save(critic.state_dict(),
                           "checkpoints/centralized_critic_best.pt")
                print(f"  *** New Best Model Saved! "
                      f"(Reward={best_reward:+.2f}, WinRate={best_win_rate:.4f}) ***")

        iteration += 1

    # ---- 持久化：最终模型存档 ----
    torch.save(actor.state_dict(), "checkpoints/vanilla_actor_final.pt")
    torch.save(critic.state_dict(), "checkpoints/centralized_critic_final.pt")
    print("=" * 70)
    print(f"最终模型已保存至 checkpoints/")
    print(f"Results 已保存至 results/vanilla_mappo_results.csv ({len(results_log)} rows)")
    print(f"总 Episodes: {total_episodes}  "
          f"红方胜: {red_wins}  蓝方胜: {blue_wins}  平局: {draws}  "
          f"红方胜率: {red_win_rate:.4f}")
    csv_file.close()

    # ---- 清理 ----
    vec_env.close()
    print("基线训练完成！")


if __name__ == "__main__":
    mp.freeze_support()
    main()
