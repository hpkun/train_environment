"""Train TAM HAPPO direct with true multiprocessing rollout workers.

Adapted from hetero_uav/scripts/train_happo_reference_parallel.py.
Replaces serial env batching with process-isolated JSBSim workers.
"""
from __future__ import annotations

import argparse, csv, json, math, multiprocessing as mp, os, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
#  Worker timeout helpers (same as hetero_uav)
# ---------------------------------------------------------------------------
class _WorkerTimeout(Exception):
    def __init__(self, env_idx: int, command: str, timeout_sec: float):
        self.env_idx = int(env_idx)
        self.command = str(command)
        self.timeout_sec = float(timeout_sec)
        super().__init__(f"[env_idx={env_idx}] {command} timed out after {timeout_sec:.1f}s")


# ---------------------------------------------------------------------------
#  Env worker process
# ---------------------------------------------------------------------------
def _env_worker(remote, parent_remote, env_kwargs: dict) -> None:
    parent_remote.close()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env = None
    try:
        from uav_env import make_env
        env = make_env(env_kwargs["config"], env_type="jsbsim_hetero",
                       hetero_reward_mode=env_kwargs["reward_mode"],
                       max_steps=env_kwargs["max_steps"])
        meta = {
            "red_ids": list(env.red_ids), "blue_ids": list(env.blue_ids),
            "agent_ids": list(env.agent_ids), "max_steps": int(getattr(env, "max_steps", 0)),
            "role_ids": [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids],
        }
        remote.send(("ready", meta, _worker_diag(env)))
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send(("ok", obs, info, _worker_diag(env)))
            elif cmd == "step":
                action_dict = {}
                for k, v in data.items():
                    action_dict[k] = np.asarray(v, dtype=np.int64)
                obs, rewards, terminated, truncated, info = env.step(action_dict)
                remote.send(("ok", obs, rewards, terminated, truncated, info, _worker_diag(env)))
            elif cmd == "close":
                remote.send(("ok",))
                break
            else:
                remote.send(("error", f"unknown worker command: {cmd}"))
    except Exception as exc:
        try:
            remote.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: remote.close()
        except Exception: pass


def _worker_diag(env) -> dict:
    red_alive = sum(1 for rid in env.red_ids if env.red_planes.get(rid) and env.red_planes[rid].is_alive)
    blue_alive = sum(1 for bid in env.blue_ids if env.blue_planes.get(bid) and env.blue_planes[bid].is_alive)
    return {
        "red_alive": int(red_alive), "blue_alive": int(blue_alive),
        "mav_alive": bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive),
        "missile_count": int(len(getattr(env, "_missiles_in_flight", {}))),
        "sim_time": float(getattr(env, "current_step", 0)) * float(getattr(env, "env_dt", 0.0)),
        "engaged_targets": list(getattr(env, "refresh_engaged_targets", lambda: set())() or set()),
    }


# ---------------------------------------------------------------------------
#  Remote proxy (same as hetero_uav)
# ---------------------------------------------------------------------------
class RemoteEnvProxy:
    def __init__(self, meta: dict, diag: dict | None = None):
        self.red_ids = list(meta["red_ids"])
        self.blue_ids = list(meta["blue_ids"])
        self.agent_ids = list(meta["agent_ids"])
        self.max_steps = int(meta.get("max_steps", 0))
        self.diag = diag or {}

    def update_diag(self, diag: dict) -> None:
        self.diag = diag or {}


# ---------------------------------------------------------------------------
#  Parallel env manager
# ---------------------------------------------------------------------------
class ParallelEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, reset_timeout: float,
                 step_timeout: float, startup_delay: float = 0.5):
        self.num_envs = int(num_envs)
        self.env_kwargs = dict(env_kwargs)
        self.reset_timeout = float(reset_timeout)
        self.step_timeout = float(step_timeout)
        self.ctx = mp.get_context("spawn")
        self.remotes, self.processes, self.metas, self.diags = [], [], [], []
        self.worker_restart_count = 0
        for idx in range(self.num_envs):
            self._start_worker(idx)
            if idx < self.num_envs - 1 and startup_delay > 0:
                time.sleep(startup_delay)

    def _start_worker(self, idx: int) -> None:
        parent_remote, worker_remote = self.ctx.Pipe()
        proc = self.ctx.Process(target=_env_worker, args=(worker_remote, parent_remote, self.env_kwargs), daemon=True)
        proc.start()
        worker_remote.close()
        if not parent_remote.poll(self.reset_timeout):
            proc.terminate()
            raise TimeoutError(f"worker {idx} did not initialize within {self.reset_timeout:.1f}s")
        status = parent_remote.recv()
        if status[0] != "ready":
            proc.terminate()
            raise RuntimeError(f"worker {idx} failed to initialize: {status}")
        if idx < len(self.remotes):
            self.remotes[idx], self.processes[idx], self.metas[idx], self.diags[idx] = parent_remote, proc, status[1], status[2]
        else:
            self.remotes.append(parent_remote); self.processes.append(proc)
            self.metas.append(status[1]); self.diags.append(status[2])

    def _restart_worker(self, idx: int) -> None:
        self.worker_restart_count += 1
        try: self.remotes[idx].close()
        except Exception: pass
        proc = self.processes[idx]
        if proc.is_alive(): proc.terminate(); proc.join(timeout=5)
        self._start_worker(idx)

    def _recv(self, idx: int, timeout: float, command: str):
        remote = self.remotes[idx]
        if remote.poll(timeout):
            msg = remote.recv()
            if msg[0] == "error": raise RuntimeError(f"worker {idx} {command} failed: {msg[1]}")
            return msg
        self._restart_worker(idx)
        raise _WorkerTimeout(idx, command, timeout)

    def reset_all(self, seed: int):
        for idx, remote in enumerate(self.remotes):
            remote.send(("reset", seed + idx))
        states = []
        for idx in range(self.num_envs):
            _status, obs, info, diag = self._recv(idx, self.reset_timeout, "reset")
            self.diags[idx] = diag
            states.append((obs, info, diag))
        return states

    def reset_one(self, idx: int, seed: int):
        self.remotes[idx].send(("reset", seed))
        _status, obs, info, diag = self._recv(idx, self.reset_timeout, "reset")
        self.diags[idx] = diag
        return obs, info, diag

    def step_all(self, action_dicts: list[dict]):
        for remote, actions in zip(self.remotes, action_dicts):
            remote.send(("step", actions))
        results = []
        for idx in range(self.num_envs):
            _status, obs, rewards, terminated, truncated, info, diag = self._recv(idx, self.step_timeout, "step")
            self.diags[idx] = diag
            results.append((obs, rewards, terminated, truncated, info, diag))
        return results

    def close(self) -> None:
        for remote in self.remotes:
            try: remote.send(("close", None)); remote.poll(2) and remote.recv()
            except Exception: pass
            try: remote.close()
            except Exception: pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive(): proc.terminate()


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--output-dir", default="outputs/tam_happo_direct_parallel")
    p.add_argument("--total-env-steps", type=int, default=1024)
    p.add_argument("--rollout-length", type=int, default=256)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--policy-arch", default="tam_categorical_recurrent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent-policy", default="tam_direct_fsm")
    p.add_argument("--reward-mode", default="tam_paper_reward_v1")
    p.add_argument("--advantage-mode", default="per_agent_reward",
                   choices=["team_average", "per_agent_reward"])
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--tam-paper-mode", action="store_true", default=True)
    p.add_argument("--happo-update-granularity", default="agent")
    p.add_argument("--eval-during-training", action="store_true")
    p.add_argument("--eval-interval-steps", type=int, default=25000)
    p.add_argument("--eval-at-start", action="store_true")
    p.add_argument("--train-eval-episodes", type=int, default=5)
    p.add_argument("--eval-configs", nargs="*", default=None)
    p.add_argument("--save-eval-checkpoints", action="store_true")
    p.add_argument("--eval-checkpoint-metric", default="combined")
    p.add_argument("--keep-eval-checkpoints", type=int, default=10)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--enable-rich-logging", action="store_true")
    p.add_argument("--rich-log-dir", default=None)
    p.add_argument("--heartbeat-log", default=None)
    p.add_argument("--heartbeat-stall-timeout-sec", type=float, default=0.0)
    p.add_argument("--exit-on-heartbeat-stall", action="store_true")
    p.add_argument("--reset-timeout-sec", type=float, default=300.0)
    p.add_argument("--step-timeout-sec", type=float, default=120.0)
    p.add_argument("--worker-startup-delay-sec", type=float, default=1.0)
    return p.parse_args()


def _alive_from_diag(diag: dict) -> dict:
    return {"red": int(diag.get("red_alive", 0)), "blue": int(diag.get("blue_alive", 0))}


def _transitions_per_rollout(length: int, num_envs: int) -> int:
    return int(length) * int(num_envs)


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def _episode_outcome(diag: dict, truncated: dict, length: int, max_steps: int) -> dict:
    ra, ba = int(diag.get("red_alive", 0)), int(diag.get("blue_alive", 0))
    timeout = bool((truncated and all(truncated.values())) or length >= max_steps)
    if ba == 0 and ra > 0: return {"winner": "red", "end_reason": "blue_eliminated", "red_alive": ra, "blue_alive": ba}
    if ra == 0 and ba > 0: return {"winner": "blue", "end_reason": "red_eliminated", "red_alive": ra, "blue_alive": ba}
    if ra == 0 and ba == 0: return {"winner": "draw", "end_reason": "mutual_elimination", "red_alive": ra, "blue_alive": ba}
    if timeout:
        winner = "red" if ra > ba else ("blue" if ba > ra else "draw")
        return {"winner": winner, "end_reason": "timeout", "red_alive": ra, "blue_alive": ba}
    return {"winner": "none", "end_reason": "ongoing", "red_alive": ra, "blue_alive": ba}


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.happo.tam_categorical_happo_trainer import TAMCategoricalHAPPOTrainer
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer
    from algorithms.happo.rollout_safety import zero_inactive_actions, zero_inactive_hidden
    from algorithms.mappo.opponent_policy import OpponentPolicy

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / args.output_dir
    for d in ["latest", "best", "checkpoints"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    DEFAULT_EVAL_CONFIGS = [
        "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml",
    ]

    vec_env = ParallelEnv(args.num_envs, {"config": args.config, "reward_mode": args.reward_mode, "max_steps": args.max_steps},
                          reset_timeout=args.reset_timeout_sec, step_timeout=args.step_timeout_sec,
                          startup_delay=args.worker_startup_delay_sec)
    proxies = [RemoteEnvProxy(meta, diag) for meta, diag in zip(vec_env.metas, vec_env.diags)]
    env = proxies[0]
    adapter = HeteroObsAdapterV2()
    actor_dim = adapter.flat_actor_obs_dim
    critic_dim = adapter.critic_state_dim
    roles = vec_env.metas[0]["role_ids"]
    action_dim = 4
    action_dist = "multidiscrete_categorical"

    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=19, actor_obs_dim=actor_dim, critic_state_dim=critic_dim,
        action_dim=action_dim, action_levels=40, rnn_hidden_size=128).to(device)
    if args.init_checkpoint:
        path = Path(args.init_checkpoint)
        if not path.is_absolute(): path = ROOT / path
        policy.load(path, map_location=device)
        print(f"Loaded init_checkpoint: {path}", flush=True)

    trainer = TAMCategoricalHAPPOTrainer(
        policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        clip_param=args.clip_param, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        happo_update_granularity=args.happo_update_granularity,
        agent_ids=env.red_ids, advantage_mode=args.advantage_mode)

    opponents = [OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17 + i) for i in range(args.num_envs)]
    env_states = vec_env.reset_all(args.seed)
    obs_list = [s[0] for s in env_states]
    info_list = [s[1] for s in env_states]
    for proxy, s in zip(proxies, env_states): proxy.update_diag(s[2])

    tpr = _transitions_per_rollout(args.rollout_length, args.num_envs)
    iterations = int(math.ceil(args.total_env_steps / tpr))
    total_steps, episodes = 0, 0
    rnn_hidden = np.zeros((args.num_envs, len(env.red_ids), 128), dtype=np.float32)
    cur_ret = [np.zeros(len(env.red_ids), dtype=np.float32) for _ in range(args.num_envs)]
    cur_len = [0] * args.num_envs
    cur_ep_id = [0] * args.num_envs
    recent = deque(maxlen=100)
    prev_hits = [{"red": 0, "blue": 0} for _ in range(args.num_envs)]
    nan_detected = False
    last_eval = -999999 if args.eval_at_start else 0

    train_log = out_dir / "train_log.csv"
    with train_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "total_steps", "avg_return", "red_win", "blue_win",
                         "draw", "timeout", "mav_survival", "red_alive_final", "blue_alive_final",
                         "red_fired", "blue_fired", "red_hits", "blue_hits",
                         "actor_loss_mav", "actor_loss_uav", "critic_loss",
                         "entropy_mav", "entropy_uav", "approx_kl_mav", "approx_kl_uav",
                         "advantage_mode", "dilution_ratio_abs", "per_agent_enabled"])

        for iteration in range(1, iterations + 1):
            rt = min(tpr, args.total_env_steps - total_steps)
            if rt <= 0: break
            buf = HAPPORolloutBuffer(rt, len(env.red_ids), actor_dim, critic_dim, action_dim,
                                     roles, rnn_hidden_size=128, action_dtype=np.int64, num_envs=args.num_envs)
            red_fired = blue_fired = red_hits = blue_hits = 0

            while len(buf) < rt and total_steps < args.total_env_steps:
                batch_records, action_dicts = [], []
                for env_idx, proxy in enumerate(proxies):
                    if len(buf) + len(batch_records) >= rt or total_steps + len(batch_records) >= args.total_env_steps:
                        break
                    obs, info = obs_list[env_idx], info_list[env_idx]
                    adapted = adapter.adapt_all(obs, info=info, red_ids=proxy.red_ids, blue_ids=proxy.blue_ids)
                    ao = np.stack([adapted["actor_obs"].get(rid, np.zeros(actor_dim, dtype=np.float32)) for rid in proxy.red_ids])
                    critic = adapted["critic_state"]
                    active = np.array([1.0 if (info.get(rid, {}).get("alive", True) if isinstance(info.get(rid, {}), dict) else True) else 0.0 for rid in proxy.red_ids], dtype=np.float32)
                    if rnn_hidden is not None:
                        rnn_hidden[env_idx] = zero_inactive_hidden(rnn_hidden[env_idx], active)
                    rnn_pre = rnn_hidden[env_idx].copy() if rnn_hidden is not None else None
                    act_kw = {}
                    if rnn_hidden is not None:
                        act_kw["rnn_hidden"] = torch.as_tensor(rnn_hidden[env_idx], device=device)
                    with torch.no_grad():
                        out = policy.act(torch.as_tensor(ao, device=device), roles=roles,
                                        critic_state=torch.as_tensor(critic, device=device),
                                        deterministic=False, **act_kw)
                    actions_raw = out["action"].cpu().numpy()
                    actions_np = zero_inactive_actions(actions_raw, active)
                    log_probs = out["log_prob"].cpu().numpy()
                    value = float(out["value"].item())
                    if rnn_hidden is not None and "rnn_hidden" in out:
                        rnn_hidden[env_idx] = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)
                    active_rows = active > 0.5
                    if active_rows.any():
                        if not np.isfinite(actions_np[active_rows]).all(): nan_detected = True
                        if not np.isfinite(value): nan_detected = True
                    # Pipe test proves np.int64 arrays survive multiprocessing fine
                    ad = {rid: actions_np[i].astype(np.int64).copy() for i, rid in enumerate(proxy.red_ids)}
                    opp_act = opponents[env_idx].act(obs, proxy.blue_ids, env=proxy)
                    for k, v in opp_act.items():
                        # tam_direct_fsm returns (4,) float32 direct-FCS values.
                        # Quantize to categorical indices for the env.
                        levels = 40
                        indices = np.round((np.clip(v, -1.0, 1.0) + 1.0) / 2.0 * (levels - 1))
                        ad[k] = indices.astype(np.int64).copy()
                    action_dicts.append(ad)
                    action_dicts.append(ad)
                    batch_records.append({"env_idx": env_idx, "ao": ao, "critic": critic,
                                          "actions": actions_np, "log_probs": log_probs,
                                          "value": value, "active": active, "rnn_pre": rnn_pre})
                if nan_detected or not batch_records: break

                step_results = vec_env.step_all(action_dicts)
                for record, result in zip(batch_records, step_results):
                    env_idx = record["env_idx"]; proxy = proxies[env_idx]
                    next_obs, rewards, terminated, truncated, next_info, diag = result
                    proxy.update_diag(diag)
                    reward_np = np.array([float(rewards.get(rid, 0.0)) for rid in proxy.red_ids], dtype=np.float32)
                    done = _team_done(terminated, truncated)
                    done_np = np.full(len(proxy.red_ids), float(done), dtype=np.float32)
                    next_value = 0.0
                    if not done:
                        na = adapter.adapt_all(next_obs, info=next_info, red_ids=proxy.red_ids, blue_ids=proxy.blue_ids)
                        with torch.no_grad():
                            next_value = float(policy.value(torch.as_tensor(na["critic_state"], device=device).unsqueeze(0)).item())
                    store_kw = {}
                    if record["rnn_pre"] is not None: store_kw["rnn_hidden"] = record["rnn_pre"]
                    buf.store(record["ao"], record["critic"], record["actions"], record["log_probs"],
                              reward_np, done_np, record["value"], record["active"],
                              next_value=next_value, env_id=env_idx, **store_kw)
                    cur_ret[env_idx] += reward_np; cur_len[env_idx] += 1; total_steps += 1
                    for aid in proxy.agent_ids:
                        f_step = int(next_info.get(aid, {}).get("missiles_fired_this_step", 0))
                        if aid.startswith("red_"): red_fired += f_step
                        else: blue_fired += f_step
                    mt = next_info.get("__missile_term__", {})
                    if isinstance(mt, dict):
                        rh = int(mt.get("red", {}).get("hit", 0)); bh = int(mt.get("blue", {}).get("hit", 0))
                        red_hits += max(rh - prev_hits[env_idx]["red"], 0)
                        blue_hits += max(bh - prev_hits[env_idx]["blue"], 0)
                        prev_hits[env_idx] = {"red": rh, "blue": bh}
                    if done:
                        out_c = _episode_outcome(diag, truncated, cur_len[env_idx], proxy.max_steps)
                        recent.append({"return": float(cur_ret[env_idx].mean()), "winner": out_c["winner"],
                                       "end_reason": out_c["end_reason"], "mav": diag.get("mav_alive", False),
                                       "red_alive": out_c["red_alive"], "blue_alive": out_c["blue_alive"]})
                        episodes += 1; cur_ret[env_idx][:] = 0.0; cur_len[env_idx] = 0
                        next_obs, next_info, reset_diag = vec_env.reset_one(env_idx, args.seed + total_steps + env_idx)
                        proxy.update_diag(reset_diag); cur_ep_id[env_idx] += 1
                        if rnn_hidden is not None: rnn_hidden[env_idx][:] = 0.0
                        prev_hits[env_idx] = {"red": 0, "blue": 0}
                    obs_list[env_idx] = next_obs; info_list[env_idx] = next_info
            if nan_detected: break

            stats = trainer.update(buf)
            rec = list(recent); n = max(len(rec), 1)
            avg_ret = float(np.mean([r["return"] for r in rec])) if rec else 0.0
            rw = sum(1 for r in rec if r["winner"] == "red") / n
            bw = sum(1 for r in rec if r["winner"] == "blue") / n
            dr = sum(1 for r in rec if r["winner"] == "draw") / n
            to = sum(1 for r in rec if r["end_reason"] == "timeout") / n
            ms = sum(1 for r in rec if r["mav"]) / n
            ra = float(np.mean([r["red_alive"] for r in rec])) if rec else 0.0
            ba = float(np.mean([r["blue_alive"] for r in rec])) if rec else 0.0

            writer.writerow([iteration, total_steps, f"{avg_ret:.4f}", f"{rw:.4f}", f"{bw:.4f}",
                             f"{dr:.4f}", f"{to:.4f}", f"{ms:.4f}", f"{ra:.2f}", f"{ba:.2f}",
                             red_fired, blue_fired, red_hits, blue_hits,
                             f"{stats['actor_loss_mav']:.6f}", f"{stats['actor_loss_uav']:.6f}",
                             f"{stats['critic_loss']:.6f}", f"{stats['entropy_mav']:.6f}",
                             f"{stats['entropy_uav']:.6f}", f"{stats['approx_kl_mav']:.6f}",
                             f"{stats['approx_kl_uav']:.6f}",
                             stats.get("advantage_mode", ""), stats.get("dilution_ratio_abs", ""),
                             stats.get("per_agent_advantage_enabled", "")])
            f.flush()
            print(f"[happo-parallel] it={iteration:04d} step={total_steps}/{args.total_env_steps} "
                  f"ret={avg_ret:+.1f} rw={rw:.2f} bw={bw:.2f} mav={ms:.2f} "
                  f"lm={stats['actor_loss_mav']:.4f} lu={stats['actor_loss_uav']:.4f}", flush=True)

            if total_steps - last_eval >= args.eval_interval_steps and args.eval_during_training:
                last_eval = total_steps
                print("[happo-parallel] eval not yet implemented for parallel mode; use eval_tam_happo_direct.py manually", flush=True)

    latest = out_dir / "latest" / "model.pt"
    policy.save(latest)
    (out_dir / "latest" / "meta.json").write_text(json.dumps({
        "algorithm": "happo_reference_v0", "runner": "multiprocessing_parallel",
        "policy_arch": args.policy_arch, "config": args.config,
        "reward_mode": args.reward_mode, "advantage_mode": args.advantage_mode,
        "num_envs": args.num_envs, "rollout_length": args.rollout_length,
        "total_env_steps_actual": total_steps, "episodes": episodes, "nan_detected": nan_detected,
    }, indent=2), encoding="utf-8")
    vec_env.close()
    print(f"Saved {latest}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
