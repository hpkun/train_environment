"""
eval_acmi.py —— 加载训练好的模型，运行一局对战并导出 TacView .acmi 录像。

用法:
    python eval_acmi.py                          # 自动选择 best > final
    python eval_acmi.py --checkpoint checkpoints/vanilla_actor_final.pt
    python eval_acmi.py --random                 # 红方也随机 (不需要 checkpoint)
    python eval_acmi.py --output my_battle.acmi  # 指定输出文件名
"""
from __future__ import annotations

import os as _os
if "KMP_DUPLICATE_LIB_OK" not in _os.environ:
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Diagnostic: write startup marker to temp file to detect silent crashes
_DIAG = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_eval_startup.txt")
try:
    with open(_DIAG, "w") as f:
        f.write("eval_acmi.py started\n")
except Exception:
    pass

import argparse
import os
import sys
import traceback
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch

# Diagnostic marker 2
try:
    with open(_DIAG, "a") as f:
        f.write("imports_stage1: numpy+torch loaded\n")
except Exception:
    pass

from my_uav_env import UavCombatEnv
from my_uav_env.pid_controller import (
    PAPER_PID_DERIVATIVE_SEMANTICS,
    PAPER_PID_ERROR_DEFINITION,
)
from acmi_boundary_utils import (
    battlefield_boundary_acmi_lines as _battlefield_boundary_acmi_lines,
    maybe_write_battlefield_boundary_acmi as _maybe_write_battlefield_boundary_acmi,
    write_battlefield_boundary_acmi as _write_battlefield_boundary_acmi,
)

# Diagnostic marker 3
try:
    with open(_DIAG, "a") as f:
        f.write("imports_stage2: my_uav_env loaded\n")
except Exception:
    pass

from my_uav_env.alignment.reward_utils import (
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
)
from configs.paper_3v3_spec import (
    PAPER_BLUE_POLICY_PROFILE,
    PAPER_ENVIRONMENT_PROFILE,
    PAPER_MISSILE_GUIDANCE_MODE,
    PAPER_PID_PROFILE,
    PAPER_REWARD_MODE,
    PID_THROTTLE_BASE,
)
from train_vanilla_mappo import (
    ACTION_DISTRIBUTION_VERSION,
    ACTION_LOG_STD_INIT,
    CHECKPOINT_SCHEMA_VERSION,
    ENTROPY_ESTIMATOR_VERSION,
    VanillaActor,
    Config,
    _classify_death_reason,
    _compute_global_state_dim,
    _compute_obs_dim,
    _episode_outcome,
    _flatten_obs,
    _checkpoint_metadata,
    _joint_team_reward_once,
    _minimal_altitude_reward_config,
    _ratio_with_denominator_zero,
    _safe_div,
    _unpack_and_validate_checkpoint,
)

# Diagnostic marker 4
try:
    with open(_DIAG, "a") as f:
        f.write("imports_stage3: all imports done\n")
except Exception:
    pass


# ==============================================================================
#  VisualMissileTracker — 镜像真实 MissileSimulator 的 ACMI 可视化轨迹
# ==============================================================================

class VisualMissileTracker:
    """Mirrors a real ``MissileSimulator`` into the ACMI with a separate visual ID.

    The environment's built-in ``_render_frame()`` already logs the real missile
    at each env step, but when a missile launches and hits within a single env
    step (0.2 s at close range), it is never rendered in-flight — only the
    explosion appears.

    This class solves the problem by holding a **reference to the real
    ``MissileSimulator``** (which survives ``_cleanup_missiles()`` via
    ``parent.launch_missiles``) and mirroring its state frame-by-frame:

    - **Timing**: reads ``missile.is_alive`` every step — the visual missile
      dies at the EXACT same ACMI frame as the real hit.
    - **Orientation**: reads ``missile.get_rpy()`` — pitch/yaw come from the
      real 6-DOF velocity vector, not a static linear interpolation.
    - **Cleanup**: emits ``-{id}`` for the entity and ``-{id}F`` for the
      explosion floating object after a short display period.
    """

    EXPLOSION_PERSIST_STEPS = 3  # show explosion for 0.6 s (3 × 0.2 s env_dt)

    def __init__(self, visual_acmi_id: int, real_missile):
        self.visual_id = visual_acmi_id
        self._m = real_missile          # MissileSimulator reference (persistent)
        self._color = real_missile.color
        self._introduced = False
        self._exploded = False
        self._entity_removed = False
        self._explosion_cleanup = 0

    @property
    def is_expired(self) -> bool:
        """True when both entity and explosion have been fully cleaned up."""
        return self._explosion_cleanup == 0 and self._entity_removed

    def tick(self) -> list[str]:
        """Call once per env step.  Returns ACMI lines to inject."""
        m = self._m
        lines: list[str] = []

        # ---- Phase 1: in flight (real missile is alive) ----
        if m.is_alive:
            lon, lat, alt = m.get_geodetic()
            rpy = m.get_rpy()
            roll_d = float(np.rad2deg(rpy[0]))
            pitch_d = float(np.rad2deg(rpy[1]))
            yaw_d = float(np.rad2deg(rpy[2]))

            if not self._introduced:
                lines.append(
                    f"{self.visual_id},T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                    f"|{roll_d:.1f}|{pitch_d:.1f}|{yaw_d:.1f},"
                    f"Name=AIM-9L,Color={self._color}"
                )
                self._introduced = True
            else:
                lines.append(
                    f"{self.visual_id},T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                    f"|{roll_d:.1f}|{pitch_d:.1f}|{yaw_d:.1f}"
                )
            return lines

        # ---- Phase 2: real missile just died ----
        if not self._exploded:
            self._exploded = True
            lon, lat, alt = m.get_geodetic()

            # Follow original MissileSimulator.log() contract:
            #   HIT  → (a) remove entity, (b) emit explosion
            #   MISS → (a) remove entity only, NO explosion
            lines.append(f"-{self.visual_id}")
            self._entity_removed = True

            if self._m.is_success:
                self._explosion_cleanup = self.EXPLOSION_PERSIST_STEPS
                lines.append(
                    f"{self.visual_id}F,T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                    f"|0|0|0,Type=Misc+Explosion,Color=Yellow,Radius=300"
                )
            return lines

        # ---- Phase 3: explosion display countdown → remove floating object ----
        if self._explosion_cleanup > 0:
            self._explosion_cleanup -= 1
            if self._explosion_cleanup == 0:
                lines.append(f"-{self.visual_id}F")

        return lines


# ==============================================================================
#  Main eval loop
# ==============================================================================

def run_acmi(checkpoint_path: str | None, output_path: str = "eval_battle.acmi",
             num_red: int = 3, num_blue: int = 3, max_steps: int = 1400,
             draw_boundary: bool = False, boundary_half_size: float = 40000.0,
             obs_mode: str = "paper_strict",
             obs_normalization: str = "paper_fixed_v1",
             pid_profile: str = PAPER_PID_PROFILE,
             pid_throttle_base: float = PID_THROTTLE_BASE,
             reward_mode: str = PAPER_REWARD_MODE,
             missile_guidance_mode: str = PAPER_MISSILE_GUIDANCE_MODE,
             blue_policy_profile: str = PAPER_BLUE_POLICY_PROFILE,
             environment_profile: str = PAPER_ENVIRONMENT_PROFILE,
             initial_condition_randomization_mode: str = "deterministic_v1"):
    """Load a model, run one episode with TacView recording, save .acmi."""

    print(f"pid_throttle_base: {pid_throttle_base}", flush=True)
    print(f"action_distribution: {ACTION_DISTRIBUTION_VERSION}", flush=True)
    print(f"entropy_estimator: {ENTROPY_ESTIMATOR_VERSION}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_hidden_size = 128  # default; overridden by checkpoint auto-inference

    # ---- 1. 创建环境 ----
    print("创建环境...", flush=True)
    try:
        env = UavCombatEnv(max_num_blue=num_blue, max_num_red=num_red,
                           max_steps=max_steps,
                           obs_mode=obs_mode,
                           pid_profile=pid_profile,
                           pid_throttle_base=pid_throttle_base,
                           reward_mode=reward_mode,
                           missile_guidance_mode=missile_guidance_mode,
                           blue_policy_profile=blue_policy_profile,
                           environment_profile=environment_profile,
                           initial_condition_randomization_mode=(
                               initial_condition_randomization_mode),
                           suppress_jsbsim_output=True)
    except Exception:
        print("ERROR: 环境创建失败:", flush=True)
        traceback.print_exc()
        return

    # ---- 2. 加载模型 (可选) ----
    actor = None
    if checkpoint_path is not None:
        print(f"加载模型: {checkpoint_path} ...", flush=True)
        try:
            payload = torch.load(
                checkpoint_path, map_location=device, weights_only=False)
            obs_dim = _compute_obs_dim(
                num_red, num_blue, is_red=True, obs_mode=obs_mode)
            config = Config()
            config.num_red = num_red
            config.num_blue = num_blue
            config.max_episode_length = max_steps
            config.obs_mode = obs_mode
            config.obs_normalization = obs_normalization
            config.pid_profile = pid_profile
            config.pid_throttle_base = pid_throttle_base
            config.reward_mode = reward_mode
            config.missile_guidance_mode = missile_guidance_mode
            config.blue_policy_profile = blue_policy_profile
            config.environment_profile = environment_profile
            config.environment_version = environment_profile
            expected_metadata = _checkpoint_metadata(
                config, obs_dim,
                _compute_global_state_dim(num_red, obs_mode))
            state = _unpack_and_validate_checkpoint(
                payload, expected_metadata, "actor")

            # Auto-infer model architecture from checkpoint weights, so the eval
            # script works with any training config (different hidden sizes, etc.)
            ckpt_hidden = None
            ckpt_rnn_hidden = None
            ckpt_obs_dim = None
            for key, tensor in state.items():
                if key == "fc_in.weight":
                    ckpt_hidden = tensor.shape[0]
                    ckpt_obs_dim = tensor.shape[1]
                elif key == "rnn.weight_ih":
                    ckpt_rnn_hidden = tensor.shape[0] // 3  # GRU 3 gates
                if ckpt_hidden and ckpt_rnn_hidden and ckpt_obs_dim:
                    break

            if ckpt_obs_dim is None:
                print("ERROR: 无法从 checkpoint 推断 obs_dim", flush=True)
                env.close()
                return

            if ckpt_obs_dim != obs_dim:
                if obs_mode == "paper_strict":
                    total_agents = ckpt_obs_dim // 11
                else:
                    total_agents = (ckpt_obs_dim - 5) // 12
                print(f"ERROR: 维度不匹配！", flush=True)
                print(f"  Checkpoint obs_dim = {ckpt_obs_dim}  (训练时约 {total_agents} 个智能体总数)",
                      flush=True)
                print(f"  当前请求 obs_dim   = {obs_dim}  (--num-red={num_red} --num-blue={num_blue})",
                      flush=True)
                print(f"  请调整 --num-red 和 --num-blue，使总数 = {total_agents}",
                      flush=True)
                env.close()
                return

            hidden = ckpt_hidden or 128
            rnn_hidden_size = ckpt_rnn_hidden or 128

            actor = VanillaActor(obs_dim=obs_dim, action_dim=3,
                                 hidden=hidden, rnn_hidden=rnn_hidden_size).to(device)
            actor.load_state_dict(state)
            actor.eval()
            print(f"  模型已加载 (obs_dim={obs_dim}, hidden={hidden}, rnn={rnn_hidden_size})",
                  flush=True)
        except Exception:
            print("ERROR: 模型加载失败:", flush=True)
            traceback.print_exc()
            env.close()
            return
    else:
        print("未加载模型，红方使用随机策略", flush=True)

    # ---- 3. 开启 TacView 录制 ----
    print(f"开启录制 → {output_path} ...", flush=True)
    try:
        env.render(output_path)
    except Exception:
        print("ERROR: env.render() 失败:", flush=True)
        traceback.print_exc()
        env.close()
        return

    # ---- 4. Reset ----
    print("环境重置...", flush=True)
    try:
        obs, _ = env.reset()
        if draw_boundary and env._tacview_recorder is not None:
            env._tacview_recorder.append_lines(
                _battlefield_boundary_acmi_lines(boundary_half_size))
            print("  ACMI boundary debug markers enabled", flush=True)
        print("  重置完成", flush=True)
    except Exception:
        print("ERROR: env.reset() 失败:", flush=True)
        traceback.print_exc()
        env.close()
        return

    red_ids = [f"red_{i}" for i in range(num_red)]
    blue_ids = [f"blue_{i}" for i in range(num_blue)]

    # 红方 RNN 隐藏状态 (仅在使用模型时)
    rnn_a = np.zeros((num_red, rnn_hidden_size), dtype=np.float32)

    # ---- Visual missile tracking ----
    trackers: list[VisualMissileTracker] = []
    next_visual_id = 9001       # well above env's 1001–1999 range
    known_missile_uids: set[str] = set()

    # ---- Death reason tracking ----
    death_reasons: dict[str, str] = {}      # agent_id → reason (recorded on death)
    red_missiles_total = 0.0
    blue_missiles_total = 0.0
    red_episode_joint_reward = 0.0

    # ---- 5. 对战循环 ----
    step = 0
    done = False
    try:
        while not done:
            actions = {}

            # 蓝方协同目标分配；导弹规避由环境层脚本处理。
            blue_obs_dict = {bid: obs[bid] for bid in blue_ids}
            actions.update(env.blue_policy_actions(blue_obs_dict))

            # 红方：模型推理 / 随机
            if actor is not None:
                alive_indices = []
                obs_batch = []
                for i, rid in enumerate(red_ids):
                    obs_np = obs[rid]
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        obs_batch.append(_flatten_obs(
                            obs_np, obs_mode=obs_mode,
                            obs_normalization=obs_normalization))
                        alive_indices.append(i)
                    else:
                        actions[rid] = np.zeros(3, dtype=np.float32)

                if alive_indices:
                    obs_t = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32,
                                            device=device)
                    rnn_t = torch.as_tensor(rnn_a[alive_indices], device=device)
                    with torch.no_grad():
                        action_dist, new_rnn = actor(obs_t, rnn_t)
                        act = action_dist.mode
                    for k, i in enumerate(alive_indices):
                        actions[red_ids[i]] = act[k].cpu().numpy()
                        rnn_a[i] = new_rnn[k].cpu().numpy()
            else:
                for rid in red_ids:
                    actions[rid] = np.random.uniform(-1, 1, 3).astype(np.float32)

            # ---- Snapshot before step ----
            prev_missile_uids = set(env._missiles_in_flight.keys())

            # 环境步进
            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            red_episode_joint_reward += _joint_team_reward_once(rewards, red_ids)

            for rid in red_ids:
                red_missiles_total += info.get(rid, {}).get("missiles_fired_this_step", 0)
            for bid in blue_ids:
                blue_missiles_total += info.get(bid, {}).get("missiles_fired_this_step", 0)

            # ---- Record death reasons from this step ----
            for aid in red_ids + blue_ids:
                if aid not in death_reasons:
                    dr = info.get(aid, {}).get("death_reason")
                    if dr:
                        death_reasons[aid] = dr
                        team = "Red" if aid.startswith("red") else "Blue"
                        print(f"  [Step {step}] {aid} ({team}) died: {dr}", flush=True)

            # ---- Detect new real missiles → create visual trackers ----
            current_missile_uids = set(env._missiles_in_flight.keys())
            new_uids = current_missile_uids - known_missile_uids

            for uid in new_uids:
                # The missile may already be dead (hit within the same env step)
                # and removed from _missiles_in_flight, so check both the dict
                # AND the parent's launch_missiles list.
                real_missile = env._missiles_in_flight.get(uid)
                if real_missile is None:
                    # Missile launched and died within this step — search
                    # parent aircraft launch lists
                    for sim in env._all_sims():
                        for m in sim.launch_missiles:
                            if m.uid == uid:
                                real_missile = m
                                break
                        if real_missile is not None:
                            break
                if real_missile is None:
                    continue

                tracker = VisualMissileTracker(next_visual_id, real_missile)
                trackers.append(tracker)
                next_visual_id += 1

            known_missile_uids |= new_uids

            # ---- Inject visual missile lines into ACMI buffer ----
            if env._tacview_recorder is not None:
                for t in trackers:
                    lines = t.tick()
                    if lines:
                        env._tacview_recorder.append_lines(lines)

            # ---- Remove fully-expired trackers ----
            trackers = [t for t in trackers if not t.is_expired]

            # 重置死掉/truncated agent 的 RNN
            if actor is not None:
                for i, rid in enumerate(red_ids):
                    don = bool(terminated.get(rid, False) or truncated.get(rid, False))
                    if don:
                        rnn_a[i] = np.zeros(rnn_hidden_size, dtype=np.float32)

            # 整局结束判定
            dones = {}
            for aid in env.agent_ids:
                dones[aid] = bool(terminated.get(aid, False) or truncated.get(aid, False))
            if all(dones.values()):
                done = True

                blue_alive = sum(1 for bid in blue_ids
                                 if info.get(bid, {}).get("alive", False))
                red_alive = sum(1 for rid in red_ids
                                if info.get(rid, {}).get("alive", False))
                if blue_alive == 0 and red_alive > 0:
                    outcome = "红方胜!"
                elif red_alive == 0 and blue_alive > 0:
                    outcome = "蓝方胜!"
                else:
                    outcome = "平局 (超时)"
                print(f"Step {step}: {outcome}  "
                      f"(红存活={red_alive}/{num_red}, 蓝存活={blue_alive}/{num_blue})",
                      flush=True)

                # ---- Death summary ----
                red_deaths = Counter()
                blue_deaths = Counter()
                for aid, dr in death_reasons.items():
                    if aid.startswith("red"):
                        red_deaths[dr] += 1
                    else:
                        blue_deaths[dr] += 1

                def _fmt(c: Counter) -> str:
                    return "  ".join(f"{k}:{v}" for k, v in sorted(c.items()))

                print(f"  Deaths: Red[{_fmt(red_deaths)}] Blue[{_fmt(blue_deaths)}]",
                      flush=True)
                # ---- Missile termination summary ----
                mt = info.get("__missile_term__", {})
                if mt:
                    print(f"  Missile terminations:", flush=True)
                    for team in ("red", "blue"):
                        reasons = mt.get(team, {})
                        if reasons:
                            print(f"    {team}: {_fmt(Counter(reasons))}", flush=True)
                red_deaths_missile = sum(
                    v for k, v in red_deaths.items()
                    if _classify_death_reason(k) == "missile")
                blue_deaths_missile = sum(
                    v for k, v in blue_deaths.items()
                    if _classify_death_reason(k) == "missile")
                red_missile_hits = blue_deaths_missile
                blue_missile_hits = red_deaths_missile
                red_total_deaths = sum(red_deaths.values())
                blue_total_deaths = sum(blue_deaths.values())
                kd_all, _ = _ratio_with_denominator_zero(
                    blue_total_deaths, red_total_deaths)
                kd_missile, _ = _ratio_with_denominator_zero(
                    blue_deaths_missile, red_deaths_missile)
                outcome_key = _episode_outcome(red_alive, blue_alive)
                rwr_single, rwr_zero = _ratio_with_denominator_zero(
                    int(outcome_key == "red"), int(outcome_key == "blue"))

                print("  Paper metrics:", flush=True)
                print(f"    EpisodeRewardRed: {red_episode_joint_reward:.6f}",
                      flush=True)
                print(f"    RedAlive / BlueAlive: {red_alive} / {blue_alive}", flush=True)
                print(f"    Red missiles fired / Blue missiles fired: "
                      f"{red_missiles_total:.0f} / {blue_missiles_total:.0f}",
                      flush=True)
                print(f"    Red missile hits / Blue missile hits: "
                      f"{red_missile_hits} / {blue_missile_hits}", flush=True)
                print(f"    Red missile hit rate / Blue missile hit rate: "
                      f"{_safe_div(red_missile_hits, red_missiles_total):.6f} / "
                      f"{_safe_div(blue_missile_hits, blue_missiles_total):.6f}",
                      flush=True)
                print(f"    KD_Red_AllDeaths: {kd_all}", flush=True)
                print(f"    KD_Red_MissileOnly: {kd_missile}", flush=True)
                print(f"    RWR: {rwr_single} "
                      f"(denominator_zero={rwr_zero})", flush=True)
                if trackers:
                    print(f"  可视化导弹轨迹: {len(trackers)} 条", flush=True)
    except Exception:
        print(f"ERROR: 对战循环在 step {step} 崩溃:", flush=True)
        traceback.print_exc()
        try:
            n = env.save_acmi()
            print(f"  已抢救 {n} 帧 → {output_path}", flush=True)
        except Exception:
            pass
        env.close()
        return

    # ---- 6. 写入 .acmi 文件 ----
    try:
        n_frames = env.save_acmi()
        print(f"已写入 {n_frames} 帧 → {output_path}", flush=True)
        print("用 TacView 打开该文件即可观看回放。", flush=True)
    except Exception:
        print("ERROR: 写入 .acmi 文件失败:", flush=True)
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    # Diagnostic marker 5
    try:
        with open(_DIAG, "a") as f:
            f.write("main_block: entered\n")
    except Exception:
        pass
    try:
        parser = argparse.ArgumentParser(description="TacView ACMI 录像生成")
        parser.add_argument("--checkpoint", type=str, default=None,
                            help="Actor checkpoint 路径 (默认: best > final > 随机)")
        parser.add_argument("--random", action="store_true",
                            help="红方使用随机策略")
        parser.add_argument("--output", type=str, default="eval_battle.acmi",
                            help="输出 .acmi 文件路径")
        parser.add_argument("--num-red", type=int, default=3)
        parser.add_argument("--num-blue", type=int, default=3)
        parser.add_argument("--max-steps", type=int, default=1400)
        parser.add_argument("--obs-mode", type=str,
                            choices=("paper_strict",),
                            default="paper_strict")
        parser.add_argument("--obs-normalization",
                            choices=("paper_fixed_v1", "none"),
                            default="paper_fixed_v1")
        parser.add_argument("--pid-profile", choices=(PAPER_PID_PROFILE,),
                            default=PAPER_PID_PROFILE)
        parser.add_argument("--pid-throttle-base", type=float,
                            default=PID_THROTTLE_BASE)
        parser.add_argument("--reward-mode", choices=(PAPER_REWARD_MODE,),
                            default=PAPER_REWARD_MODE)
        parser.add_argument("--missile-guidance-mode",
                            choices=(PAPER_MISSILE_GUIDANCE_MODE,),
                            default=PAPER_MISSILE_GUIDANCE_MODE)
        parser.add_argument("--blue-policy-profile",
            choices=(PAPER_BLUE_POLICY_PROFILE,),
            default=PAPER_BLUE_POLICY_PROFILE)
        parser.add_argument("--environment-profile",
            choices=(PAPER_ENVIRONMENT_PROFILE,),
            default=PAPER_ENVIRONMENT_PROFILE)
        parser.add_argument("--initial-condition-randomization-mode",
                            choices=("deterministic_v1",),
                            default="deterministic_v1")
        parser.add_argument("--draw-boundary", action="store_true", default=False,
                            help="Draw battlefield boundary in ACMI for debugging.")
        parser.add_argument("--boundary-half-size", type=float, default=40000.0,
                            help="Half-size of square battlefield boundary in meters.")
        args = parser.parse_args()

        ckpt = None
        if args.random:
            ckpt = None
            print("红方使用随机策略", flush=True)
        elif args.checkpoint is not None:
            ckpt = args.checkpoint
        elif os.path.exists("checkpoints/vanilla_actor_best.pt"):
            ckpt = "checkpoints/vanilla_actor_best.pt"
            print("[Auto] 使用最佳模型: vanilla_actor_best.pt", flush=True)
        elif os.path.exists("checkpoints/vanilla_actor_final.pt"):
            ckpt = "checkpoints/vanilla_actor_final.pt"
            print("[Auto] 使用最终模型: vanilla_actor_final.pt (best.pt 未找到)", flush=True)
        else:
            print("[WARN] 未找到任何 checkpoint，回退至随机策略", flush=True)

        if ckpt is not None and not os.path.exists(ckpt):
            print(f"[WARN] checkpoint 不存在: {ckpt}", flush=True)
            ckpt = None

        run_acmi(checkpoint_path=ckpt, output_path=args.output,
                 num_red=args.num_red, num_blue=args.num_blue,
                 max_steps=args.max_steps,
                 draw_boundary=args.draw_boundary,
                 boundary_half_size=args.boundary_half_size,
                 obs_mode=args.obs_mode,
                 obs_normalization=args.obs_normalization,
                 pid_profile=args.pid_profile,
                 pid_throttle_base=args.pid_throttle_base,
                 reward_mode=args.reward_mode,
                 missile_guidance_mode=args.missile_guidance_mode,
                 blue_policy_profile=args.blue_policy_profile,
                 environment_profile=args.environment_profile,
                 initial_condition_randomization_mode=(
                     args.initial_condition_randomization_mode))
    except Exception:
        print("FATAL: 未捕获的异常:", flush=True)
        traceback.print_exc()
