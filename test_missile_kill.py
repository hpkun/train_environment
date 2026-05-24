"""
test_missile_kill.py —— 导弹发射 / 比例导引弹道 / 击杀判定 验证脚本

1v1 场景: 蓝方平飞当靶子，红方使用纯追踪逻辑追击。
预期结果: 红方逼近至 AO<45 且 距离<14km 后自动发射导弹，
         导弹沿 PN 弹道飞行，命中后蓝方被击落 (+200 奖励)，
         生成 missile_hit.acmi 供 TacView 回放验证。

用法:  python test_missile_kill.py
"""
from __future__ import annotations

import sys
import numpy as np

from my_uav_env import UavCombatEnv

# 解决 Windows GBK 终端编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def pure_pursuit_action(ego_obs: dict) -> np.ndarray:
    """纯追踪: 保持水平飞行，不对准航向。

    在 1v1 对头场景中，蓝方和红方初始航向天然指向对方 (AO ~ 0)。
    不做航向机动，仅保持水平飞行和巡航速度，依赖环境自动发射导弹。

    Args:
        ego_obs: 红方 agent 的观测 dict

    Returns:
        action: np.ndarray(3,) 归一化动作 (全零 = 水平巡航)
    """
    # 对头飞行: AO 天然接近 0, 无需机动
    return np.zeros(3, dtype=np.float32)


def _get_crash_reason(env, aid: str) -> str:
    """诊断 agent 被终止的原因。"""
    sim = env.blue_planes.get(aid) or env.red_planes.get(aid)
    if sim is None:
        return "not found"
    if sim.is_shotdown:
        return "shotdown (missile hit)"
    if hasattr(sim, "_crashed") and sim._crashed:
        alt = sim.get_geodetic()[2] if sim.is_alive else None
        return f"crashed (alt ~ {alt:.0f} m)" if alt else "crashed"
    if not sim.is_alive:
        return "not alive (reason unknown)"
    return "alive (not terminated)"


def run():
    # ==========================================================================
    #  1. 创建 1v1 环境 — 尾追场景（满足 TA > 90° 后半球约束）
    # ==========================================================================
    env = UavCombatEnv(
        max_num_blue=1, max_num_red=1,
        num_missiles_per_plane=2,
        max_steps=3000,
    )

    # 重写初始条件：Red 在 Blue 后方同向飞行（尾追几何）
    _original_make_init = env._make_init_state
    def _tail_chase_init(color: str, index: int) -> dict:
        state = _original_make_init(color, index)
        if color == "Blue":
            state["ic/long-gc-deg"] = 120.00       # Blue 在前
            state["ic/psi-true-deg"] = 90.0          # 向东飞
        else:  # Red
            state["ic/long-gc-deg"] = 119.93        # Red 在后
            state["ic/psi-true-deg"] = 90.0          # 同向飞行（尾追）
        return state
    env._make_init_state = _tail_chase_init

    MAX_STEPS = 3000

    print("=" * 60)
    print("  Missile Kill Chain Verification -- 1v1 Tail Chase")
    print("=" * 60)
    print(f"  Blue (Blue_0): target ahead, level flight east  (zero action)")
    print(f"  Red  (Red_0):  pursuer behind, level flight east (zero action)")
    print(f"  Launch window:   AO < 45 deg  AND  range < 14 km")
    print(f"                   TA > 90 deg  (rear-hemisphere)")
    print(f"  Kill radius:     300 m")
    print("-" * 60)

    # ==========================================================================
    #  2. 启用 TacView 录像
    # ==========================================================================
    env.render(filepath="missile_hit.acmi")

    # ==========================================================================
    #  3. 重置
    # ==========================================================================
    obs_dict, _ = env.reset()
    red_id = "red_0"
    blue_id = "blue_0"

    prev_missile_count = 0
    missile_launched = False
    kill_confirmed = False
    last_printed_step = -40  # 每 40 步打印状态

    # ==========================================================================
    #  4. 主循环
    # ==========================================================================
    for step in range(1, MAX_STEPS + 1):
        # ---- 动作 ----
        actions = {
            blue_id: np.zeros(3, dtype=np.float32),
        }

        red_obs = obs_dict[red_id]
        red_alive = not np.allclose(red_obs["ego_state"], 0.0)
        actions[red_id] = pure_pursuit_action(red_obs) if red_alive else np.zeros(3, dtype=np.float32)

        # ---- 步进 ----
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # ---- 指标 ----
        blue_ego = obs_dict[blue_id]["ego_state"]
        red_enemy_view = obs_dict[red_id]["enemy_states"]
        blue_alive = not np.allclose(blue_ego, 0.0)
        red_alive = not np.allclose(obs_dict[red_id]["ego_state"], 0.0)
        blue_sim = env.blue_planes.get(blue_id)
        red_sim = env.red_planes.get(red_id)

        ao_rad = float(red_enemy_view[0, 3])
        r_m   = float(red_enemy_view[0, 5])
        du_m  = float(red_enemy_view[0, 2])
        ao_deg = np.rad2deg(ao_rad)

        current_missiles = len(env._missiles_in_flight)

        # ---- 导弹发射检测 ----
        if not missile_launched and current_missiles > prev_missile_count:
            missile_launched = True
            print(f"\n  *** MISSILE LAUNCHED! (step {step})")
            print(f"      Range at launch: {r_m/1000:.1f} km")
            print(f"      AO at launch:    {ao_deg:.1f} deg")
            print(f"      Red speed:       {np.linalg.norm(red_sim.get_velocity()):.0f} m/s" if red_sim else "")
            print(f"      Blue speed:      {np.linalg.norm(blue_sim.get_velocity()):.0f} m/s" if blue_sim else "")

        # ---- 击杀检测 ----
        if not kill_confirmed and not blue_alive:
            kill_confirmed = True
            # 确认是导弹击杀
            rw_red = float(rewards[red_id])
            rw_blue = float(rewards[blue_id])
            print(f"\n  *** TARGET DESTROYED! (step {step})")
            print(f"      Red  reward: {rw_red:+.1f}  (win-lose: +30 × survivors)")
            print(f"      Blue reward: {rw_blue:+.1f}")
            print(f"      Missiles in flight: {current_missiles}")
            for mid, missile in env._missiles_in_flight.items():
                print(f"        {mid}: status={'HIT' if missile.is_success else 'MISS' if missile.is_done else 'FLYING'}")
            if blue_sim:
                print(f"      Blue shotdown: {blue_sim.is_shotdown}")

        prev_missile_count = current_missiles

        # ---- 定期状态 ----
        if step - last_printed_step >= 40 and blue_alive and not kill_confirmed:
            r_speed = np.linalg.norm(red_sim.get_velocity()) if red_sim else 0
            b_speed = np.linalg.norm(blue_sim.get_velocity()) if blue_sim else 0
            print(f"  [step {step:5d}]  range={r_m/1000:6.1f} km  AO={ao_deg:+5.1f} deg  "
                  f"dAlt={du_m:+6.0f} m  msles={current_missiles}  "
                  f"Vr={r_speed:.0f} Vb={b_speed:.0f}  "
                  f"R={red_alive} B={blue_alive}")
            last_printed_step = step

        # ---- 终止 ----
        any_done = any(terminated.values()) or any(truncated.values())
        if any_done:
            if kill_confirmed:
                print(f"\n  Episode terminated (step {step}) -- Blue shot down.")
            else:
                print(f"\n  Episode terminated early (step {step}).")
                for aid, term in terminated.items():
                    if term:
                        reason = _get_crash_reason(env, aid)
                        print(f"    {aid}: {reason}")
                        sim = env.blue_planes.get(aid) or env.red_planes.get(aid)
                        if sim:
                            alt = sim.get_geodetic()[2] if hasattr(sim, "get_geodetic") else "?"
                            vel = np.linalg.norm(sim.get_velocity())
                            print(f"         alt={alt}, vel={vel:.0f} m/s")
                for aid, trunc in truncated.items():
                    if trunc:
                        print(f"    {aid}: time limit")
            break
    else:
        print(f"\n  WARNING: max steps ({MAX_STEPS}) exhausted without kill.")
        print(f"    Final range: {r_m/1000:.1f} km, AO: {ao_deg:.1f} deg")

    # ==========================================================================
    #  5. 保存录像
    # ==========================================================================
    n_frames = env.save_acmi()
    print(f"\n  ACMI saved: missile_hit.acmi  ({n_frames} frames)")
    print("=" * 60)

    if kill_confirmed:
        print("  [PASS] Missile kill chain verified!")
        print("         Open missile_hit.acmi in TacView to review the trajectory.")
    else:
        print("  [WARN] Kill not achieved this run.")
        print("         Check the diagnostics above for failure reason.")

    env.close()


if __name__ == "__main__":
    run()
