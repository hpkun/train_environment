"""BRMA-TAM Role-Situation v3 reward computation (contract revision 5).

Design combines:
- TAM-HAPPO: heterogeneous MAV/UAV roles, MAV safety/support, UAV flight/situation
- BRMA-MAPPO: multi-entity offensive/defensive situation, softmax aggregation
- Current JSBSim: scripted launch, MAV shared observation, alive-before team mean
"""
from __future__ import annotations

import math
import numpy as np


def _softmax_agg(values, tau: float = 0.2):
    """Numerically stable softmax-weighted aggregation. Returns 0 for empty."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    shifted = arr / max(float(tau), 1e-9)
    shifted -= shifted.max()  # numerical stability
    exps = np.exp(shifted)
    weights = exps / exps.sum()
    return float(np.sum(weights * arr))


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def compute_v3_reward(env, base_rewards, components):
    """Main v3 reward computation called from HeteroUavCombatEnv._compute_rewards."""
    cfg = env.brma_tam_role_situation_v3_config
    task_cfg = cfg["task"]
    sit_cfg = cfg["situation"]
    mav_cfg = cfg["mav"]
    uav_cfg = cfg["uav"]
    flight_cfg = cfg["flight"]

    # ── Identify alive entities ──
    alive_blue = {bid: env.blue_planes[bid] for bid in env.blue_ids
                  if env.blue_planes.get(bid) and env.blue_planes[bid].is_alive}
    alive_red = {rid: env.red_planes[rid] for rid in env.red_ids
                 if env.red_planes.get(rid) and env.red_planes[rid].is_alive}
    mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), None)
    mav = env.red_planes.get(mav_id) if mav_id else None
    attack_ids = [rid for rid in env.red_ids if env.agent_roles.get(rid) == "attack_uav"]
    alive_attack = {rid: env.red_planes[rid] for rid in attack_ids
                    if env.red_planes.get(rid) and env.red_planes[rid].is_alive}

    n_blue_alive = len(alive_blue)
    n_red_alive = len(alive_red)
    n_attack_alive = len(alive_attack)
    n_blue_initial = len(env.blue_ids)
    n_attack_initial = len(attack_ids)
    round_over = n_blue_alive == 0 or n_red_alive == 0 or env.current_step >= env.max_steps

    # ── Alive-before tracking ──
    alive_before = getattr(env, "_v3_alive_before", {})
    mav_alive_initial = bool(mav and mav.is_alive)

    # ── Team task: attrition ──
    _v3_state = getattr(env, "_v3_episode_state", None)
    if _v3_state is None:
        env._v3_episode_state = {"prev_blue_dead": 0, "prev_attack_dead": 0, "prev_mav_dead": 0,
                                  "terminal_applied": False, "last_losses": (0.0, 0.0, 0)}
        _v3_state = env._v3_episode_state

    blue_dead_now = n_blue_initial - n_blue_alive
    attack_dead_now = n_attack_initial - n_attack_alive
    mav_dead_now = 0 if (mav and mav.is_alive) else 1

    delta_blue = (blue_dead_now - _v3_state["prev_blue_dead"]) / max(n_blue_initial, 1)
    delta_attack = (attack_dead_now - _v3_state["prev_attack_dead"]) / max(n_attack_initial, 1)
    delta_mav = float(mav_dead_now - _v3_state["prev_mav_dead"])

    lambda_u = float(task_cfg.get("attack_uav_loss_weight", 1.0))
    lambda_m = float(task_cfg.get("mav_loss_weight", 0.75))
    delta_j = delta_blue - lambda_u * delta_attack - lambda_m * delta_mav
    attrition_scale = float(task_cfg.get("attrition_scale", 10.0))
    r_attrition = attrition_scale * delta_j

    _v3_state["prev_blue_dead"] = blue_dead_now
    _v3_state["prev_attack_dead"] = attack_dead_now
    _v3_state["prev_mav_dead"] = mav_dead_now

    # ── Team task: terminal ──
    r_terminal = 0.0
    terminal_applied = 0.0
    if round_over and not _v3_state["terminal_applied"]:
        if n_blue_alive == 0 and n_red_alive > 0:
            r_terminal = float(task_cfg.get("decisive_win_bonus", 10.0))
        elif n_red_alive == 0 and n_blue_alive > 0:
            r_terminal = float(task_cfg.get("decisive_loss_penalty", -10.0))
        else:
            j_combat = (blue_dead_now / max(n_blue_initial, 1)
                        - lambda_u * attack_dead_now / max(n_attack_initial, 1)
                        - lambda_m * mav_dead_now)
            if j_combat > 1e-9:
                r_terminal = float(task_cfg.get("timeout_advantage_bonus", 2.0))
            elif j_combat < -1e-9:
                r_terminal = -float(task_cfg.get("timeout_advantage_bonus", 2.0))
        _v3_state["terminal_applied"] = True
        terminal_applied = 1.0

    r_common = r_attrition + r_terminal

    # ── Launch range reference ──
    R_launch = getattr(env, "_missile_launch_range_m_effective", env.MISSILE_LAUNCH_RANGE_THRESH)
    R_min = getattr(env, "_missile_launch_min_range_m_effective", env.MISSILE_LAUNCH_MIN_RANGE)

    # ── UAV situation: multi-entity per-UAV per-blue ──
    D_low = float(sit_cfg.get("distance_optimal_low_ratio", 0.35)) * R_launch
    D_high = float(sit_cfg.get("distance_optimal_high_ratio", 0.75)) * R_launch
    tau = float(sit_cfg.get("softmax_temperature", 0.2))
    threat_w = float(sit_cfg.get("threat_weight", 1.0))
    local_w = float(sit_cfg.get("local_weight", 0.6))
    team_w = float(sit_cfg.get("team_weight", 0.4))
    spd_min = float(sit_cfg.get("speed_modulation_min", 0.75))
    spd_max = float(sit_cfg.get("speed_modulation_max", 1.0))

    # Precompute O_ij, T_ji per UAV-blue pair
    O_matrix = {}  # (uav_id, blue_id) -> offense quality
    T_matrix = {}  # (uav_id, blue_id) -> threat from blue to UAV
    blue_ids_list = list(alive_blue.keys())

    for uid, uav in alive_attack.items():
        for bid, blue in alive_blue.items():
            geom = env._brma_tam_3d_geometry(uav, blue)
            ata = float(geom.get("tam_ata_rad", np.pi))
            aa = float(geom.get("tam_aa_rad", 0.0))
            d = float(geom.get("target_distance_m", float("inf")))
            # Angle quality (offense view: UAV -> blue)
            q_angle_o = _clip01(1.0 - (ata + aa) / np.pi) if np.isfinite(ata + aa) else 0.0
            # Angle quality (threat view: blue -> UAV) — re-compute from blue perspective
            geom_rev = env._brma_tam_3d_geometry(blue, uav)
            ata_r = float(geom_rev.get("tam_ata_rad", np.pi))
            aa_r = float(geom_rev.get("tam_aa_rad", 0.0))
            q_angle_t = _clip01(1.0 - (ata_r + aa_r) / np.pi) if np.isfinite(ata_r + aa_r) else 0.0
            # Distance quality
            if d < R_min: q_dist = 0.0
            elif d < D_low: q_dist = (d - R_min) / max(D_low - R_min, 1e-9)
            elif d <= D_high: q_dist = 1.0
            else: q_dist = float(np.exp(-(d - D_high) / max(R_launch, 1e-9)))
            q_dist = _clip01(q_dist)
            # Speed modulation
            vr = float(np.linalg.norm(env._brma_tam_safe_vec(uav, "get_velocity")))
            vb = float(np.linalg.norm(env._brma_tam_safe_vec(blue, "get_velocity")))
            if vr > 1e-8 and np.isfinite(vr) and np.isfinite(vb):
                ratio = vb / vr
                if vb < 0.5 * vr: sr = 1.0
                elif vb <= 1.5 * vr: sr = 2.0 - 2.0 * ratio
                else: sr = -1.0
            else: sr = 0.0
            q_speed = _clip01((sr + 1.0) / 2.0)
            m_speed = spd_min + (spd_max - spd_min) * q_speed
            O_ij = q_angle_o * q_dist * m_speed
            T_ji = q_angle_t * q_dist * m_speed  # same distance, speed, reversed angles
            O_matrix[(uid, bid)] = O_ij
            T_matrix[(uid, bid)] = T_ji

    # Per-UAV local situation
    uav_local_offense = {}
    uav_local_threat = {}
    uav_situation = {}
    for uid in alive_attack:
        off_vals = [O_matrix.get((uid, bid), 0.0) for bid in blue_ids_list]
        thr_vals = [T_matrix.get((uid, bid), 0.0) for bid in blue_ids_list]
        loc_off = _softmax_agg(off_vals, tau)
        loc_thr = _softmax_agg(thr_vals, tau)
        uav_local_offense[uid] = loc_off
        uav_local_threat[uid] = loc_thr
        loc_sit = loc_off - threat_w * loc_thr
        # Team coverage/exposure
        team_cover = float(np.mean([max(O_matrix.get((auid, bid), 0.0) for auid in alive_attack)
                                    for bid in blue_ids_list])) if blue_ids_list and alive_attack else 0.0
        team_expose = float(np.mean([max(T_matrix.get((uid2, bid), 0.0) for bid in blue_ids_list)
                                     for uid2 in alive_attack])) if alive_attack and blue_ids_list else 0.0
        sit = local_w * loc_sit + team_w * (team_cover - team_expose)
        uav_situation[uid] = _clip01(float(np.clip(sit, -1.0, 1.0)))

    # ── MAV role ──
    S_mav = 0.0
    I_marginal = 0.0
    P_support = 0.0
    T_mav = 0.0
    mav_sit_logs = {}
    if mav and mav.is_alive and alive_blue and alive_attack:
        # Marginal information
        marginal_sum = 0.0
        marginal_pairs = 0
        for uid in alive_attack:
            for bid in blue_ids_list:
                ts = env._mav_shared_track_state(uid, bid)
                shared = bool(ts.get("mav_shared_visible", False))
                direct = bool(ts.get("direct_visible", False))
                if shared and not direct:
                    marginal_pairs += 1
                    W = max(O_matrix.get((uid, bid), 0.0), T_matrix.get((uid, bid), 0.0))
                    marginal_sum += W
        I_marginal = marginal_sum / max(marginal_pairs, 1) if marginal_pairs > 0 else 0.0
        # Support position
        uav_positions = [env._brma_tam_safe_vec(alive_attack[uid], "get_position") for uid in alive_attack]
        blue_positions = [env._brma_tam_safe_vec(alive_blue[bid], "get_position") for bid in blue_ids_list]
        C_U = np.mean(uav_positions, axis=0)
        C_B = np.mean(blue_positions, axis=0)
        P_M = env._brma_tam_safe_vec(mav, "get_position")
        e_rear = C_U - C_B
        e_rear_norm = float(np.linalg.norm(e_rear))
        if e_rear_norm > 1e-8:
            e_rear = e_rear / e_rear_norm
            rear_proj = float(np.dot(P_M - C_U, e_rear))
        else:
            rear_proj = 0.0
        mav_dist = float(np.linalg.norm(P_M - C_U))
        s_min = float(mav_cfg.get("support_min_distance_ratio", 0.5)) * R_launch
        s_max = float(mav_cfg.get("support_max_distance_ratio", 1.5)) * R_launch
        if mav_dist < s_min: dq = mav_dist / max(s_min, 1e-9)
        elif mav_dist <= s_max: dq = 1.0
        else: dq = max(0.0, 1.0 - (mav_dist - s_max) / max(R_launch, 1e-9))
        rear_ref = float(mav_cfg.get("rear_reference_ratio", 0.75)) * R_launch
        rq = _clip01(rear_proj / max(rear_ref, 1e-9))
        P_support = dq * rq
        # MAV threat
        mav_threats = []
        for bid in blue_ids_list:
            blue = alive_blue[bid]
            geom_rev = env._brma_tam_3d_geometry(blue, mav)
            ata = float(geom_rev.get("tam_ata_rad", np.pi))
            aa = float(geom_rev.get("tam_aa_rad", 0.0))
            d = float(geom_rev.get("target_distance_m", float("inf")))
            qa = _clip01(1.0 - (ata + aa) / np.pi) if np.isfinite(ata + aa) else 0.0
            if d < R_min: qd = 0.0
            elif d < D_low: qd = (d - R_min) / max(D_low - R_min, 1e-9)
            elif d <= D_high: qd = 1.0
            else: qd = float(np.exp(-(d - D_high) / max(R_launch, 1e-9)))
            mav_threats.append(qa * _clip01(qd))
        geom_threat = _softmax_agg(mav_threats, tau)
        mw = 1.0 if (hasattr(mav, "check_missile_warning") and mav.check_missile_warning() is not None) else 0.0
        T_mav = _clip01(0.7 * geom_threat + 0.3 * mw)
        # MAV role
        miw = float(mav_cfg.get("marginal_information_weight", 0.5))
        spw = float(mav_cfg.get("support_position_weight", 0.3))
        mtw = float(mav_cfg.get("threat_weight", 0.4))
        S_mav = _clip01(float(np.clip(miw * I_marginal + spw * P_support - mtw * T_mav, -1.0, 1.0)))
        mav_sit_logs = {
            "role_situation_v3_mav_marginal_information": I_marginal,
            "role_situation_v3_mav_support_distance": dq,
            "role_situation_v3_mav_support_rear": rq,
            "role_situation_v3_mav_support_position": P_support,
            "role_situation_v3_mav_geometric_threat": geom_threat,
            "role_situation_v3_mav_missile_warning": mw,
            "role_situation_v3_mav_threat": T_mav,
        }

    # ── Flight ──
    mav_flight_scale = float(flight_cfg.get("mav_scale", 0.01))
    uav_flight_scale = float(flight_cfg.get("uav_scale", 0.01))
    mav_role_scale = float(mav_cfg.get("role_scale", 0.05))
    uav_sit_scale = float(uav_cfg.get("situation_scale", 0.05))

    n_R = max(n_red_alive, 1)
    n_U = max(n_attack_alive, 1)

    # ── Per-agent reward assembly ──
    for rid in env.red_ids:
        comp = components.setdefault(rid, {})
        sim = env.red_planes.get(rid)
        role = env.agent_roles.get(rid, "")
        ab = bool(alive_before.get(rid, getattr(sim, "is_alive", False)))
        pitch = float(comp.get("r_pitch", 0.0))
        roll_v = float(comp.get("r_roll", 0.0))
        vel = float(comp.get("r_vel", 0.0))
        flight_raw = pitch + roll_v + vel
        flight_norm = _clip01(float(np.clip(flight_raw, -1.0, 1.0)))
        vals = {
            "reward_contract_revision": 5.0,
            "role_situation_v3_task_attrition": r_attrition if ab else 0.0,
            "role_situation_v3_task_terminal": r_terminal if ab else 0.0,
            "role_situation_v3_common": r_common if ab else 0.0,
            "role_situation_v3_blue_loss_delta": delta_blue,
            "role_situation_v3_uav_loss_delta": delta_attack,
            "role_situation_v3_mav_loss_delta": delta_mav,
            "role_situation_v3_j_combat": delta_j,
            "role_situation_v3_flight_raw": flight_raw if ab else 0.0,
            "role_situation_v3_flight_norm": flight_norm if ab else 0.0,
            "role_situation_v3_alive_red_count": float(n_red_alive),
            "role_situation_v3_alive_uav_count": float(n_attack_alive),
            "role_situation_v3_alive_blue_count": float(n_blue_alive),
        }
        if not ab:
            for k in ("role_situation_v3_task_attrition", "role_situation_v3_task_terminal",
                       "role_situation_v3_common", "role_situation_v3_flight_raw",
                       "role_situation_v3_flight_norm"):
                vals[k] = 0.0
            vals.update({k: 0.0 for k in [
                "role_situation_v3_uav_local_offense", "role_situation_v3_uav_local_threat",
                "role_situation_v3_team_coverage", "role_situation_v3_team_exposure",
                "role_situation_v3_uav_situation", "role_situation_v3_mav_role",
                "role_situation_v3_role_encoded", "role_situation_v3_total",
                "role_situation_v3_identity_error",
            ]})
            total = 0.0
        elif role == "mav":
            flight_contrib = n_R * mav_flight_scale * flight_norm
            role_contrib = n_R * mav_role_scale * S_mav
            total = r_common + flight_contrib + role_contrib
            for k in mav_sit_logs:
                vals[k] = mav_sit_logs[k]
            vals.update({
                "role_situation_v3_uav_local_offense": 0.0,
                "role_situation_v3_uav_local_threat": 0.0,
                "role_situation_v3_team_coverage": 0.0,
                "role_situation_v3_team_exposure": 0.0,
                "role_situation_v3_uav_situation": 0.0,
                "role_situation_v3_mav_role": role_contrib / max(n_R, 1),
                "role_situation_v3_role_encoded": 1.0,
            })
        else:
            flight_contrib = n_R / max(n_U, 1) * uav_flight_scale * flight_norm
            sit_contrib = n_R / max(n_U, 1) * uav_sit_scale * uav_situation.get(rid, 0.0)
            total = r_common + flight_contrib + sit_contrib
            for k in mav_sit_logs:
                vals[k] = 0.0
            vals.update({
                "role_situation_v3_uav_local_offense": uav_local_offense.get(rid, 0.0),
                "role_situation_v3_uav_local_threat": uav_local_threat.get(rid, 0.0),
                "role_situation_v3_team_coverage": team_cover if alive_attack and blue_ids_list else 0.0,
                "role_situation_v3_team_exposure": team_expose if alive_attack and blue_ids_list else 0.0,
                "role_situation_v3_uav_situation": sit_contrib / max(n_R / max(n_U, 1), 1e-9),
                "role_situation_v3_mav_role": 0.0,
                "role_situation_v3_role_encoded": 0.0,
            })
        vals["role_situation_v3_total"] = total
        vals["role_situation_v3_identity_error"] = 0.0
        comp.update(vals)
        comp["total"] = float(total)
        base_rewards[rid] = float(total)

    return base_rewards, components
