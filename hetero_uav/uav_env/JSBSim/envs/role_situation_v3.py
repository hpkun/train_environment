"""BRMA-TAM Role-Situation v3 reward computation (contract revision 5).

Design combines:
- TAM-HAPPO: heterogeneous MAV/UAV roles, MAV safety/support, UAV flight/situation
- BRMA-MAPPO: multi-entity offensive/defensive situation, softmax aggregation
- Current JSBSim: scripted launch, MAV shared observation, alive-before team mean
"""
from __future__ import annotations

import math
import numpy as np

# ── Unified field constants (used by reward, logging, tests) -------------------
V3_REWARD_COMPONENT_FIELDS = (
    "reward_contract_revision",
    "role_situation_v3_task_attrition", "role_situation_v3_task_terminal",
    "role_situation_v3_common", "role_situation_v3_delta_j", "role_situation_v3_j_combat",
    "role_situation_v3_blue_loss_fraction", "role_situation_v3_uav_loss_fraction",
    "role_situation_v3_mav_loss_indicator", "role_situation_v3_blue_loss_delta",
    "role_situation_v3_uav_loss_delta", "role_situation_v3_mav_loss_delta",
    "role_situation_v3_uav_local_offense_raw", "role_situation_v3_uav_local_threat_raw",
    "role_situation_v3_team_coverage_raw", "role_situation_v3_team_exposure_raw",
    "role_situation_v3_uav_situation_raw", "role_situation_v3_uav_situation_scaled",
    "role_situation_v3_uav_situation_encoded",
    "role_situation_v3_mav_marginal_information_raw", "role_situation_v3_mav_support_distance_raw",
    "role_situation_v3_mav_support_rear_raw", "role_situation_v3_mav_support_position_raw",
    "role_situation_v3_mav_geometric_threat_raw", "role_situation_v3_mav_missile_warning",
    "role_situation_v3_mav_threat_raw", "role_situation_v3_mav_role_raw",
    "role_situation_v3_mav_role_scaled", "role_situation_v3_mav_role_encoded",
    "role_situation_v3_flight_raw", "role_situation_v3_flight_norm", "role_situation_v3_flight_scaled",
    "role_situation_v3_flight_encoded", "role_situation_v3_role_encoded",
    "role_situation_v3_total", "role_situation_v3_component_sum", "role_situation_v3_identity_error",
    "role_situation_v3_alive_red_before_count", "role_situation_v3_alive_uav_before_count",
    "role_situation_v3_alive_blue_after_count", "role_situation_v3_is_mav", "role_situation_v3_is_attack_uav",
)

V3_EFFECTIVE_FIELDS = (
    "effective_role_situation_v3_task_attrition", "effective_role_situation_v3_task_terminal",
    "effective_role_situation_v3_common", "effective_role_situation_v3_delta_j",
    "effective_role_situation_v3_j_combat",
    "effective_role_situation_v3_uav_local_offense", "effective_role_situation_v3_uav_local_threat",
    "effective_role_situation_v3_team_coverage", "effective_role_situation_v3_team_exposure",
    "effective_role_situation_v3_uav_situation_raw", "effective_role_situation_v3_uav_situation_scaled",
    "effective_role_situation_v3_uav_situation_encoded",
    "effective_role_situation_v3_mav_marginal_information", "effective_role_situation_v3_mav_support_position",
    "effective_role_situation_v3_mav_threat",
    "effective_role_situation_v3_mav_role_raw", "effective_role_situation_v3_mav_role_scaled",
    "effective_role_situation_v3_mav_role_encoded",
    "effective_role_situation_v3_uav_flight_encoded", "effective_role_situation_v3_mav_flight_encoded",
    "effective_role_situation_v3_role_encoded",
    "effective_role_situation_v3_total", "effective_role_situation_v3_component_sum",
    "effective_role_situation_v3_identity_error",
)

V3_EPISODE_FIELDS = (
    "episode_role_situation_v3_task_attrition_sum", "episode_role_situation_v3_task_terminal_sum",
    "episode_role_situation_v3_common_sum",
    "episode_role_situation_v3_uav_situation_encoded_sum", "episode_role_situation_v3_mav_role_encoded_sum",
    "episode_role_situation_v3_flight_encoded_sum", "episode_role_situation_v3_total_sum",
    "episode_role_situation_v3_final_j_combat", "episode_role_situation_v3_max_abs_identity_error",
)

# ── Clipping helpers -----------------------------------------------------------
def _clip_unit(x):
    return float(np.clip(x, 0.0, 1.0))

def _clip_signed(x):
    return float(np.clip(x, -1.0, 1.0))


# ── SoftMax aggregation --------------------------------------------------------
def _softmax_agg(values, tau: float = 0.2):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    shifted = arr / max(float(tau), 1e-9)
    shifted -= shifted.max()
    exps = np.exp(shifted)
    weights = exps / max(exps.sum(), 1e-300)
    return float(np.sum(weights * arr))


# ── Pair quality (shared for offense and threat) -------------------------------
def _compute_pair_quality(attacker, target, env,
                           R_min, D_low, D_high, R_launch,
                           spd_min, spd_max):
    """Compute (angle_q, dist_q, speed_q, modulation, combined, distance_m)
    for attacker -> target.  All quality values in [0,1].
    """
    geom = env._brma_tam_3d_geometry(attacker, target)
    ata = float(geom.get("tam_ata_rad", np.pi))
    aa = float(geom.get("tam_aa_rad", 0.0))
    d = float(geom.get("target_distance_m", float("inf")))

    angle_q = _clip_unit(1.0 - (ata + aa) / np.pi) if np.isfinite(ata + aa) else 0.0

    if not np.isfinite(d) or d < R_min:
        dist_q = 0.0
    elif d < D_low:
        dist_q = (d - R_min) / max(D_low - R_min, 1e-9)
    elif d <= D_high:
        dist_q = 1.0
    else:
        dist_q = float(np.exp(-(d - D_high) / max(R_launch, 1e-9)))
    dist_q = _clip_unit(dist_q)

    a_speed = float(np.linalg.norm(env._brma_tam_safe_vec(attacker, "get_velocity")))
    t_speed = float(np.linalg.norm(env._brma_tam_safe_vec(target, "get_velocity")))
    if a_speed > 1e-8 and np.isfinite(a_speed) and np.isfinite(t_speed):
        ratio = t_speed / a_speed
        if t_speed < 0.5 * a_speed:       sr = 1.0
        elif t_speed <= 1.5 * a_speed:     sr = 2.0 - 2.0 * ratio
        else:                              sr = -1.0
    else:
        sr = 0.0
    speed_q = _clip_unit((sr + 1.0) / 2.0)
    modulation = spd_min + (spd_max - spd_min) * speed_q

    combined = _clip_unit(angle_q * dist_q * modulation)
    return {
        "angle_q": angle_q, "dist_q": dist_q, "speed_q": speed_q,
        "modulation": modulation, "combined": combined, "distance_m": d,
    }


# ── Main v3 reward computation -------------------------------------------------
def compute_v3_reward(env, base_rewards, components):
    """Main v3 reward computation called from HeteroUavCombatEnv._compute_rewards."""
    cfg = env.brma_tam_role_situation_v3_config
    task_cfg = cfg["task"]
    sit_cfg = cfg["situation"]
    mav_cfg = cfg["mav"]
    uav_cfg = cfg["uav"]
    flight_cfg = cfg["flight"]

    # ── Identify entities (post-step status) ──
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

    # ── Alive-before counts ──
    ab = getattr(env, "_v3_alive_before", {})
    alive_red_before = [rid for rid in env.red_ids if ab.get(rid, False)]
    n_R_before = len(alive_red_before)
    n_U_before = len([rid for rid in attack_ids if ab.get(rid, False)])

    # ── Episode state ──
    es = getattr(env, "_v3_episode_state", None)
    if es is None:
        raise RuntimeError("v3 episode state not initialized; call _reset_role_situation_v3_episode_state()")

    blue_dead_now = n_blue_initial - n_blue_alive
    attack_dead_now = n_attack_initial - n_attack_alive
    mav_dead_now = 0 if (mav and mav.is_alive) else 1

    # ── Team task: loss fractions ──
    L_blue = blue_dead_now / max(n_blue_initial, 1)
    L_attack = attack_dead_now / max(n_attack_initial, 1)
    L_mav = float(mav_dead_now)

    lambda_u = float(task_cfg.get("attack_uav_loss_weight", 1.0))
    lambda_m = float(task_cfg.get("mav_loss_weight", 0.75))
    J_combat = L_blue - lambda_u * L_attack - lambda_m * L_mav

    delta_blue = L_blue - es["prev_blue_loss"]
    delta_attack = L_attack - es["prev_attack_loss"]
    delta_mav = float(mav_dead_now - es["prev_mav_dead"])
    delta_J = delta_blue - lambda_u * delta_attack - lambda_m * delta_mav

    attrition_scale = float(task_cfg.get("attrition_scale", 10.0))
    r_attrition = attrition_scale * delta_J

    # Update episode cumulative state
    es["prev_blue_dead"] = blue_dead_now
    es["prev_attack_dead"] = attack_dead_now
    es["prev_mav_dead"] = mav_dead_now
    es["prev_blue_loss"] = L_blue
    es["prev_attack_loss"] = L_attack
    es["latest_j_combat"] = J_combat

    # ── Team task: terminal ──
    r_terminal = 0.0
    terminal_applied = 0.0
    if round_over and not es["terminal_applied"]:
        if n_blue_alive == 0 and n_red_alive == 0:
            r_terminal = 0.0  # mutual elimination
        elif n_blue_alive == 0 and n_red_alive > 0:
            r_terminal = float(task_cfg.get("decisive_win_bonus", 10.0))
        elif n_red_alive == 0 and n_blue_alive > 0:
            r_terminal = float(task_cfg.get("decisive_loss_penalty", -10.0))
        elif env.current_step >= env.max_steps:
            if J_combat > 1e-9:
                r_terminal = float(task_cfg.get("timeout_advantage_bonus", 2.0))
            elif J_combat < -1e-9:
                r_terminal = -float(task_cfg.get("timeout_advantage_bonus", 2.0))
            else:
                r_terminal = 0.0
        es["terminal_applied"] = True
        terminal_applied = 1.0

    r_common = r_attrition + r_terminal

    # ── Launch range reference ──
    R_launch = getattr(env, "_missile_launch_range_m_effective", env.MISSILE_LAUNCH_RANGE_THRESH)
    R_min = getattr(env, "_missile_launch_min_range_m_effective", env.MISSILE_LAUNCH_MIN_RANGE)

    # ── Quality parameters ──
    D_low = float(sit_cfg.get("distance_optimal_low_ratio", 0.35)) * R_launch
    D_high = float(sit_cfg.get("distance_optimal_high_ratio", 0.75)) * R_launch
    tau_v = float(sit_cfg.get("softmax_temperature", 0.2))
    threat_w = float(sit_cfg.get("threat_weight", 1.0))
    local_w = float(sit_cfg.get("local_weight", 0.6))
    team_w = float(sit_cfg.get("team_weight", 0.4))
    spd_min = float(sit_cfg.get("speed_modulation_min", 0.75))
    spd_max = float(sit_cfg.get("speed_modulation_max", 1.0))

    blue_ids_list = list(alive_blue.keys())

    # ── Pair quality matrices ──
    O_ij_dict = {}; T_ji_dict = {}
    for uid, uav in alive_attack.items():
        for bid, blue in alive_blue.items():
            pq_off = _compute_pair_quality(uav, blue, env, R_min, D_low, D_high, R_launch, spd_min, spd_max)
            O_ij_dict[(uid, bid)] = pq_off
            pq_thr = _compute_pair_quality(blue, uav, env, R_min, D_low, D_high, R_launch, spd_min, spd_max)
            T_ji_dict[(uid, bid)] = pq_thr

    # ── UAV situation ──
    uav_local_off = {}; uav_local_thr = {}; uav_sit_raw = {}
    for uid in alive_attack:
        off_vals = [O_ij_dict.get((uid, bid), {"combined": 0.0})["combined"] for bid in blue_ids_list]
        thr_vals = [T_ji_dict.get((uid, bid), {"combined": 0.0})["combined"] for bid in blue_ids_list]
        loc_off = _softmax_agg(off_vals, tau_v)
        loc_thr = _softmax_agg(thr_vals, tau_v)
        uav_local_off[uid] = loc_off
        uav_local_thr[uid] = loc_thr
    # Team coverage/exposure
    if blue_ids_list and alive_attack:
        team_coverage = float(np.mean([
            max(O_ij_dict.get((auid, bid), {"combined": 0.0})["combined"] for auid in alive_attack)
            for bid in blue_ids_list]))
        team_exposure = float(np.mean([
            max(T_ji_dict.get((uid2, bid), {"combined": 0.0})["combined"] for bid in blue_ids_list)
            for uid2 in alive_attack]))
    else:
        team_coverage = 0.0; team_exposure = 0.0
    for uid in alive_attack:
        loc_sit = uav_local_off.get(uid, 0.0) - threat_w * uav_local_thr.get(uid, 0.0)
        sit = local_w * loc_sit + team_w * (team_coverage - team_exposure)
        uav_sit_raw[uid] = _clip_signed(sit)

    # ── MAV role ──
    S_mav = 0.0
    mag_logs = {"marginal_raw": 0.0, "support_dist_q": 0.0, "support_rear_q": 0.0,
                "support_pos": 0.0, "geom_threat": 0.0, "mw": 0.0, "threat_raw": 0.0}
    if mav and mav.is_alive and alive_blue and alive_attack:
        # Marginal information
        all_pair_count = n_attack_alive * n_blue_alive
        marginal_sum = 0.0
        for uid in alive_attack:
            for bid in blue_ids_list:
                ts = env._mav_shared_track_state(uid, bid)
                shared = bool(ts.get("mav_shared_visible", False))
                direct = bool(ts.get("direct_visible", False))
                if shared and not direct:
                    W = max(O_ij_dict.get((uid, bid), {"combined": 0.0})["combined"],
                            T_ji_dict.get((uid, bid), {"combined": 0.0})["combined"])
                    marginal_sum += W
        I_marginal = marginal_sum / max(all_pair_count, 1)
        # Support position
        uav_positions = [env._brma_tam_safe_vec(alive_attack[uid], "get_position") for uid in alive_attack]
        blue_positions = [env._brma_tam_safe_vec(alive_blue[bid], "get_position") for bid in blue_ids_list]
        C_U = np.mean(uav_positions, axis=0)
        C_B = np.mean(blue_positions, axis=0)
        P_M = env._brma_tam_safe_vec(mav, "get_position")
        e_rear = C_U - C_B
        e_rear_norm = float(np.linalg.norm(e_rear))
        rear_proj = 0.0
        if e_rear_norm > 1e-8:
            e_rear = e_rear / e_rear_norm
            rear_proj = float(np.dot(P_M - C_U, e_rear))
        mav_dist = float(np.linalg.norm(P_M - C_U))
        s_min = float(mav_cfg.get("support_min_distance_ratio", 0.5)) * R_launch
        s_max = float(mav_cfg.get("support_max_distance_ratio", 1.5)) * R_launch
        if mav_dist < s_min: dq = _clip_unit(mav_dist / max(s_min, 1e-9))
        elif mav_dist <= s_max: dq = 1.0
        else: dq = _clip_unit(1.0 - (mav_dist - s_max) / max(R_launch, 1e-9))
        rear_ref = float(mav_cfg.get("rear_reference_ratio", 0.75)) * R_launch
        rq = _clip_unit(rear_proj / max(rear_ref, 1e-9))
        P_support = dq * rq
        # MAV threat (from blue to MAV perspective)
        mav_threat_vals = []
        for bid in blue_ids_list:
            blue = alive_blue[bid]
            pq_mav = _compute_pair_quality(blue, mav, env, R_min, D_low, D_high, R_launch, spd_min, spd_max)
            mav_threat_vals.append(pq_mav["combined"])
        geom_threat = _softmax_agg(mav_threat_vals, tau_v)
        mw = 1.0 if (hasattr(mav, "check_missile_warning") and mav.check_missile_warning() is not None) else 0.0
        T_mav_raw = _clip_unit(0.7 * geom_threat + 0.3 * mw)
        # MAV role
        miw = float(mav_cfg.get("marginal_information_weight", 0.5))
        spw = float(mav_cfg.get("support_position_weight", 0.3))
        mtw = float(mav_cfg.get("threat_weight", 0.4))
        S_mav = _clip_signed(miw * I_marginal + spw * P_support - mtw * T_mav_raw)
        mag_logs = {"marginal_raw": I_marginal, "support_dist_q": dq, "support_rear_q": rq,
                    "support_pos": P_support, "geom_threat": geom_threat, "mw": mw, "threat_raw": T_mav_raw}

    # ── Flight ──
    mav_flight_scale = float(flight_cfg.get("mav_scale", 0.01))
    uav_flight_scale = float(flight_cfg.get("uav_scale", 0.01))
    mav_role_scale = float(mav_cfg.get("role_scale", 0.05))
    uav_sit_scale = float(uav_cfg.get("situation_scale", 0.05))

    # ── Per-agent reward assembly ──
    # Build fixed field dict per agent
    FIELD_SET = {
        "reward_contract_revision", "role_situation_v3_task_attrition", "role_situation_v3_task_terminal",
        "role_situation_v3_common", "role_situation_v3_delta_j", "role_situation_v3_j_combat",
        "role_situation_v3_blue_loss_fraction", "role_situation_v3_uav_loss_fraction",
        "role_situation_v3_mav_loss_indicator", "role_situation_v3_blue_loss_delta",
        "role_situation_v3_uav_loss_delta", "role_situation_v3_mav_loss_delta",
        "role_situation_v3_uav_local_offense_raw", "role_situation_v3_uav_local_threat_raw",
        "role_situation_v3_team_coverage_raw", "role_situation_v3_team_exposure_raw",
        "role_situation_v3_uav_situation_raw", "role_situation_v3_uav_situation_scaled",
        "role_situation_v3_uav_situation_encoded",
        "role_situation_v3_mav_marginal_information_raw", "role_situation_v3_mav_support_distance_raw",
        "role_situation_v3_mav_support_rear_raw", "role_situation_v3_mav_support_position_raw",
        "role_situation_v3_mav_geometric_threat_raw", "role_situation_v3_mav_missile_warning",
        "role_situation_v3_mav_threat_raw", "role_situation_v3_mav_role_raw",
        "role_situation_v3_mav_role_scaled", "role_situation_v3_mav_role_encoded",
        "role_situation_v3_flight_raw", "role_situation_v3_flight_norm", "role_situation_v3_flight_scaled",
        "role_situation_v3_flight_encoded", "role_situation_v3_role_encoded",
        "role_situation_v3_total", "role_situation_v3_component_sum", "role_situation_v3_identity_error",
        "role_situation_v3_alive_red_before_count", "role_situation_v3_alive_uav_before_count",
        "role_situation_v3_alive_blue_after_count", "role_situation_v3_is_mav", "role_situation_v3_is_attack_uav",
    }

    for rid in env.red_ids:
        comp = components.setdefault(rid, {})
        sim = env.red_planes.get(rid)
        role = env.agent_roles.get(rid, "")
        is_mav = 1.0 if role == "mav" else 0.0
        is_att = 1.0 if role == "attack_uav" else 0.0
        alive_before = bool(ab.get(rid, False))
        pitch = float(comp.get("r_pitch", 0.0)); roll_v = float(comp.get("r_roll", 0.0))
        vel = float(comp.get("r_vel", 0.0))
        flight_raw_agent = pitch + roll_v + vel

        vals = {k: 0.0 for k in FIELD_SET}
        vals["reward_contract_revision"] = 5.0
        vals["role_situation_v3_delta_j"] = delta_J
        vals["role_situation_v3_j_combat"] = J_combat
        vals["role_situation_v3_blue_loss_fraction"] = L_blue
        vals["role_situation_v3_uav_loss_fraction"] = L_attack
        vals["role_situation_v3_mav_loss_indicator"] = float(L_mav)
        vals["role_situation_v3_blue_loss_delta"] = delta_blue
        vals["role_situation_v3_uav_loss_delta"] = delta_attack
        vals["role_situation_v3_mav_loss_delta"] = delta_mav
        vals["role_situation_v3_alive_red_before_count"] = float(n_R_before)
        vals["role_situation_v3_alive_uav_before_count"] = float(n_U_before)
        vals["role_situation_v3_alive_blue_after_count"] = float(n_blue_alive)
        vals["role_situation_v3_is_mav"] = is_mav
        vals["role_situation_v3_is_attack_uav"] = is_att

        if not alive_before:
            total = 0.0; comp_sum = 0.0
        else:
            vals["role_situation_v3_task_attrition"] = r_attrition
            vals["role_situation_v3_task_terminal"] = r_terminal
            vals["role_situation_v3_common"] = r_common
            flight_norm = _clip_signed(flight_raw_agent)
            vals["role_situation_v3_flight_raw"] = flight_raw_agent
            vals["role_situation_v3_flight_norm"] = flight_norm

            if role == "mav":
                f_scaled = mav_flight_scale * flight_norm
                f_encoded = n_R_before * f_scaled
                r_scaled = mav_role_scale * S_mav
                r_encoded = n_R_before * r_scaled
                vals.update({
                    "role_situation_v3_uav_local_offense_raw": 0.0,
                    "role_situation_v3_uav_local_threat_raw": 0.0,
                    "role_situation_v3_team_coverage_raw": 0.0,
                    "role_situation_v3_team_exposure_raw": 0.0,
                    "role_situation_v3_uav_situation_raw": 0.0,
                    "role_situation_v3_uav_situation_scaled": 0.0,
                    "role_situation_v3_uav_situation_encoded": 0.0,
                    "role_situation_v3_mav_marginal_information_raw": mag_logs["marginal_raw"],
                    "role_situation_v3_mav_support_distance_raw": mag_logs["support_dist_q"],
                    "role_situation_v3_mav_support_rear_raw": mag_logs["support_rear_q"],
                    "role_situation_v3_mav_support_position_raw": mag_logs["support_pos"],
                    "role_situation_v3_mav_geometric_threat_raw": mag_logs["geom_threat"],
                    "role_situation_v3_mav_missile_warning": mag_logs["mw"],
                    "role_situation_v3_mav_threat_raw": mag_logs["threat_raw"],
                    "role_situation_v3_mav_role_raw": S_mav,
                    "role_situation_v3_mav_role_scaled": r_scaled,
                    "role_situation_v3_mav_role_encoded": r_encoded,
                    "role_situation_v3_role_encoded": r_encoded,
                })
                total = r_common + r_encoded + f_encoded
            else:
                S_uav = uav_sit_raw.get(rid, 0.0)
                sit_scaled = uav_sit_scale * S_uav
                sit_encoded = n_R_before / max(n_U_before, 1) * sit_scaled
                f_scaled = uav_flight_scale * flight_norm
                f_encoded = n_R_before / max(n_U_before, 1) * f_scaled
                vals.update({
                    "role_situation_v3_uav_local_offense_raw": uav_local_off.get(rid, 0.0),
                    "role_situation_v3_uav_local_threat_raw": uav_local_thr.get(rid, 0.0),
                    "role_situation_v3_team_coverage_raw": team_coverage,
                    "role_situation_v3_team_exposure_raw": team_exposure,
                    "role_situation_v3_uav_situation_raw": S_uav,
                    "role_situation_v3_uav_situation_scaled": sit_scaled,
                    "role_situation_v3_uav_situation_encoded": sit_encoded,
                    "role_situation_v3_mav_marginal_information_raw": 0.0,
                    "role_situation_v3_mav_support_distance_raw": 0.0,
                    "role_situation_v3_mav_support_rear_raw": 0.0,
                    "role_situation_v3_mav_support_position_raw": 0.0,
                    "role_situation_v3_mav_geometric_threat_raw": 0.0,
                    "role_situation_v3_mav_missile_warning": 0.0,
                    "role_situation_v3_mav_threat_raw": 0.0,
                    "role_situation_v3_mav_role_raw": 0.0,
                    "role_situation_v3_mav_role_scaled": 0.0,
                    "role_situation_v3_mav_role_encoded": 0.0,
                    "role_situation_v3_role_encoded": sit_encoded,
                })
                total = r_common + sit_encoded + f_encoded
            vals["role_situation_v3_flight_scaled"] = f_scaled
            vals["role_situation_v3_flight_encoded"] = f_encoded
            comp_sum = vals["role_situation_v3_common"] + vals["role_situation_v3_role_encoded"] + vals["role_situation_v3_flight_encoded"]

        vals["role_situation_v3_total"] = total
        vals["role_situation_v3_component_sum"] = comp_sum
        vals["role_situation_v3_identity_error"] = total - comp_sum
        if abs(total - comp_sum) > 1e-6:
            raise ValueError(f"v3 identity failure: agent={rid} total={total} comp_sum={comp_sum} error={total-comp_sum}")

        comp.update(vals)
        comp["total"] = float(total)
        base_rewards[rid] = float(total)

    return base_rewards, components
