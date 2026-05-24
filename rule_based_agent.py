"""
rule_based_agent.py —— 蓝方规则策略（四层状态机自动驾驶仪 + 协同目标分配）。

Single source of truth for the blue-team rule-based policy.
Imported by both ``train_vanilla_mappo.py`` (training) and ``eval_acmi.py``
(evaluation / TacView replay).

Action-space contract (env.py ``_parse_actions``, paper §2.4):
    action[0] = pitch_cmd    ∈ [−1, 1]  → target_pitch   ∈ [−90°, +90°]   (absolute)
    action[1] = heading_cmd  ∈ [−1, 1]  → target_heading ∈ [−180°, +180°]  (absolute ψ)
    action[2] = vel_cmd      ∈ [−1, 1]  → target_velocity ∈ [102, 408] m/s (M0.3–M1.2)

Observation vector (11-dim entity, env.py ``_make_entity_vec``, body-frame):
    [Δx, Δy, Δz, AO_signed, TA, R, V_tgt,
     sin(φ), cos(φ), sin(θ), cos(θ)]
    idx: 0   1   2      3      4  5   6       7       8       9      10

Key: AO_signed ∈ [−π, π]  (+ right, − left)
     TA        ∈ [0, π]    (unsigned)

Internal action convention (before rescaling):
    pitch_cmd_int    ∈ [−1, 1]  → ±90°  (full paper §2.4 authority)
    heading_cmd_int  ∈ [−1, 1]  → ±10° heading delta (legacy, converted to absolute)
    vel_cmd_int      ∈ [−1, 1]  → [250, 350] m/s (legacy, remapped to [102, 408])
"""
from __future__ import annotations

import numpy as np

# ==============================================================================
#  Per-agent hysteresis state (module-level — survives across env steps)
# ==============================================================================
_prev_heading_cmd: dict[int, float] = {}     # blue_id → last heading_cmd (internal [−1,1])
_prev_lead_bearing: dict[int, float] = {}    # blue_id → last lead_bearing (rad)

# ==============================================================================
#  State thresholds
# ==============================================================================

# ---- Hard Deck: never fight below 4500 m ----
HARD_DECK        = 4500.0   # < this → force climb, full throttle, no combat
SAFE_COMBAT_ALT  = 6000.0   # below this → graduated dive restriction + full throttle
DOOMED_ALT       = 3000.0   # ignore enemy targets below this altitude

# ---- Descent-rate safety ----
MAX_DESCENT_RATE = 40.0     # m/s — if descending faster than this below 5500 m, force climb
DESCENT_WARN_ALT = 5500.0   # m — check descent rate below this altitude

# ---- Trim (physics correction: 0° target_pitch ≠ level flight) ----
# F-16 at 20 000 ft / M0.88 / ~28 000 lb requires ≈ 4° pitch-up for
# steady level flight.  target_pitch = trim × (π/2)  = trim × 90°
#   0.0467 → 4.2°  (level cruise with slight margin)
_TRIM_BASELINE = 0.0467

# ---- Combat authority ----
_COMBAT_PITCH_LIMIT   = 0.45          # ±40.5° for pursuit (was ±15°)
_COMBAT_HEADING_LIMIT = 1.0           # full ±10° heading delta
_COMBAT_MAX_BANK      = np.deg2rad(75)  # 75° bank → 3.86G turn, radius 2.4 km @ 300 m/s
_HARD_TURN_AO_RAD     = np.deg2rad(45)

# ---- Safe zone: no dive between Hard Deck and SAFE_COMBAT_ALT ----
_DIVE_FREEZE_ALT = 5000.0  # below this, pitch must be >= 0 (no dive at all)

# ---- Stall protection ----
_STALL_SPEED        = 200.0    # m/s — below this, stall risk at manoeuvring AoA
_STALL_PROTECT_ALT  = 5000.0   # m — below this altitude, low speed triggers recovery

# ---- Heading hysteresis (anti-oscillation) ----
_HEADING_HYST_DEG   = 5.0      # degrees — don't update target_heading for Δ < this
_HEADING_HYST_RAD   = np.deg2rad(_HEADING_HYST_DEG)


# ==============================================================================
#  blue_coordinated_actions —— 协同目标分配入口（推荐调用此函数）
# ==============================================================================

def blue_coordinated_actions(
    blue_obs: dict[str, dict],
    num_blue: int,
    num_red: int,
    engaged_targets: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Greedy target deconfliction: distribute blues across different reds.

    Algorithm:
      1. Each blue independently scores every alive, unengaged red.
      2. Blues are sorted by their best score (most promising engagement first).
      3. Greedy assignment: best blue → best red, next blue → best UNTAKEN red, ...

    The optional ``engaged_targets`` set (red UIDs like ``"red_0"``) is both
    consumed and mutated in-place.  Reds already in the set are excluded from
    scoring (they already have a friendly missile in flight or were assigned
    to another blue earlier in the same allocation).  When a blue is assigned
    to a red, that red's UID is immediately added to ``engaged_targets`` so
    subsequent blues in the same call skip it — flight-level "no-ganging-up".

    If ``engaged_targets`` is None, the function works as before (backward
    compatible for training loops that don't have env-level access).
    """
    blue_ids = [f"blue_{i}" for i in range(num_blue)]

    # ---- Convert engaged UIDs to red indices ----
    engaged_red_indices: set[int] = set()
    if engaged_targets:
        for uid in engaged_targets:
            if uid.startswith("red_"):
                try:
                    engaged_red_indices.add(int(uid.split("_")[1]))
                except (ValueError, IndexError):
                    pass

    # ---- Build score matrix: score[b][r] for alive reds ----
    # Pre-filter alive reds (same across all blues since death_mask is shared)
    # We read from the first blue's obs (all blues share the same red state).
    first_obs = blue_obs[blue_ids[0]]
    death_mask = first_obs["death_mask"]
    alive_reds_all = [i for i in range(num_red) if death_mask[num_blue + i] > 0.5]

    if not alive_reds_all:
        actions = {}
        for bid in blue_ids:
            # No targets at all — each blue cruises independently
            actions[bid] = _blue_pursuit_action_impl(
                blue_obs[bid], num_blue, num_red, int(bid.split("_")[1]),
                forced_target_idx=None)
        return actions

    # Score every blue × red pair
    score = np.zeros((num_blue, num_red), dtype=np.float32)
    for b_idx, bid in enumerate(blue_ids):
        obs = blue_obs[bid]
        enemy_states = obs["enemy_states"]
        for r_idx in alive_reds_all:
            # --- Flight-level deconfliction: skip reds already engaged ---
            if r_idx in engaged_red_indices:
                score[b_idx, r_idx] = -1.0
                continue
            tgt_vec = enemy_states[r_idx]
            R  = float(tgt_vec[5]) * 80000.0
            AO = float(tgt_vec[3]) * np.pi
            TA = float(tgt_vec[4]) * np.pi
            # AWACS fallback: when radar can't see the target (TA=0), use a
            # 5° floor so Blue still pursues based on AWACS position data.
            TA_eff = max(TA, np.deg2rad(5))
            # Merge fix: AO floor prevents score→0 when target passes behind
            # (AO≈π), which would otherwise cause Blue to lose lock and cruise.
            ao_weight = max(0.1, 1.0 - abs(AO) / np.pi)
            score[b_idx, r_idx] = (1.0 / max(R, 300.0)) * ao_weight * (TA_eff / np.pi)

    # ---- Greedy assignment ----
    # Sort blues by their best score (descending)
    blue_best_score = np.max(score, axis=1, initial=0.0)  # best unengaged score per blue
    blue_order = sorted(range(num_blue), key=lambda i: blue_best_score[i], reverse=True)

    taken_reds: set[int] = set()
    assignments: dict[int, int | None] = {}  # blue_idx → red_idx or None

    for b_idx in blue_order:
        # Find best UNTAKEN red
        best_r = None
        best_s = 0.0
        for r_idx in alive_reds_all:
            if r_idx in taken_reds or r_idx in engaged_red_indices:
                continue
            if score[b_idx, r_idx] > best_s:
                best_s = score[b_idx, r_idx]
                best_r = r_idx
        if best_r is not None:
            taken_reds.add(best_r)
            assignments[b_idx] = best_r
            # ---- Hot-update: immediately mark red as engaged ----
            # Subsequent blues in this same allocation will skip this red,
            # preventing flight-level "ganging up" on a single target.
            engaged_red_indices.add(best_r)
            if engaged_targets is not None:
                engaged_targets.add(f"red_{best_r}")
        else:
            assignments[b_idx] = None  # all reds taken — fall back to free selection

    # ---- Generate actions ----
    actions: dict[str, np.ndarray] = {}
    for b_idx, bid in enumerate(blue_ids):
        actions[bid] = _blue_pursuit_action_impl(
            blue_obs[bid], num_blue, num_red, b_idx,
            forced_target_idx=assignments[b_idx])
    return actions


# ==============================================================================
#  blue_pursuit_action —— 单机调用（向后兼容）
# ==============================================================================

def blue_pursuit_action(obs: dict, num_blue: int, num_red: int, blue_id: int,
                        missile_warning: bool = False) -> np.ndarray:
    """Per-aircraft entry point (legacy — prefer ``blue_coordinated_actions``)."""
    return _blue_pursuit_action_impl(obs, num_blue, num_red, blue_id,
                                     forced_target_idx=None)


# ==============================================================================
#  _blue_pursuit_action_impl —— 核心自动驾驶仪
# ==============================================================================

def _blue_pursuit_action_impl(
    obs: dict,
    num_blue: int,
    num_red: int,
    blue_id: int,
    forced_target_idx: int | None,
) -> np.ndarray:
    """四层状态机自动驾驶仪（优先级从高到低）。

    HARD_DECK      → 强制爬升 +45° + 满油门, 禁止作战 (< 4500 m)
    DESCENT_WARN   → 下坠率过大时预判拉起 (< 5500 m, 下坠率 > 40 m/s)
    STALL PROTECT  → 低速 + 低空组合保护 (< 5000 m, 速度 < 200 m/s)
    ANTI-STALL     → 早期能量管理 (速度 < 250 m/s 且抬头)
    COMBAT         → 带进近前置角的拦截引导 (4500–15000 m, JSBSim 自然升限约束)
    """

    # ---- 读取物理姿态 ----
    ego_state     = obs["ego_state"]
    ego_sin_roll  = float(ego_state[7])
    ego_cos_roll  = float(ego_state[8])
    ego_roll      = np.arctan2(ego_sin_roll, ego_cos_roll)
    ego_roll_abs  = abs(ego_roll)
    ego_sin_pitch = float(ego_state[9])
    ego_pitch     = np.arctan2(ego_sin_pitch, float(ego_state[10]))
    ego_vel       = float(ego_state[6]) * 600.0   # V_tgt / MAX_SPEED → m/s
    alt_m         = float(obs["altitude"][0])
    v_up          = float(obs["velocity"][2])
    # Current heading from NED velocity (needed for delta→absolute conversion)
    our_vn = float(obs["velocity"][0])
    our_ve = float(obs["velocity"][1])
    our_heading = np.arctan2(our_ve, our_vn)  # rad, [−π, π]

    # Rescale helper: internal convention → env.py paper §2.4 targets
    #   pitch_new  = pitch_int × 1.0        → target_pitch  ∈ [−90°, +90°]
    #   heading_new = δ_heading → absolute  → target_heading ∈ [−180°, +180°]
    #   vel_new     = [250,350] → [102,408] → target_velocity ∈ [M0.3, M1.2]
    def _rescale(pitch_int, heading_int, vel_int):
        pitch_new = pitch_int                         # full ±90° authority
        target_heading = our_heading + heading_int * np.deg2rad(10.0)  # δ→absolute
        target_heading = (target_heading + np.pi) % (2 * np.pi) - np.pi
        heading_new = target_heading / np.pi            # → [−1, 1]
        vel_new = (90.0 + 100.0 * vel_int) / 306.0     # [250,350]→[102,408]
        vel_new = np.clip(vel_new, -1.0, 1.0)
        return np.array([pitch_new, heading_new, vel_new], dtype=np.float32)

    # =========================================================================
    #  HARD DECK — emergency climb, bypass combat
    # =========================================================================
    if alt_m < HARD_DECK:
        heading_cmd = -np.sign(ego_roll) * 1.0             # wings level
        return _rescale(0.45, heading_cmd, 1.0)            # +40.5°, full throttle

    # =========================================================================
    #  DESCENT-RATE SAFETY — pre-emptive climb before the hard deck
    # =========================================================================
    if alt_m < DESCENT_WARN_ALT and v_up < -MAX_DESCENT_RATE:
        heading_cmd = -np.sign(ego_roll) * 1.0
        return _rescale(0.45, heading_cmd, 1.0)            # +40.5°, full throttle

    # =========================================================================
    #  STALL PROTECTION — combined low altitude + low speed (Fix 3)
    #
    #  F-16 stall speed at 20 000 ft / M0.85 / ~13 000 kg is ≈ 130–150 m/s
    #  calibrated.  At high AoA (manoeuvring) the margin shrinks.  This
    #  layer triggers BEFORE the aircraft reaches critical AoA and forces
    #  a wings-level energy recovery.
    # =========================================================================
    if alt_m < _STALL_PROTECT_ALT and ego_vel < _STALL_SPEED:
        heading_cmd = -np.sign(ego_roll) * 1.0             # wings level
        return _rescale(0.45, float(heading_cmd), 1.0)     # +40.5°, full throttle

    # =========================================================================
    #  ANTI-STALL — early energy management before the stall-protect layer
    # =========================================================================
    if ego_vel < 250.0 and ego_pitch > 0 and alt_m >= SAFE_COMBAT_ALT:
        heading_cmd = -np.clip(ego_roll / np.deg2rad(45), -0.5, 0.5)
        return _rescale(-0.12, float(heading_cmd), 1.0)    # −10.8° dive, full throttle
    if ego_vel < 250.0 and ego_pitch > 0 and alt_m < SAFE_COMBAT_ALT:
        heading_cmd = -np.clip(ego_roll / np.deg2rad(45), -0.5, 0.5)
        return _rescale(0.22, float(heading_cmd), 1.0)     # +19.8° climb, full throttle

    # =========================================================================
    #  COMBAT — Lead Pursuit with graduated dive restriction
    # =========================================================================
    enemy_states = obs["enemy_states"]
    death_mask   = obs["death_mask"]

    # --- Doomed target filter ---
    alive_reds_raw = [i for i in range(num_red)
                      if death_mask[num_blue + i] > 0.5]

    alive_reds = []
    for idx in alive_reds_raw:
        tgt_vec = enemy_states[idx]
        delta_alt = float(tgt_vec[2]) * 10000.0
        tgt_alt = alt_m + delta_alt
        if tgt_alt > DOOMED_ALT:
            alive_reds.append(idx)

    # --- Target selection (radar-detected reds only, TA > 0) ---
    target_idx: int | None = None
    if forced_target_idx is not None and forced_target_idx in alive_reds:
        target_idx = forced_target_idx
    elif alive_reds:
        best_score = 0.0
        for idx in alive_reds:
            tgt_vec = enemy_states[idx]
            R  = float(tgt_vec[5]) * 80000.0
            AO = float(tgt_vec[3]) * np.pi
            TA = float(tgt_vec[4]) * np.pi
            # AWACS fallback: TA floor prevents zero-score when radar blind
            TA_eff = max(TA, np.deg2rad(5))
            # Merge fix: AO floor prevents score→0 after head-on pass (AO≈π)
            ao_weight = max(0.1, 1.0 - abs(AO) / np.pi)
            score_ = (1.0 / max(R, 300.0)) * ao_weight * (TA_eff / np.pi)
            if score_ > best_score:
                best_score = score_
                target_idx = idx

    # --- Cruise: all reds either dead or undetected by radar ---
    if target_idx is None:
        _prev_lead_bearing.pop(blue_id, None)
        _prev_heading_cmd.pop(blue_id, None)
        alt_error = SAFE_COMBAT_ALT - alt_m
        cruise_pitch = np.clip(alt_error / 2000.0, -0.10, 0.12) + _TRIM_BASELINE
        return _rescale(float(cruise_pitch), 0.0, 1.0)

    tgt = enemy_states[target_idx]

    # De-normalise
    delta_alt = float(tgt[2]) * 10000.0
    AO        = float(tgt[3]) * np.pi        # radians, signed: +right/−left
    TA        = float(tgt[4]) * np.pi        # radians, unsigned
    R         = float(tgt[5]) * 80000.0
    V_tgt     = float(tgt[6]) * 600.0        # target speed, m/s

    # =========================================================================
    #  Pitch — Pure Pursuit with graduated dive restriction
    # =========================================================================
    pitch_cmd = np.clip(delta_alt / max(R, 300.0) * 3.0, -0.45, 0.45)

    if alt_m < _DIVE_FREEZE_ALT:
        pitch_cmd = max(pitch_cmd, 0.0)
    elif alt_m < SAFE_COMBAT_ALT:
        alt_risk = np.clip((SAFE_COMBAT_ALT - alt_m) / (SAFE_COMBAT_ALT - _DIVE_FREEZE_ALT),
                           0.0, 1.0)
        max_allowed_dive = -0.50 * (1.0 - alt_risk) + (-0.25) * alt_risk
        pitch_cmd = max(pitch_cmd, max_allowed_dive)

    pitch_cmd = pitch_cmd + _TRIM_BASELINE

    # G-compensation: at 75° bank (3.86G), factor 0.05 adds ≈12° beyond trim,
    # giving total ≈16.2° — enough for a level sustained turn.
    if ego_roll_abs > np.deg2rad(15):
        required_g = 1.0 / max(np.cos(ego_roll), 0.1)
        pitch_cmd = pitch_cmd + float((required_g - 1.0) * 0.05)

    if ego_roll_abs > np.deg2rad(70) and ego_pitch < np.deg2rad(-3):
        pitch_cmd = max(pitch_cmd, 0.13)

    pitch_cmd = np.clip(pitch_cmd, -_COMBAT_PITCH_LIMIT, _COMBAT_PITCH_LIMIT)

    # =========================================================================
    #  Heading — Lead Pursuit (intercept course, not pure pursuit)
    #
    #  Pure pursuit (point nose at target's current position) produces a
    #  tail-chase that never reaches the 3-9 line.  Lead pursuit predicts
    #  the target's future position and steers toward the intercept point,
    #  which naturally converges to the target's beam aspect.
    #
    #  Derivation:
    #    α = our heading   (atan2(v_east, v_north))
    #    β = target heading = α + AO + TA   (from AO/TA geometry)
    #
    #    tgt_velocity = V_tgt · (cos β,  sin β)
    #
    #    Lead time:        t_lead = R / max(V_ego, V_tgt, 1)
    #    Lead point:       P_lead = P_tgt + V_tgt · t_lead
    #    Lead AO:          AO_lead = bearing_to(P_lead) − α
    #
    #  The resulting heading command points the nose at the intercept point
    #  rather than the target's current position.
    # =========================================================================
    our_vn = float(obs["velocity"][0])
    our_ve = float(obs["velocity"][1])
    our_heading = np.arctan2(our_ve, our_vn)        # radians, [−π, π]
    our_speed = max(np.hypot(our_vn, our_ve), 1.0)

    # ---- Body-frame → NED rotation for position deltas ----
    # The observation provides target position in body frame (env.py
    # _build_body_frame_entity applies R_BI to NED deltas).  We must rotate
    # back to NED before mixing with NED velocity vectors, otherwise the
    # lead-point prediction mixes frames and steers toward a phantom location.
    psi = our_heading       # yaw ≈ heading (exact for coordinated flight)
    theta = ego_pitch
    phi = ego_roll

    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)

    # De-normalise body-frame deltas.
    # tgt[0:2] ÷ 40000 → body-frame Δx, Δy (forward, right).
    # tgt[2]    ÷ 10000 → pseudo-up = −body_z (env.py §_normalize_obs_vec).
    #                    Negate to recover body-frame Δz (down = positive).
    dx_body = float(tgt[0]) * 40000.0
    dy_body = float(tgt[1]) * 40000.0
    dz_body = -float(tgt[2]) * 10000.0

    # R_IB = R_z(ψ)·R_y(θ)·R_x(φ) applied to [dx_body, dy_body, dz_body]^T
    dn = (c_psi * c_theta) * dx_body \
       + (c_psi * s_theta * s_phi - s_psi * c_phi) * dy_body \
       + (c_psi * s_theta * c_phi + s_psi * s_phi) * dz_body
    de = (s_psi * c_theta) * dx_body \
       + (s_psi * s_theta * s_phi + c_psi * c_phi) * dy_body \
       + (s_psi * s_theta * c_phi - c_psi * s_phi) * dz_body

    # Target heading from AO/TA geometry: β = α + AO + TA
    target_heading = our_heading + AO + TA

    # Lead time estimate (capped to prevent wild predictions at long range)
    t_lead = min(R / max(our_speed, V_tgt, 1.0), 30.0)

    # Target velocity vector (horizontal NED plane)
    tgt_vn = V_tgt * np.cos(target_heading)
    tgt_ve = V_tgt * np.sin(target_heading)

    # Lead point: where the target WILL be at intercept (all in NED now)
    lead_n = dn + tgt_vn * t_lead
    lead_e = de + tgt_ve * t_lead

    # Bearing from us to the lead point
    lead_bearing = np.arctan2(lead_e, lead_n)
    AO_lead = lead_bearing - our_heading
    AO_lead = (AO_lead + np.pi) % (2 * np.pi) - np.pi   # wrap to [−π, π]

    # Map lead AO to heading command
    heading_cmd = np.clip(AO_lead / (np.pi / 3), -1.0, 1.0)

    # ---- Heading hysteresis: suppress small bearing changes (anti-wobble) ----
    # When the lead-point bearing shifts by < 5°, keep the previous heading
    # command.  This prevents the BTT PID from reacting to every tiny frame-
    # to-frame jitter in the intercept geometry, which would otherwise excite
    # the roll loop at 5 Hz.
    prev_lead = _prev_lead_bearing.get(blue_id)
    if prev_lead is not None:
        bearing_delta = (lead_bearing - prev_lead + np.pi) % (2 * np.pi) - np.pi
        if abs(bearing_delta) < _HEADING_HYST_RAD and blue_id in _prev_heading_cmd:
            heading_cmd = _prev_heading_cmd[blue_id]
    _prev_lead_bearing[blue_id] = float(lead_bearing)
    _prev_heading_cmd[blue_id] = float(heading_cmd)

    # ---- Bank limiting (anti-oscillation fix) ----
    # OLD behaviour:  ego_roll > 75° → flip heading_cmd sign → rapid un-bank →
    #   lead pursuit re-acquires → hard turn → exceeds 75° again → BANG-BANG.
    #
    # NEW behaviour:  progressively dampen the turn-rate command as bank
    #   approaches the limit, without ever reversing direction.  At > 75°
    #   the aircraft still gets a turn command in the SAME direction, just
    #   at 10 % strength — enough to maintain current bank without deepening.
    if ego_roll_abs > _COMBAT_MAX_BANK:                          # > 75°
        heading_cmd *= 0.1                                        # heavily dampen, same sign
    elif ego_roll_abs > _COMBAT_MAX_BANK - np.deg2rad(15):       # > 60°
        frac = (_COMBAT_MAX_BANK - ego_roll_abs) / np.deg2rad(15)  # 1→0
        heading_cmd *= max(0.0, frac)                              # smooth roll-off

    heading_cmd = np.clip(heading_cmd, -_COMBAT_HEADING_LIMIT, _COMBAT_HEADING_LIMIT)

    # =========================================================================
    #  Throttle — energy management
    # =========================================================================
    if alt_m < SAFE_COMBAT_ALT or ego_vel < 250.0:
        vel_cmd = 1.0
    elif abs(AO) > _HARD_TURN_AO_RAD:
        vel_cmd = 1.0
    elif R > 8000:
        vel_cmd = 1.0
    elif R < 2000:
        vel_cmd = 0.3
    else:
        vel_cmd = 0.6

    if TA > np.deg2rad(90):
        vel_cmd = max(vel_cmd, 1.0)

    return _rescale(pitch_cmd, heading_cmd, vel_cmd)
