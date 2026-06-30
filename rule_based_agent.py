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
_last_target_bearing: dict[int, float] = {}  # blue_id -> last radar/AWACS target bearing (rad)
_lost_target_steps: dict[int, int] = {}       # blue_id -> short-horizon reacquisition age
_simple_last_seen_bearing: dict[int, float] = {}
_simple_lost_steps: dict[int, int] = {}
_simple_debug_state: dict[int, dict] = {}
BLUE_POLICY_DEBUG = False

# ==============================================================================
#  State thresholds
# ==============================================================================

# ---- Hard Deck: never fight below 4500 m ----
HARD_DECK        = 4500.0   # < this → force climb, full throttle, no combat
SAFE_COMBAT_ALT  = 6000.0   # below this → graduated dive restriction + full throttle
DOOMED_ALT       = 3000.0   # deprecated: env death_mask handles target validity

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
_SIMPLE_REACQUIRE_STEPS = 15


def _boundary_patrol_heading_command(
    own_position: np.ndarray,
    current_heading: float,
    boundary_half_size: float = 40000.0,
    boundary_margin: float = 12000.0,
) -> float:
    """Return no-target patrol heading command near the battlefield boundary.

    The return value is the rule-agent internal heading command in [-1, 1],
    where 1.0 corresponds to the existing +10 degree heading authority.  This
    helper only uses ownship position and heading; it does not use enemy state.

    Heading strength is pressure- and outward-motion-scaled: outbound flight
    near the boundary gets strong correction, inbound flight gets weak
    correction to reduce oscillation.
    """

    pressure = _boundary_patrol_pressure(
        own_position, boundary_half_size, boundary_margin)
    if pressure <= 0.0:
        return 0.0

    pos = np.asarray(own_position, dtype=np.float32)
    x, y = float(pos[0]), float(pos[1])
    center_bearing = np.arctan2(-y, -x)
    heading_error = (center_bearing - current_heading + np.pi) % (2 * np.pi) - np.pi
    center_cmd = heading_error / np.deg2rad(10.0)
    outward = _boundary_outward_heading_component(own_position, current_heading)
    if outward > 0.2:
        gain = float(np.clip(0.7 + pressure, 0.7, 1.0))
    elif outward > -0.2:
        gain = float(np.clip(0.3 + 0.5 * pressure, 0.3, 0.8))
    else:
        gain = float(np.clip(0.15 + 0.3 * pressure, 0.15, 0.5))
    heading_cmd = center_cmd * gain
    return float(np.clip(heading_cmd, -1.0, 1.0))


def _boundary_outward_heading_component(
    own_position: np.ndarray,
    current_heading: float,
) -> float:
    """Return how much current heading points outward from battlefield center.

    Returns approximately +1 for directly outward, 0 for tangent, and -1 for
    directly inward. Coordinates follow the project convention:
    own_position = [north, east, up], current_heading = atan2(ve, vn).
    """

    pos = np.asarray(own_position, dtype=np.float32)
    if pos.shape[0] < 2:
        return 0.0
    xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float32)
    norm = float(np.hypot(xy[0], xy[1]))
    if norm < 1e-6:
        return 0.0
    outward_unit = xy / norm
    heading_unit = np.array(
        [np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)
    return float(np.clip(np.dot(heading_unit, outward_unit), -1.0, 1.0))


def _boundary_patrol_pressure(
    own_position: np.ndarray,
    boundary_half_size: float = 40000.0,
    boundary_margin: float = 12000.0,
) -> float:
    """Return no-target boundary pressure for cruise patrol.

    0 means safely inside the inner patrol area, 0..1 means approaching the
    battlefield boundary, and >1 means outside the nominal battlefield.
    """

    pos = np.asarray(own_position, dtype=np.float32)
    if pos.shape[0] < 2:
        return 0.0
    x, y = float(pos[0]), float(pos[1])
    radial_distance = max(abs(x), abs(y))
    inner_limit = boundary_half_size - boundary_margin
    pressure = (radial_distance - inner_limit) / max(boundary_margin, 1e-6)
    return float(np.clip(pressure, 0.0, 1.5))


def _current_heading_from_obs(obs: dict) -> float:
    """Return current heading from observation velocity in north/east order."""

    return float(np.arctan2(float(obs["velocity"][1]), float(obs["velocity"][0])))


def _blue_cruise_heading_command(
    obs: dict,
    blue_id: int,
    own_position: np.ndarray | None = None,
    current_heading: float | None = None,
) -> float:
    """Return internal heading command for no-target cruise.

    Internal heading command is delta-heading in [-1, 1], where `_rescale`
    maps it to current_heading + heading_cmd * 10 deg.

    This function must not use enemy position. It only uses own position,
    own velocity/heading, and battlefield-boundary logic. Current env
    observations do not expose own battlefield x/y, so callers must pass
    `own_position` to enable boundary-aware cruise.
    """

    if own_position is None:
        # TODO: Boundary-aware cruise needs own battlefield x/y from an
        # env-side helper or explicit caller-supplied position.
        return 0.0

    if current_heading is None:
        current_heading = _current_heading_from_obs(obs)
    return _boundary_patrol_heading_command(own_position, current_heading)


def _blue_cruise_speed_command(
    own_position: np.ndarray | None = None,
    current_heading: float | None = None,
) -> float:
    """Return internal velocity command for no-target cruise."""

    if own_position is None:
        return 1.0
    pressure = _boundary_patrol_pressure(own_position)
    if pressure <= 0.0:
        return 1.0
    outward = 0.0
    if current_heading is not None:
        outward = _boundary_outward_heading_component(own_position, current_heading)
    base = 1.0 - 0.8 * min(pressure, 1.0)
    if outward > 0.2:
        base -= 0.2 * min(outward, 1.0)
    return float(np.clip(base, 0.15, 1.0))


def _should_override_for_boundary_safety(
    own_position: np.ndarray | None,
    current_heading: float,
    boundary_half_size: float = 40000.0,
    boundary_margin: float = 12000.0,
) -> bool:
    """Return True when boundary safety should override combat pursuit.

    This only uses Blue ownship position and heading. It does not inspect
    enemy state or change target selection.
    """

    if own_position is None:
        return False
    pressure = _boundary_patrol_pressure(
        own_position, boundary_half_size, boundary_margin)
    outward = _boundary_outward_heading_component(own_position, current_heading)
    return bool(
        (pressure >= 0.75 and outward > 0.15)
        or (pressure >= 1.0 and outward > -0.2)
    )


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def reset_rule_memory() -> None:
    """Clear module-level blue rule memory for a new evaluation/training run."""

    _prev_heading_cmd.clear()
    _prev_lead_bearing.clear()
    _last_target_bearing.clear()
    _lost_target_steps.clear()
    _simple_last_seen_bearing.clear()
    _simple_lost_steps.clear()
    _simple_debug_state.clear()


def _simple_center_bearing(own_position: np.ndarray | None, current_heading: float) -> float:
    if own_position is None:
        return float(current_heading)
    pos = np.asarray(own_position, dtype=np.float64).reshape(-1)
    if pos.size < 2:
        return float(current_heading)
    if np.linalg.norm(pos[:2]) < 1e-6:
        return float(current_heading)
    return _wrap_pi(np.arctan2(-float(pos[1]), -float(pos[0])))


def _simple_valid_targets(
    obs: dict,
    num_blue: int,
    num_red: int,
    excluded: set[int] | None = None,
) -> list[tuple[float, int, np.ndarray]]:
    enemy_states = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
    death_mask = np.asarray(obs.get("death_mask", []), dtype=np.float32).reshape(-1)
    if enemy_states.ndim != 2 or enemy_states.shape[0] == 0:
        return []
    excluded = excluded or set()
    candidates: list[tuple[float, int, np.ndarray]] = []
    for red_idx in range(min(num_red, enemy_states.shape[0])):
        if red_idx in excluded:
            continue
        if death_mask.size > num_blue + red_idx and death_mask[num_blue + red_idx] <= 0.5:
            continue
        state = np.asarray(enemy_states[red_idx], dtype=np.float32)
        if state.size < 6 or np.allclose(state, 0.0):
            continue
        range_m = float(state[5]) * 80000.0
        if not np.isfinite(range_m) or range_m <= 1.0:
            continue
        candidates.append((range_m, red_idx, state))
    return sorted(candidates, key=lambda item: item[0])


def _simple_nearest_target(
    obs: dict,
    num_blue: int,
    num_red: int,
    excluded: set[int] | None = None,
) -> tuple[int | None, np.ndarray | None, float | None]:
    candidates = _simple_valid_targets(obs, num_blue, num_red, excluded)
    if not candidates and excluded:
        candidates = _simple_valid_targets(obs, num_blue, num_red, set())
    if not candidates:
        return None, None, None
    range_m, idx, state = candidates[0]
    return idx, state, range_m


def _simple_rescale_absolute_heading(pitch_int: float, target_heading_abs: float, vel_int: float) -> np.ndarray:
    vel_new = (90.0 + 100.0 * float(vel_int)) / 306.0
    return np.array([
        float(np.clip(pitch_int, -1.0, 1.0)),
        float(np.clip(_wrap_pi(target_heading_abs) / np.pi, -1.0, 1.0)),
        float(np.clip(vel_new, -1.0, 1.0)),
    ], dtype=np.float32)


def _blue_simple_pursuit_action_impl(
    obs: dict,
    num_blue: int,
    num_red: int,
    blue_id: int,
    forced_target_idx: int | None,
    own_position: np.ndarray | None = None,
    own_heading: float | None = None,
) -> np.ndarray:
    """Simple safe-pursuit opponent path.

    This opt-in path is deliberately separate from the legacy BRMA rule:
    nearest valid target, current AO bearing, short last-seen reacquire, and
    fixed throttle choices. It does not use lead pursuit, target scoring,
    heading hysteresis, bank damping, or G compensation.
    """

    ego_state = np.asarray(obs.get("ego_state", np.zeros(11)), dtype=np.float32)
    velocity = np.asarray(obs.get("velocity", np.zeros(3)), dtype=np.float32)
    altitude = np.asarray(obs.get("altitude", [SAFE_COMBAT_ALT]), dtype=np.float32).reshape(-1)
    ego_roll = np.arctan2(float(ego_state[7]) if ego_state.size > 7 else 0.0,
                          float(ego_state[8]) if ego_state.size > 8 else 1.0)
    ego_vel = float(ego_state[6]) * 600.0 if ego_state.size > 6 else float(np.linalg.norm(velocity))
    alt_m = float(altitude[0]) if altitude.size else SAFE_COMBAT_ALT
    v_up = float(velocity[2]) if velocity.size > 2 else 0.0
    if own_heading is None:
        if velocity.size >= 2 and np.linalg.norm(velocity[:2]) > 1e-6:
            our_heading = float(np.arctan2(float(velocity[1]), float(velocity[0])))
        else:
            our_heading = 0.0
    else:
        our_heading = float(own_heading)

    def _record(source: str, target_idx: int | None, state: np.ndarray | None,
                range_m: float | None, desired_heading: float, action: np.ndarray,
                reacquire: bool = False) -> np.ndarray:
        _simple_debug_state[blue_id] = {
            "pursuit_variant": "simple_safe_pursuit",
            "simple_target_selection": "nearest_valid",
            "desired_heading_source": source,
            "uses_red_action_bounds": 1,
            "simple_reacquire_active": int(reacquire),
            "simple_lost_steps": int(_simple_lost_steps.get(blue_id, 0)),
            "selected_target_idx": target_idx,
            "selected_range_m": range_m if range_m is not None else "",
            "selected_AO_rad": float(state[3]) * np.pi if state is not None and state.size > 3 else "",
            "selected_TA_rad": float(state[4]) * np.pi if state is not None and state.size > 4 else "",
            "selected_target_quality": _target_track_quality(state) if state is not None else "invalid",
            "action_heading_abs_rad": _wrap_pi(desired_heading),
            "action_heading_norm": float(action[1]),
            "roll_recovery_active": 0,
            "extreme_roll_recovery_active": 0,
        }
        return action

    center_heading = _simple_center_bearing(own_position, our_heading)
    if alt_m < HARD_DECK:
        action = _simple_rescale_absolute_heading(0.45, center_heading, 1.0)
        return _record("safety", None, None, None, center_heading, action)
    if alt_m < DESCENT_WARN_ALT and v_up < -MAX_DESCENT_RATE:
        action = _simple_rescale_absolute_heading(0.45, center_heading, 1.0)
        return _record("safety", None, None, None, center_heading, action)
    if ego_vel < 220.0:
        action = _simple_rescale_absolute_heading(max(0.15, _TRIM_BASELINE), our_heading, 1.0)
        return _record("low_speed_recovery", None, None, None, our_heading, action)
    if _should_override_for_boundary_safety(own_position, our_heading):
        pitch = np.clip((SAFE_COMBAT_ALT - alt_m) / 2000.0, -0.10, 0.15) + _TRIM_BASELINE
        action = _simple_rescale_absolute_heading(float(pitch), center_heading, 0.8)
        return _record("safety", None, None, None, center_heading, action)

    target_idx = None
    target_state = None
    range_m = None
    if forced_target_idx is not None:
        candidates = _simple_valid_targets(obs, num_blue, num_red, set())
        for cand_range, cand_idx, cand_state in candidates:
            if cand_idx == forced_target_idx:
                target_idx, target_state, range_m = cand_idx, cand_state, cand_range
                break
    if target_state is None:
        target_idx, target_state, range_m = _simple_nearest_target(obs, num_blue, num_red)

    if target_state is not None:
        ao = float(target_state[3]) * np.pi if target_state.size > 3 else 0.0
        desired_heading = _wrap_pi(our_heading + ao)
        _simple_last_seen_bearing[blue_id] = desired_heading
        _simple_lost_steps[blue_id] = 0
        delta_alt = float(target_state[2]) * 10000.0 if target_state.size > 2 else 0.0
        pitch = np.clip(delta_alt / max(float(range_m or 300.0), 300.0) * 2.0 + _TRIM_BASELINE, -0.20, 0.25)
        if alt_m < SAFE_COMBAT_ALT:
            pitch = max(float(pitch), 0.0)
        action = _simple_rescale_absolute_heading(float(pitch), desired_heading, 0.8)
        return _record("current_target", target_idx, target_state, range_m, desired_heading, action)

    if blue_id in _simple_last_seen_bearing and _simple_lost_steps.get(blue_id, 0) < _SIMPLE_REACQUIRE_STEPS:
        _simple_lost_steps[blue_id] = _simple_lost_steps.get(blue_id, 0) + 1
        desired_heading = _simple_last_seen_bearing[blue_id]
        pitch = np.clip((SAFE_COMBAT_ALT - alt_m) / 2000.0, -0.10, 0.15) + _TRIM_BASELINE
        action = _simple_rescale_absolute_heading(float(pitch), desired_heading, 0.8)
        return _record("reacquire_last_seen", None, None, None, desired_heading, action, reacquire=True)

    _simple_last_seen_bearing.pop(blue_id, None)
    _simple_lost_steps.pop(blue_id, None)
    source = "center_cruise" if own_position is not None else "hold_heading"
    desired_heading = center_heading if own_position is not None else our_heading
    pitch = np.clip((SAFE_COMBAT_ALT - alt_m) / 2000.0, -0.10, 0.15) + _TRIM_BASELINE
    action = _simple_rescale_absolute_heading(float(pitch), desired_heading, 0.6)
    return _record(source, None, None, None, desired_heading, action)


def _target_track_quality(tgt_vec: np.ndarray) -> str:
    """Return 'radar', 'awacs', or 'invalid' for an enemy entity vector.

    In the current environment observation, radar tracks have TA > 0, while
    AWACS coarse blind-zone tracks retain body-frame position/range but have
    TA == 0 and masked target speed/attitude.  Speed must not be used to reject
    AWACS tracks because V_tgt == 0 is expected for coarse observations.
    """

    vec = np.asarray(tgt_vec, dtype=np.float32)
    if vec.ndim != 1 or vec.shape[0] < 6 or np.allclose(vec, 0.0):
        return "invalid"
    R = float(vec[5]) * 80000.0
    if R < 1.0:
        return "invalid"
    TA = abs(float(vec[4]) * np.pi)
    return "radar" if TA > 1e-4 else "awacs"


def _target_selection_score(tgt_vec: np.ndarray) -> float:
    """Score radar and AWACS-coarse target tracks for Blue assignment."""

    quality = _target_track_quality(tgt_vec)
    if quality == "invalid":
        return 0.0
    R = float(tgt_vec[5]) * 80000.0
    AO = float(tgt_vec[3]) * np.pi
    TA = abs(float(tgt_vec[4]) * np.pi)
    if quality == "radar":
        TA_eff = max(TA, np.deg2rad(5))
        quality_weight = 1.0
    else:
        TA_eff = np.deg2rad(5)
        quality_weight = 0.6
    ao_weight = max(0.1, 1.0 - abs(AO) / np.pi)
    return float(quality_weight * (1.0 / max(R, 300.0)) * ao_weight * (TA_eff / np.pi))


# ==============================================================================
#  blue_coordinated_actions —— 协同目标分配入口（推荐调用此函数）
# ==============================================================================

def blue_coordinated_actions(
    blue_obs: dict[str, dict],
    num_blue: int,
    num_red: int,
    engaged_targets: set[str] | None = None,
    own_positions: dict[str, np.ndarray] | None = None,
    own_headings: dict[str, float] | None = None,
    pursuit_mode: str = "delta10",
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

    if pursuit_mode == "safe_pursuit":
        taken: set[int] = set()
        assignments: dict[int, int | None] = {}
        for b_idx, bid in enumerate(blue_ids):
            obs = blue_obs.get(bid, {})
            target_idx, _state, _range = _simple_nearest_target(
                obs, num_blue, num_red, excluded=taken)
            assignments[b_idx] = target_idx
            if target_idx is not None:
                taken.add(target_idx)
        return {
            bid: _blue_simple_pursuit_action_impl(
                blue_obs.get(bid, {}), num_blue, num_red, b_idx,
                forced_target_idx=assignments.get(b_idx),
                own_position=own_positions.get(bid) if own_positions else None,
                own_heading=own_headings.get(bid) if own_headings else None)
            for b_idx, bid in enumerate(blue_ids)
        }

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
                forced_target_idx=None,
                own_position=own_positions.get(bid) if own_positions else None,
                own_heading=own_headings.get(bid) if own_headings else None,
                pursuit_mode=pursuit_mode)
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
            score[b_idx, r_idx] = _target_selection_score(tgt_vec)

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
            forced_target_idx=assignments[b_idx],
            own_position=own_positions.get(bid) if own_positions else None,
            own_heading=own_headings.get(bid) if own_headings else None,
            pursuit_mode=pursuit_mode)
    return actions


# ==============================================================================
#  blue_pursuit_action —— 单机调用（向后兼容）
# ==============================================================================

def blue_pursuit_action(obs: dict, num_blue: int, num_red: int, blue_id: int,
                        missile_warning: bool = False,
                        own_position: np.ndarray | None = None,
                        own_heading: float | None = None,
                        pursuit_mode: str = "delta10") -> np.ndarray:
    """Per-aircraft entry point (legacy — prefer ``blue_coordinated_actions``)."""
    return _blue_pursuit_action_impl(obs, num_blue, num_red, blue_id,
                                     forced_target_idx=None,
                                     own_position=own_position,
                                     own_heading=own_heading,
                                     pursuit_mode=pursuit_mode)


# ==============================================================================
#  _blue_pursuit_action_impl —— 核心自动驾驶仪
# ==============================================================================

def _blue_pursuit_action_impl(
    obs: dict,
    num_blue: int,
    num_red: int,
    blue_id: int,
    forced_target_idx: int | None,
    own_position: np.ndarray | None = None,
    own_heading: float | None = None,
    pursuit_mode: str = "delta10",
) -> np.ndarray:
    """四层状态机自动驾驶仪（优先级从高到低）。

    HARD_DECK      → 强制爬升 +45° + 满油门, 禁止作战 (< 4500 m)
    DESCENT_WARN   → 下坠率过大时预判拉起 (< 5500 m, 下坠率 > 40 m/s)
    STALL PROTECT  → 低速 + 低空组合保护 (< 5000 m, 速度 < 200 m/s)
    ANTI-STALL     → 早期能量管理 (速度 < 250 m/s 且抬头)
    COMBAT         → 带进近前置角的拦截引导 (4500–15000 m, JSBSim 自然升限约束)
    """

    # ---- 读取物理姿态 ----
    if pursuit_mode == "safe_pursuit":
        return _blue_simple_pursuit_action_impl(
            obs, num_blue, num_red, blue_id,
            forced_target_idx=forced_target_idx,
            own_position=own_position,
            own_heading=own_heading,
        )

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
    track_heading = np.arctan2(our_ve, our_vn)
    # Prefer aircraft yaw from sim.get_rpy()[2] for absolute heading actions.
    # Velocity track heading is only the fallback for legacy callers.
    our_heading = float(own_heading) if own_heading is not None else track_heading

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

    def _rescale_absolute_heading(pitch_int, target_heading_abs, vel_int):
        pitch_new = pitch_int
        heading_new = _wrap_pi(float(target_heading_abs)) / np.pi
        vel_new = (90.0 + 100.0 * vel_int) / 306.0
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

    if _should_override_for_boundary_safety(own_position, our_heading):
        alt_error = SAFE_COMBAT_ALT - alt_m
        boundary_pitch = np.clip(alt_error / 2000.0, -0.05, 0.15) + _TRIM_BASELINE
        heading_cmd = _boundary_patrol_heading_command(own_position, our_heading)
        vel_cmd = max(
            _blue_cruise_speed_command(own_position, current_heading=our_heading),
            0.35,
        )
        return _rescale(float(boundary_pitch), float(heading_cmd), float(vel_cmd))

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

    # --- Alive target list ---
    alive_reds_raw = [i for i in range(num_red)
                      if death_mask[num_blue + i] > 0.5]
    # enemy_states[idx][2] is a body-frame / pseudo-up component, not reliable
    # world altitude. True death / low-altitude status is represented by the
    # environment death_mask, so Blue must not discard alive Reds using body z.
    alive_reds = alive_reds_raw

    # --- Target selection (radar tracks and AWACS coarse tracks) ---
    target_idx: int | None = None
    if (forced_target_idx is not None and forced_target_idx in alive_reds
            and _target_track_quality(enemy_states[forced_target_idx]) != "invalid"):
        target_idx = forced_target_idx
    elif alive_reds:
        best_score = 0.0
        for idx in alive_reds:
            tgt_vec = enemy_states[idx]
            score_ = _target_selection_score(tgt_vec)
            if score_ > best_score:
                best_score = score_
                target_idx = idx

    # --- Cruise: all reds either dead or undetected by radar ---
    if target_idx is None:
        _prev_lead_bearing.pop(blue_id, None)
        _prev_heading_cmd.pop(blue_id, None)
        if not alive_reds:
            _last_target_bearing.pop(blue_id, None)
            _lost_target_steps.pop(blue_id, None)
        elif blue_id in _last_target_bearing and _lost_target_steps.get(blue_id, 0) < 50:
            # Short-horizon reacquisition: keep turning toward the last
            # radar/AWACS bearing for about 10 s at the 5 Hz policy rate. This
            # stores bearing only, never Red world position.
            heading_error = (_last_target_bearing[blue_id] - our_heading + np.pi) % (2 * np.pi) - np.pi
            heading_cmd = np.clip(heading_error / np.deg2rad(10.0), -1.0, 1.0)
            _lost_target_steps[blue_id] = _lost_target_steps.get(blue_id, 0) + 1
            alt_error = SAFE_COMBAT_ALT - alt_m
            reacquire_pitch = np.clip(alt_error / 2000.0, -0.10, 0.12) + _TRIM_BASELINE
            return _rescale(float(reacquire_pitch), float(heading_cmd), 0.8)
        alt_error = SAFE_COMBAT_ALT - alt_m
        cruise_pitch = np.clip(alt_error / 2000.0, -0.10, 0.12) + _TRIM_BASELINE
        heading_cmd = _blue_cruise_heading_command(
            obs, blue_id, own_position=own_position,
            current_heading=our_heading)
        vel_cmd = _blue_cruise_speed_command(
            own_position, current_heading=our_heading)
        return _rescale(float(cruise_pitch), float(heading_cmd), float(vel_cmd))

    tgt = enemy_states[target_idx]

    # De-normalise
    delta_alt = float(tgt[2]) * 10000.0
    AO        = float(tgt[3]) * np.pi        # radians, signed: +right/−left
    TA        = float(tgt[4]) * np.pi        # radians, unsigned
    R         = float(tgt[5]) * 80000.0
    V_tgt     = float(tgt[6]) * 600.0        # target speed, m/s
    quality = _target_track_quality(tgt)

    if quality == "awacs":
        # AWACS coarse tracks provide body-frame bearing/range but not target
        # heading or speed. Do not run velocity lead pursuit on masked data.
        pitch_cmd = np.clip(delta_alt / max(R, 300.0) * 2.0, -0.20, 0.25)
        if alt_m < _DIVE_FREEZE_ALT:
            pitch_cmd = max(pitch_cmd, 0.0)
        elif alt_m < SAFE_COMBAT_ALT:
            pitch_cmd = max(pitch_cmd, -0.25)
        pitch_cmd = float(np.clip(pitch_cmd + _TRIM_BASELINE,
                                  -_COMBAT_PITCH_LIMIT, _COMBAT_PITCH_LIMIT))
        heading_cmd = float(np.clip(AO / (np.pi / 3), -1.0, 1.0))
        vel_cmd = 1.0 if R > 5000.0 else 0.6
        _prev_lead_bearing.pop(blue_id, None)
        _prev_heading_cmd.pop(blue_id, None)
        _last_target_bearing[blue_id] = float((our_heading + AO + np.pi) % (2 * np.pi) - np.pi)
        _lost_target_steps[blue_id] = 0
        if pursuit_mode == "safe_pursuit":
            desired_heading = _wrap_pi(our_heading + AO)
            return _rescale_absolute_heading(pitch_cmd, desired_heading, float(vel_cmd))
        return _rescale(pitch_cmd, heading_cmd, float(vel_cmd))

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
    _last_target_bearing[blue_id] = float(lead_bearing)
    _lost_target_steps[blue_id] = 0
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

    if pursuit_mode == "safe_pursuit":
        return _rescale_absolute_heading(pitch_cmd, float(lead_bearing), vel_cmd)

    return _rescale(pitch_cmd, heading_cmd, vel_cmd)
