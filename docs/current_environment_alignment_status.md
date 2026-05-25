# Current environment alignment status

Last updated: after situation-reward 3D body-x q_LOS switch.

## 1. Current reward / environment version

```python
REWARD_VERSION = "fixed_ta_alt_eq17_3dlos_v1"
```

This version includes:

1. **fixed Ta curve** — continuous, non-negative, normalized `[0, 1]` angle-advantage function
   (`ta_angle_advantage_fixed`).
2. **pairwise eq.17-style altitude reward** — mean of `altitude_reward_paper_eq17` over
   alive enemies, with high-altitude `0.1` tail.
3. **3D body-x q_LOS situation reward** — `_situation_reward()` uses
   `compute_body_x_q_los(ego_pos, ego_rpy, enemy_pos)` instead of the old 2D horizontal
   `get2d_AO_TA_R`.
4. **3D Euclidean distance in situation reward** — `compute_3d_range` replaces the old
   horizontal-only `R`.
5. **signed AO collinear fix** — `_make_entity_vec` uses `_signed_ao_from_unsigned_and_side`
   so a target directly behind (unsigned AO ≈ π) is no longer collapsed to 0 in the 11-dim
   entity observation vector.

Older reward versions (`fixed_ta_v1`, `fixed_ta_alt_eq17_v1`, and legacy pre-pass19 logs)
should **not** be mixed with `fixed_ta_alt_eq17_3dlos_v1` results.

## 2. Aligned or mostly aligned items

| Item | Paper reference | Status |
|---|---|---|
| JSBSim F-16 flight dynamics | §2.2 | Shared engine for both teams |
| PID Bank-to-Turn high-level action | §2.4 | Three-loop roll/pitch/velocity PID with gimbal protection, heading LPF, anti-inversion |
| Action range pitch/heading/velocity | §2.4 | pitch ±90°, heading ±180°, velocity 102–408 m/s |
| Missile cooldown | 0.5 s | `missile_cooldown_frames` scaled with `sim_freq` |
| Missile lock delay | 0.25 s | `missile_lock_delay_frames` scaled with `sim_freq` |
| Missile hit probability | directional | Uses missile velocity vs LOS dot product; `MissileSimulator` resolves hit/miss |
| Radar FOV | ±60° azimuth, [-10°,+32°] elevation | `_is_detected_by_radar` checks both |
| Radar Rmax | `Rmax = K * RCS^(1/4)` | `_compute_radar_max_range` uses `RADAR_K * rcs^0.25` |
| Boundary reward (eq.18) | `-10` if `\|x\|` or `\|y\|` > 4×10⁴ | `_boundary_penalty` uses `BATTLEFIELD_HALF_SIZE` |
| Roll reward (eq.16) | Dual-condition `\|φ\|>π/4 & \|θ\|>π/4` | `_roll_penalty` matches formula |
| Altitude reward (eq.17-style) | Pairwise relative, quadratic segments | `_altitude_reward` uses `altitude_reward_pairwise_mean_eq17` |
| Situation reward geometry | 3D body-frame LOS | `_situation_reward` uses `compute_body_x_q_los` + `compute_3d_range` |
| Terminal reward (eq.23) | Team-level `r_end`, per-agent share | Computed as `raw_r_end / max_num_team`, sum equals paper value |
| GCAS asymmetry | Blue-only safety net | `enable_gcas_for_blue` flag; training uses `False` |

## 3. Still approximate / not fully aligned

| Item | Gap | Priority |
|---|---|---|
| RCS model | Front/side approximation, not paper table interpolation | P2 |
| Pitch reward (eq.15) | Middle-segment slope needs paper text visual verification | P1 |
| Speed reward (eq.19) | Mach conversion constant (340 m/s) approximate; needs paper verification | P1 |
| Ta scale | Current `fixed_ta_v1` uses `[0, 1]` scale; paper eq.20 may use `10` first segment | P1 — needs ablation, not silent swap |
| q_LOS definition | Current choice is body-x LOS angle; velocity-q candidate exists in `situation_reward_candidates.py` | P1 — pending paper confirmation |
| Observation space | Still 11-dim engineering Dict, not strict Table 1 / Table 2 10-dim | P1 |
| Strict paper observation | `train_attention_mappo.py --obs-adapter strict` can use strict 10-dim actor observations via env worker method calls | P1 |
| Strict observation API | `UavCombatEnv.get_strict_entity_observation()` and `get_strict_team_observations()` exposed; `reset()`/`step()` still return 11-dim engineering Dict | P1 |
| Critic global state | Still flattened red observations concat; `global_state.py` candidate exists (2v2 strict dim=88 vs current 106) | P1 |
| Global state candidate | `global_state.py` provides `build_strict_team_global_state()` / `infer_strict_team_global_state_dim()`; not yet wired into training | P1 |
| Blue rule policy | Engineering implementation; not guaranteed identical to paper script | P2 |
| `num_missiles_per_plane` | Default `999` (no limit); paper does not specify a fixed value | P2 |
| PID stabilisation | Engineering additions (deadband, heading LPF, velocity R_BI, anti-inversion) | P2 |

## 4. Current module layout

```
my_uav_env/
  alignment/
    __init__.py
    reward_utils.py          — REWARD_VERSION, Ta/Td/pitch/speed/altitude helpers
    los_geometry.py           — canonical compute_body_x_q_los, compute_3d_range, etc.
    entity_obs.py             — build_entity_observation, infer_entity_layout
    obs_adapter.py            — 11→10 placeholder adapter (build_paper_entity_observation_from_env_obs)
    state_extractor.py        — strict Table 1/2 extractor prototype (not wired)
    geometry_diagnostics.py   — AO/TA/q_LOS comparison tool
    situation_reward_candidates.py — 2D AO/TA, 3D body-x, 3D velocity candidate formulas
  env.py                      — UavCombatEnv (uses los_geometry, reward_utils)

Root compatibility shims (still retained):
  reward_utils.py             → from my_uav_env.alignment.reward_utils import *
  entity_obs_utils.py         → from my_uav_env.alignment.entity_obs import *
  paper_obs_utils.py          → from my_uav_env.alignment.obs_adapter import *
  paper_state_extractor.py    → from my_uav_env.alignment.state_extractor import *
```

## 5. Recommended next steps

1. **Do not delete root compatibility shims yet.** They protect external imports.
2. **Run a short 2v2 vanilla baseline** under `fixed_ta_alt_eq17_3dlos_v1` to verify the
   3D q_LOS switch does not destabilise training.
3. **Evaluate the trained baseline** with `evaluate_vanilla_mappo.py` to get metrics under
   the new reward version.
4. **Validate strict Table 1/Table 2 attention training locally** with
   `train_attention_mappo.py --obs-adapter strict` or the
   `attention_1v1_strict_smoke` preset.
5. **Design native global state critic** input (paper-style, not just flat concat).
6. **Only then** proceed to MaskVectorGenerator / BRMA-MAPPO.
