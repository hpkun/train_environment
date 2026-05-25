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
| Strict paper observation | `train_attention_mappo.py --obs-adapter strict` uses strict 10-dim actor observations with normalization | P1 |
| Strict observation API | `UavCombatEnv.get_strict_entity_observation()` and `get_strict_team_observations()` exposed; `reset()`/`step()` still return 11-dim engineering Dict | P1 |
| Critic global state | `train_attention_mappo.py --critic-state strict-global` wires strict team global state into critic; `--critic-state engineering` keeps legacy flattened obs | P1 — needs training validation |
| Global state candidate | `global_state.py` wired into attention training via `--critic-state strict-global` (2v2 dim=88 vs engineering 106) | P1 |
| Blue rule policy | No-target cruise boundary patrol tuned: starts ~12km before boundary (was 18km), heading gain pressure-scaled (gentle early, strong near edge). Combat / target selection unchanged | P2 |
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
6. **BRMA preparation pass 1** adds standalone `brma.mask_generator`
   infrastructure for type-aware / uniform random masks. It is still not wired
   into `AttentionActor` or `train_attention_mappo.py`.
7. **Only after paper mask formulas are verified** proceed to
   MaskVectorGenerator integration / BRMA-MAPPO.

## 6. Blue no-target cruise boundary patrol

The old blue no-target cruise behavior kept `heading_cmd = 0.0`, which means
"keep current heading" in the rule-agent internal convention. If no red target
was selected, Blue could therefore continue straight until it crossed the
battlefield boundary.

`rule_based_agent.py` now provides and the training/evaluation entry points now
pass through a boundary patrol helper for no-target cruise:

- `_boundary_patrol_heading_command(own_position, current_heading, ...)`
- `_blue_cruise_heading_command(obs, blue_id, own_position=None)`

This helper only uses Blue ownship position and velocity-derived heading. It
does not use enemy state and does not give Blue radar-blind target tracking.
`UavCombatEnv.get_blue_own_positions()` returns only alive Blue ownship
positions and does not expose Red positions. Remaining Blue policy items still
need separate audit: AWACS fallback, `DOOMED_ALT` body-frame z handling, and
whether the full rule policy matches the paper baseline.

The no-target patrol is pre-boundary and speed-aware: it starts applying
center-turn pressure before the 40 km battlefield edge and reduces cruise
speed as boundary pressure increases. It still only uses Blue ownship position
and does not impose a hard boundary, teleport, bounce, or termination rule.

## 7. ACMI battlefield boundary debug

Tacview ACMI battlefield boundary visualization is available as an opt-in debug
flag in `eval_acmi.py`:

```powershell
python eval_acmi.py ... --draw-boundary --boundary-half-size 40000
```

It is disabled by default. The current implementation writes four static corner
markers rather than permanent map lines, so normal ACMI exports remain clean.
