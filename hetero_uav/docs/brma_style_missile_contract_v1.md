# BRMA-Style Missile Contract v1

## Goal

This contract aligns the project missile environment closer to the
BRMA-MAPPO missile contract while keeping the current environment structure.
It is not a full reproduction of BRMA missile propulsion or aerodynamic
parameters.

The change is limited to:

- PN lateral guidance for the scripted missile.
- Configurable scripted missile-warning evasion for red, blue, both, or none.
- Event-level diagnostics for missile termination and evasion.

## Paper Alignment

The launch decision remains controlled by the environment fire-control script.
The learning policy does not output a fire action and does not directly control
missile evasion.  This follows the BRMA-style contract where missile launch and
missile escape commands are preset scripts while MARL learns maneuvering.

The retained hit probability is:

```text
P_hit = 0.05 + 0.95 * max(0, Vm dot LOS / (|Vm| |LOS|))
```

PN guidance structure follows the BRMA-MAPPO missile-guidance contract.  `K=3`
and `30g` are kept from the current project configuration and are also
supported by TAM-HAPPO.  The scalar missile speed is treated as a project
environment parameter, not as a full reproduction of BRMA's thrust/drag speed
equation.

## Preserved Behavior

This contract preserves:

- BRMA-style launch gate.
- Continuous lock delay.
- Launch cooldown.
- 3-9 line rear-hemisphere launch rule.
- Engaged-target deconfliction.
- Kill cooldown / multi-kill blocking.
- Existing reward modes.
- Existing action and observation dimensions.
- Existing PID and aircraft XML.

No missile launch, hit, dodge, fire, lock, or guided reward is added.

## Changed Behavior

### PN Lateral Guidance

When `missile_guidance.mode: pn` is configured, the missile computes:

```text
r = target_pos - missile_pos
v_rel = target_vel - missile_vel
R = |r|
r_hat = r / R
closing_speed = -dot(v_rel, r_hat)
omega_los = cross(r, v_rel) / R^2
a_cmd = K * closing_speed * cross(omega_los, missile_dir)
```

The acceleration is projected perpendicular to missile velocity and clamped by
`max_overload_g`.  The missile still uses constant scalar speed:

```text
velocity = unit(velocity + a_cmd * dt) * speed_mps
```

### Scripted Evasion

The config block is:

```yaml
missile_evasion:
  mode: brma_scripted
  teams: both
```

Supported `teams` values:

- `both`
- `red_only`
- `blue_only`
- `none`

The scripted evasion maneuver is BRMA-style scripted missile-warning evasion;
BRMA specifies preset missile escape commands but does not disclose exact
maneuver angles.  The configured `25/60/30/15` degree values are project
scripted-evasion parameters.

### Threat Selection

Incoming missile selection uses `under_missiles`, filters non-closing missiles,
and selects the missile with the smallest estimated time-to-go:

```text
t_go = range / closing_speed
```

## Diagnostics

Missile termination records include:

- `min_range_m`
- `directional_match_at_hit_check`
- `P_hit_at_hit_check`
- `speed_at_termination_mps`
- `closing_speed_at_termination_mps`

Evasion records are written to `missile_events.csv` with:

- `event_type=evasion`
- `evasion_triggered`
- `evasion_team`
- `evasion_agent_id`
- `incoming_missile_id`
- `incoming_range_m`
- `incoming_closing_speed_mps`
- `incoming_t_go_sec`
- `evasion_mode`

## New Configs

- `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_brma_role_no_missile_reward_v8_pn_missile.yaml`
- `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_dynamics_f22_visual_mav_brma_role_no_missile_reward_v8_pn_missile.yaml`

Both explicitly set:

```yaml
missile_guidance:
  mode: pn
  navigation_gain: 3.0
  max_overload_g: 30.0
  speed_mode: constant_scalar
  speed_mps: 600.0

missile_evasion:
  mode: brma_scripted
  teams: both
```

## Not Implemented

This contract does not implement:

- BRMA thrust/drag active/passive missile speed equation.
- Missile low-speed termination.
- Overshoot termination.
- New action or observation fields.
- New active missile reward.
- TAM-HAPPO dodge reward.
- Full missile state in actor observation.

## Smoke Command

```powershell
python -u scripts\train_happo_reference.py `
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_brma_role_no_missile_reward_v8_pn_missile.yaml `
  --output-dir outputs/smoke_brma_v8_pn_missile_2k `
  --total-env-steps 2048 `
  --rollout-length 256 `
  --num-envs 1 `
  --max-steps 1000 `
  --device cpu `
  --policy-arch brma_recurrent_masked `
  --opponent-policy brma_rule_safe_pursuit `
  --eval-during-training `
  --eval-interval-steps 1024 `
  --train-eval-episodes 2 `
  --enable-rich-logging `
  --rich-log-dir outputs/smoke_brma_v8_pn_missile_2k/rich_logs
```

If Windows hits duplicate OpenMP runtime initialization, set
`KMP_DUPLICATE_LIB_OK=TRUE` only for that smoke process.  Do not write it into
project code.

## Result Interpretation

Old v8 runs and new `_pn_missile` runs are not directly comparable as the
environment missile guidance/evasion contract changed.  Compare them as
separate environment contracts.
