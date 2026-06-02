# JSBSim BRMA Paper Alignment

## jsbsim_brma Baseline

`env_type: jsbsim_brma` uses the BRMA-style `UavCombatEnv` under `uav_env/JSBSim`.
It preserves the core environment assumptions from the BRMA-MAPPO reproduction
environment:

- Action space is the 3D high-level command `[target_pitch, target_heading, target_velocity]`.
- Velocity target range is `102-408 m/s`, corresponding to the BRMA high-level speed interval.
- JSBSim provides aircraft dynamics and `PIDController` maps high-level commands to aileron, elevator, rudder, and throttle commands.
- Missile launches keep the original BRMA lock delay and cooldown logic.
- Missile launch gates keep range, angle-off, target-aspect, and engaged-target checks.
- Radar/sensor logic keeps the BRMA field-of-view assumptions, including azimuth/elevation gates and RCS-based range.
- Battlefield constraints keep the BRMA boundary, altitude, overload, and speed checks.
- Blue GCAS remains available as the original scripted safety layer.
- Missile warning remains a scripted observation/control input for evasion behavior.
- Observation remains the original dict structure: `ego_state`, `ally_states`, `enemy_states`, `death_mask`, `missile_warning`, `altitude`, and `velocity`.
- Reward, termination, launch diagnostics, missile termination diagnostics, and Tacview support are preserved.

## jsbsim_hetero Minimal Extension

`env_type: jsbsim_hetero` extends the BRMA environment only at aircraft creation
and metadata level:

- `red_0` uses `A-4` and is tagged as `mav`.
- `red_1` uses `f16` and is tagged as `attack_uav`.
- `blue_0` and `blue_1` use `f16` and are tagged as `attack_uav`.
- `agent_types`, `agent_roles`, and `agent_models` are added to `info`.
- Per-type missile counts are supported.

The current heterogeneous version intentionally does not change:

- observation;
- reward;
- missile dynamics or launch rules;
- action space;
- PID controller;
- termination.

## Next Steps

1. Finish stability validation for the formal homogeneous and heterogeneous JSBSim environments.
2. Add type and role explicitly to observations after the baseline remains stable.
3. Add a plain MAPPO baseline to verify trainability.
4. Only after that, introduce type-aware attention, mask generators, or other algorithm-side changes.
