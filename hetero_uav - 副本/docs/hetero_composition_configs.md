# Heterogeneous Composition Configs

Formal JSBSim heterogeneous composition configs live under
`uav_env/JSBSim/configs/`.

The existing `hetero_2v2_mav_attack.yaml` is retained as a debug alias. The
formal train config is `hetero_train_2v2_mav_attack.yaml`.

## Composition Matrix

| Config | Red Team | Blue Team | Purpose |
| --- | --- | --- | --- |
| `hetero_train_2v2_mav_attack.yaml` | 1 MAV + 1 attack UAV | 2 attack UAV | Train composition |
| `hetero_test_3v3_mav_2attack.yaml` | 1 MAV + 2 attack UAV | 3 attack UAV | Larger same-role test |
| `hetero_test_3v3_mav_attack_scout.yaml` | 1 MAV + 1 attack UAV + 1 scout UAV | 3 attack UAV | Scout role test |
| `hetero_test_3v3_mav_attack_interceptor.yaml` | 1 MAV + 1 attack UAV + 1 interceptor UAV | 3 attack UAV | Interceptor role test |

All configs use:

- `env_type: jsbsim_hetero`
- `sim_freq: 60`
- `agent_interaction_steps: 12`
- `max_steps: 20`
- `enable_gcas_for_blue: true`
- `suppress_jsbsim_output: true`

## Type Defaults

| Type | Role | Aircraft Model | Missiles | Init Altitude Offset | Init Speed Offset |
| --- | --- | --- | --- | --- | --- |
| `mav` | `mav` | `A-4` | 2 | `2000.0 m` | `0.0 m/s` |
| `attack_uav` | `attack_uav` | `f16` | 2 | `0.0 m` | `0.0 m/s` |
| `scout_uav` | `scout_uav` | `f16` | 0 | `0.0 m` | `0.0 m/s` |
| `interceptor_uav` | `interceptor_uav` | `f16` | 2 | `0.0 m` | `0.0 m/s` |

`scout_uav` is currently modeled as an F-16 with zero missiles so the first
composition tests can isolate role/type observation plumbing without changing
reward, missile, action, PID, or termination logic.

`interceptor_uav` is currently modeled as an F-16 with two missiles. Its role is
exposed in observation metadata, but no interceptor-specific reward or control
logic is added at this stage.

## Observation Expectations

`HeteroUavCombatEnv` exposes type/role metadata fields only in the hetero
environment:

- `ego_type`, `ego_role`: shape `(4,)`
- `ally_types`, `ally_roles`: shape `(max_allies, 4)`
- `enemy_types`, `enemy_roles`: shape `(max_enemies, 4)`

For the 3v3 test configs, `red_0` has:

- `ally_types`: shape `(2, 4)`
- `enemy_types`: shape `(3, 4)`

The original `UavCombatEnv` / `jsbsim_brma` observation is unchanged.

## Verification

Run:

```bash
python scripts/diagnose_hetero_compositions.py
pytest tests/test_jsbsim_hetero_compositions.py
```

MAPPO training should not start until these composition configs and hetero
observation fields pass smoke tests.

This stage does not change reward, missile, PID, termination, action, aircraft
XML, or add MAV GCAS.
