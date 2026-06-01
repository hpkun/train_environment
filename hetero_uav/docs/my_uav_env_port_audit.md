# my_uav_env Port Audit

## Scope

This audit compares the parent-project `my_uav_env` BRMA-MAPPO reproduction environment with the current `hetero_uav/uav_env/JSBSim` skeleton. The goal of this port is to keep a reliable BRMA baseline environment inside `hetero_uav` without importing the parent project at runtime.

## Already Equivalent

- High-level action shape is conceptually aligned: both environments expose `[pitch, heading, speed]` commands.
- Both environments have JSBSim-backed aircraft wrappers and a simple kinematic fallback path in `hetero_uav`.
- Both environments maintain aircraft identity, alive/dead status, team membership, missiles, observations, rewards, and episode termination.

## Functionally Similar but Different Code

- `my_uav_env/env.py` is a monolithic BRMA combat environment with integrated JSBSim aircraft, PID control, missile launch checks, lock delay, missile warning, reward components, and Tacview rendering.
- `hetero_uav/uav_env/JSBSim/tasks/hetero_combat_task.py` splits the same broad responsibilities across task, sensor, missile, observation, reward, and termination modules.
- `my_uav_env/simulator.py` contains the mature JSBSim aircraft and proportional-navigation missile simulators. The current hetero skeleton has separate JSBSim aircraft code plus a migrated missile manager, but the integration surface is not equivalent.
- `my_uav_env/pid_controller.py` is the baseline controller used by the original environment. `hetero_uav/uav_env/JSBSim/core/controller.py` is a smaller high-level controller designed for the hetero skeleton.
- `my_uav_env/alignment/*.py` contains paper-aligned geometry, strict observation/state helpers, launch-quality diagnostics, and reward utilities. The hetero skeleton has independent observation and reward builders.

## Must Be Copied Directly First

- `env.py`
- `simulator.py`
- `pid_controller.py`
- `catalog.py`
- `utils.py`
- `render_tacview.py`
- `alignment/`
- `data/`

These files are copied into `hetero_uav/uav_env/brma_env/` with imports rewritten to package-local relative imports so `hetero_uav` can be moved independently.

## Requires Later Heterogeneous Adaptation

- Aircraft construction in `brma_env.env.UavCombatEnv` currently assumes homogeneous F-16 aircraft. Later adaptation should introduce `aircraft_type_params`, with MAV using A-4 and UAV roles using F-16-derived types.
- Observation and global-state builders need type and role features if `brma_env` becomes the heterogeneous training environment.
- Reward logic needs role-aware terms for MAV survival, attack, scout, and interceptor behavior.
- The action/controller interface should preserve the BRMA high-level command semantics while allowing model-specific control signs and performance limits.
- Missile and sensor logic should remain close to the copied BRMA implementation unless a specific heterogeneous requirement forces a change.

## Recommendation

Keep the copied `brma_env` runnable as an original BRMA compatibility baseline first. Use it as the reference implementation for future heterogeneous work, either by extending `brma_env` directly or by migrating its reliable JSBSim, PID, missile, sensor, and diagnostics modules into `HeteroUAVEnv` in small verified steps.
