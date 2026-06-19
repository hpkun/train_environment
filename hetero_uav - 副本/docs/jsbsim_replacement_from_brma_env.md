# JSBSim Replacement From BRMA Environment

`uav_env/JSBSim` is now the formal environment directory for BRMA-style JSBSim combat environments in `hetero_uav`.

The implementation was copied from the reliable `uav_env/brma_env` port of the parent `my_uav_env` environment, then made self-contained under `uav_env/JSBSim`. The backup `brma_env` package is retained as the original migration snapshot, but runtime code in `uav_env/JSBSim` must not import `uav_env.brma_env` or parent-project `my_uav_env`.

## Entry Points

- `env_type: jsbsim_brma` creates the original homogeneous F-16 BRMA baseline through `uav_env.JSBSim.envs.uav_combat_env.UavCombatEnv`.
- `env_type: jsbsim_hetero` creates the minimal MAV/UAV environment through `uav_env.JSBSim.envs.hetero_uav_combat_env.HeteroUavCombatEnv`.

## Current Heterogeneous Scope

The first `jsbsim_hetero` version changes only:

- aircraft model;
- role metadata;
- missile count.

It intentionally keeps the original BRMA behavior for:

- observation dict;
- reward;
- missile launch, lock delay, and missile warning;
- action space;
- PID control;
- termination;
- Tacview support.

This keeps the BRMA baseline behavior stable while providing a clear place for future MAV/UAV heterogeneous changes.

## Legacy Skeleton

The older early skeleton modules under `uav_env/JSBSim/core`, `uav_env/JSBSim/tasks`, and older wrappers are retained for compatibility with existing smoke tests and scripts. New official environment work should target `uav_env/JSBSim/env.py`, `uav_env/JSBSim/simulator.py`, and `uav_env/JSBSim/envs/`.
