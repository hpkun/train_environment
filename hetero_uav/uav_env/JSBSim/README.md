# JSBSim Environment

This directory is now the formal BRMA-style JSBSim environment path for `hetero_uav`.

The implementation is based on the reliable BRMA environment port kept in
`uav_env/brma_env`, but this package is self-contained at runtime. Code under
`uav_env/JSBSim` must not import the parent project `my_uav_env` or the backup
package `uav_env.brma_env`.

Primary entry points:

- `uav_env.JSBSim.envs.uav_combat_env.UavCombatEnv`: original homogeneous F-16 BRMA baseline.
- `uav_env.JSBSim.envs.hetero_uav_combat_env.HeteroUavCombatEnv`: minimal MAV/UAV extension that changes only aircraft model, role, and missile count.

The older early skeleton modules under `core/`, `tasks/`, and legacy env wrappers are retained only for compatibility with existing smoke tests and scripts. The recommended runtime path is `env_type: jsbsim_brma` or `env_type: jsbsim_hetero` through `uav_env.make_env`.
