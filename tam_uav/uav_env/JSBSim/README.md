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
# TAM paper environment v5 fidelity contract

The formal TAM configurations use revision
`published_environment_reconstruction_v5`. Published values are isolated in
`published_parameters`; indispensable engineering choices are explicit in
`unpublished_parameters`; only the unambiguous `12 / 60 = 0.2 s` decision
interval is stored under `derived_parameters`.

Matched paper-visible contracts are the 60 Hz/5 Hz loop, 4D direct-FCS
40-bin action, aircraft and missile limits listed in the paper, role missile
counts, proportional-navigation gains, and Table 1 reward weights/events.
All alive fixed-slot aircraft are observable; no 14/28 km detection gate is
used. One target snapshot is frozen per decision step for observation ordering,
blue action selection, weapon launch, reward, and diagnostics.

This is not an exact environment reproduction. The paper does not publish the
complete height sub-functions, reference-[8] blue FSM/action mapping, JSBSim
carrier model, missile propulsion/drag/hit/timeout details, combat-zone radius,
tie breaks, or complete simultaneous-event/termination semantics. These remain
explicit unresolved assumptions. F16/F22 XML names are execution substrates,
not paper-specified aircraft identities. Blue candidates are compared through
independent 12-frame JSBSim clones; the minimal action mapping remains marked
unpublished and `reference_8_exact_blue_fsm_reproduced=false`.

Removed v4 assumptions include attack-range-based observation gates and the
presentation of 2x range boundaries, MAV distance thresholds, and `2R/V`
missile timeout as derived paper facts. The retained height implementation is
reported as `unpublished_height_reward_approximation` with
`height_reward_exact_formula_available=false`.
