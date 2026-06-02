# Heterogeneous Observation Design

This note documents the first type-aware observation extension for
`HeteroUavCombatEnv`.

## Scope

Only `uav_env.JSBSim.envs.hetero_uav_combat_env.HeteroUavCombatEnv` is extended.
The original BRMA-compatible `UavCombatEnv` / `jsbsim_brma` observation space is
unchanged.

No reward, missile, PID, termination, action, aircraft XML, or algorithm logic is
changed by this observation update.

## Why Type And Role Fields Are Needed

The heterogeneous environment can assign different aircraft types and roles to
agents while keeping the BRMA high-level action interface. A shared policy needs
explicit metadata to distinguish MAV, attack UAV, scout UAV, and interceptor UAV
agents from the same numeric state layout.

The added fields provide this metadata without changing the existing BRMA state
vectors:

- `ego_type`: current agent aircraft type one-hot, shape `(4,)`
- `ego_role`: current agent role one-hot, shape `(4,)`
- `ally_types`: ally aircraft type one-hot rows, shape `(max_allies, 4)`
- `ally_roles`: ally role one-hot rows, shape `(max_allies, 4)`
- `enemy_types`: enemy aircraft type one-hot rows, shape `(max_enemies, 4)`
- `enemy_roles`: enemy role one-hot rows, shape `(max_enemies, 4)`

Dead agents keep their type and role metadata. Padding slots remain zero in
future variable-count extensions.

## Vocabulary

Both type and role vocabularies currently use:

```text
["mav", "attack_uav", "scout_uav", "interceptor_uav"]
```

Unknown type or role names encode as all zeros.

## Ordering

The metadata ordering matches the original BRMA observation ordering:

- Red agent allies follow `red_ids` order with the ego agent removed.
- Red agent enemies follow `blue_ids` order.
- Blue agent allies follow `blue_ids` order with the ego agent removed.
- Blue agent enemies follow `red_ids` order.

This keeps `ally_types` aligned with `ally_states` and `enemy_types` aligned
with `enemy_states`.

## Next Step

The intended next training step is a plain MAPPO baseline that consumes these
metadata fields. Type-aware attention, mask generators, HAPPO, and BRMA-MAPPO
extensions are intentionally out of scope for this environment change.
