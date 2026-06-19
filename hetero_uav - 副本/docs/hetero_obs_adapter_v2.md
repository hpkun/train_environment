# HeteroObsAdapterV2

`HeteroObsAdapterV2` consumes `observation_mode = "mav_shared_geo"` raw
observations. It is an incremental adapter; v1 remains available for
`brma_sensor` observations.

## Dimensions

- `max_red = 5`
- `max_blue = 4`
- `max_allies = 4`
- `max_enemies = 4`
- `ego_feature_dim = 12`
- `ally_entity_dim = 9`
- `enemy_entity_dim = 7`
- `flat_actor_obs_dim = 96`
- `critic_state_dim = 480`

## Flattening Order

```text
flat_actor_obs =
  ego_feature
  + ally_entities flattened
  + enemy_entities flattened
  + ally_valid_mask
  + ally_alive_mask
  + enemy_valid_mask
  + enemy_alive_mask
  + enemy_observed_mask
```

`ego_feature`:

```text
ego_geo_state(7) + ego_role(4) + missile_warning(1)
```

`ally_entity`:

```text
ally_geo_state(5) + ally_role(4)
```

`enemy_entity`:

```text
enemy_geo_state(5) + enemy_track_source(2)
```

## Source Priority

The raw environment applies:

```text
direct observation > MAV shared observation > unavailable
```

`enemy_track_source` encodes:

- own/direct: `[1, 0]`
- MAV shared: `[0, 1]`
- unavailable: `[0, 0]`

`alive_mask` and `observed_mask` are distinct. An alive but unobserved enemy is
legal:

```text
valid=1, alive=1, observed=0
```

For that case, the adapter keeps `enemy_alive_mask=1` but keeps the enemy entity
feature and source at zero because the actor does not receive that enemy's
geometry. Dead real enemies use `valid=1, alive=0, observed=0`; padding enemies
use `valid=0, alive=0, observed=0`.

The adapter uses raw `ally_alive_mask` and `enemy_alive_mask` from the
environment. It does not infer alive status from geometry or observed status.

## Difference From V1

V1 consumes BRMA-style `ego_state`, `ally_states`, and `enemy_states`, producing
a 140-dimensional actor input and 700-dimensional critic state.

V2 consumes geometric fields:

- `ego_geo_state`
- `ally_geo_states`
- `enemy_geo_states`
- `enemy_observed_mask`
- `enemy_track_source`

It produces a 96-dimensional actor input and 480-dimensional critic state.

## Boundaries

V2 does not change action, missile, evasion, reward, termination, aircraft XML,
or the MAPPO algorithm.

The default MAV missile count is 0. Armed MAV scenarios must configure missiles
explicitly in YAML.
