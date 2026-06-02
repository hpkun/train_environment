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

If `enemy_observed_mask` is 0, the adapter keeps that enemy entity feature and
source as zero.

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
