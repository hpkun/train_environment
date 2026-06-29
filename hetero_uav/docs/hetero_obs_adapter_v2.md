# HeteroObsAdapterV2

`HeteroObsAdapterV2` consumes `observation_mode = "mav_shared_geo"` raw
observations. It is a canonical full-geometry adapter; v1 remains available for
`brma_sensor` observations.

## Dimensions (fixed 5V4 capacity)

Training main path uses a **fixed 5V4-capacity adapter**. Small scenarios (e.g.
3V2) are zero-padded to this capacity. Some audit scripts may use native scenario
capacity — audit dimensions and training dimensions can differ and should not be
confused.

- `max_red = 5`
- `max_blue = 4`
- `max_allies = 4`
- `max_enemies = 4`
- `ego_feature_dim = 12`
- `ally_entity_dim = 9`
- `enemy_entity_dim = 18`
- `flat_actor_obs_dim = 140`
- `critic_state_dim = 700`

### Enemy entity composition (18 dims)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 5 | `enemy_geo_states` (compact relative: speed_diff, delta_h, dist, ata, aa) |
| 5 | 2 | `enemy_track_source` (own_sensor=1,0 / mav_shared=0,1 / unavailable=0,0) |
| 7 | 3 | `enemy_relative_pos_xyz` |
| 10 | 3 | `enemy_relative_vel_xyz` |
| 13 | 2 | `enemy_bearing_elevation` |
| 15 | 2 | `enemy_speed_heading` |
| 17 | 1 | `enemy_full_geo_valid_mask` |

### Flat actor obs layout

```text
flat_actor_obs =
  ego_feature(12)
  + ally_entities_flattened(max_allies * 9)
  + enemy_entities_flattened(max_enemies * 18)
  + ally_valid_mask(max_allies)
  + ally_alive_mask(max_allies)
  + enemy_valid_mask(max_enemies)
  + enemy_alive_mask(max_enemies)
  + enemy_observed_mask(max_enemies)
```

### Ego feature (12 dims)

```text
ego_geo_state(7) + ego_role(4) + missile_warning(1)
```

### Ally entity (9 dims)

```text
ally_geo_state(5) + ally_role(4)
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

## Legacy (pre-full-geometry) dimensions

> **Historical note**: The pre-full-geometry V2 adapter used:
> - `enemy_entity_dim = 7` (compact geo only)
> - `flat_actor_obs_dim = 96`
> - `critic_state_dim = 480`
>
> These are **no longer used** by the canonical `mav_shared_geo` training path.
> Checkpoints with these dimensions may still load for backward compatibility
> but will report `full_geometry_features_used = false`.

## Boundaries

V2 does not change action, missile, evasion, reward, termination, aircraft XML,
or the MAPPO algorithm.

The default MAV missile count is 0. Armed MAV scenarios must configure missiles
explicitly in YAML.
