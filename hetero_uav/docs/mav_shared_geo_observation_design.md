# MAV-Shared Geometric Observation Design

## Existing BRMA Observation Path

`uav_env/JSBSim/env.py` is the current BRMA-compatible environment base.
Its observation path is:

- `observation_space` defines per-agent dict observations with `ego_state`,
  `ally_states`, `enemy_states`, `death_mask`, `missile_warning`, `altitude`,
  and `velocity`.
- `_get_agent_obs()` builds those fields from `red_planes` and `blue_planes`.
- `_make_entity_vec()` produces the 11-dimensional entity state used by
  `ego_state`, `ally_states`, and full enemy tracks.
- `missile_warning` is derived from `AircraftSimulator.check_missile_warning()`.
- Enemy full-track observation currently calls `_is_detected_by_radar()`.

The BRMA-style sensing path includes explicit radar/FOV/RCS logic:

- azimuth and elevation FOV gates;
- RCS-dependent detection range through `_compute_radar_max_range()`;
- fallback coarse enemy position when the radar gate fails.

`HeteroUavCombatEnv` currently preserves that BRMA raw observation and only adds
type/role metadata fields. `HeteroObsAdapter` v1 is retained for compatibility:
it consumes the BRMA-style raw fields and keeps `flat_actor_obs_dim = 140` and
`critic_state_dim = 700`.

## TAM-HAPPO-Style Observation Abstraction

The heterogeneous TAM-HAPPO direction is modeled here as a higher-level
self-state plus relative-geometry observation abstraction. The source paper's
Eq. 13 is treated as self-state plus relative geometry, not as an explicit
radar/FOV/RCS/infrared sensor model.

This project therefore separates two concepts:

- BRMA sensor mechanics remain in the environment for compatibility and
  ablations.
- The main heterogeneous observation candidate uses abstract direct observation
  and MAV-mediated information sharing.

Sensing uncertainty, communication constraints, delayed sharing, jamming, and
explicit infrared/radar/RCS models are future work.

## Observation Modes

### `observation_mode = "brma_sensor"`

This is the default compatibility mode. It keeps current raw observation
behavior unchanged and is used for debugging, legacy MAPPO smoke tests, and
ablation against the newer observation mode.

### `observation_mode = "mav_shared_geo"`

This is the main experimental observation candidate. It does not use BRMA
radar/RCS/FOV as the actor observation-generation dependency. Instead it adds
geometric fields to the existing raw observation dict, preserving old fields so
v1 adapters and baseline scripts continue to run.

New raw fields:

- `ego_geo_state`: shape `(7,)`
- `ally_geo_states`: shape `(max_allies, 5)`
- `enemy_geo_states`: shape `(max_enemies, 5)`
- `enemy_observed_mask`: shape `(max_enemies,)`
- `enemy_track_source`: shape `(max_enemies, 2)`

## MAV-Mediated Sharing Rule

For each non-MAV red UAV `i` and enemy `j`:

```text
if UAV_i directly observes enemy_j:
    use own relative geometry
    source = own
elif red MAV is alive and MAV observes enemy_j:
    use MAV-shared relative geometry, expressed in UAV_i's relative frame
    source = mav_shared
else:
    enemy geometry = zero
    observed_mask = 0
    source = unavailable
```

For the red MAV itself:

- alive enemy within the MAV observation range is observed;
- source is `own`;
- this is the project abstraction of MAV situation-support, not a literal
  sensor reproduction from the paper.

For blue agents:

- only direct observation is used;
- red MAV sharing is not available to blue;
- source is either `own` or `unavailable`.

Priority is therefore:

```text
direct observation > MAV shared observation > unavailable
```

## Direct Observation Abstraction

The direct observation abstraction is not called radar and does not use RCS/FOV.
The first version uses only a distance threshold:

- `uav_direct_observation_range_m = 10000.0`
- `mav_observation_range_m = 80000.0`

These are environment abstraction parameters, not radar parameters.

## HeteroObsAdapterV2 Spec

The v2 adapter uses `max_red = 5` and `max_blue = 4`.

`ego_feature`:

- `ego_geo_state`: 7, `[x, y, z, speed, pitch, yaw, roll]`, normalized;
- `ego_role`: 4;
- `missile_warning`: 1;
- `ego_feature_dim = 12`.

`ally_entities`:

- `ally_geo_state`: 5, `[delta_v, delta_h, distance, ATA, AA]`, normalized;
- `ally_role`: 4;
- `ally_entity_dim = 9`;
- shape `(4, 9)`.

`enemy_entities`:

- `enemy_geo_state`: 5, `[delta_v, delta_h, distance, ATA, AA]`, normalized;
- `enemy_track_source`: 2, `[own_observed, mav_shared]`;
- `enemy_entity_dim = 7`;
- shape `(4, 7)`.

Masks:

- `ally_valid_mask`: 4;
- `ally_alive_mask`: 4;
- `enemy_valid_mask`: 4;
- `enemy_alive_mask`: 4;
- `enemy_observed_mask`: 4;
- `mask_dim = 20`.

Flattening:

```text
flat_actor_obs_dim = 12 + 4*9 + 4*7 + 20 = 96
critic_state_dim = 96 * 5 = 480
```

## What Is Not Changed

This observation mode does not change:

- action;
- missile;
- evasion;
- reward;
- termination;
- aircraft XML;
- MAPPO algorithm.
