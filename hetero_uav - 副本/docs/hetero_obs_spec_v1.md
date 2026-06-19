# HeteroObsSpec v1

## 1. Research Target

The target is **heterogeneous UAV/MAV composition zero-shot transfer**:
train on one red-blue composition, evaluate on a different composition without
retraining.  This is NOT a full reproduction of TAM-HAPPO or BRMA-MAPPO.  It
uses BRMA-style high-level actions and environment backend, with TAM-HAPPO's
MAV/UAV heterogeneous role concept.

Blue team is assumed to be mostly homogeneous UAVs in the paper setting.
Heterogeneity is primarily on the red side (MAV + UAV mixed formations).
Therefore the v1 actor adapter does not use `enemy_types` / `enemy_roles`.

## 2. Raw Environment Observation

HeteroUavCombatEnv currently returns these keys per agent:

- `ego_state`  (11,)
- `ally_states` (max_allies, 11)
- `enemy_states` (max_enemies, 11)
- `death_mask`   (max_allies + max_enemies,)
- `missile_warning` (1,)
- `altitude`        (1,)
- `velocity`        (3,)
- `ego_type`        (4,) — one-hot over [mav, attack_uav, scout_uav, interceptor_uav]
- `ego_role`        (4,) — one-hot
- `ally_types`      (max_allies, 4)
- `ally_roles`      (max_allies, 4)
- `enemy_types`     (max_enemies, 4)
- `enemy_roles`     (max_enemies, 4)

## 3. HeteroObsAdapter v1 — Actor Input Fields

### Used fields

**ego block:**
| Field            | Dim | Notes                              |
|------------------|-----|------------------------------------|
| ego_state        | 11  | body-frame engineering entity      |
| ego_role         | 4   | one-hot role                       |
| missile_warning  | 1   | 0 or 1                             |
| altitude         | 1   | metres, normalised /10000 in adapter |
| velocity         | 3   | m/s, normalised /600 in adapter    |
| **ego total**    |**20**|                                  |

**ally entity block:**
| Field       | Dim | Notes              |
|-------------|-----|--------------------|
| ally_state  | 11  | per-ally state     |
| ally_role   | 4   | one-hot role       |
| **per-ally**|**15**|                  |

**enemy entity block:**
| Field        | Dim | Notes             |
|--------------|-----|-------------------|
| enemy_state  | 11  | per-enemy state   |
| **per-enemy**|**11**|                 |

### Explicitly excluded in v1

- `enemy_types` / `enemy_roles` — Blue is mostly homogeneous UAV; constant info.
- `ego_type` / `ally_types` — role already encodes type for current vocab.
- Capability vector (missile_left_norm, max_speed, radar_range).
- Incoming missile states — TAM-HAPPO treats them as entity j; v2.
- Reward components.
- GCAS state.
- Hidden missile cooldown / lock timer.

## 4. Padding Rule

Target paper-scale composition: **max_red = 5, max_blue = 4**.

For a red agent:
- max_allies = max_red − 1 = **4**
- max_enemies = max_blue = **4**

## 5. Flat Actor Dimension

```
ego_feature_dim   = 11 + 4 + 1 + 1 + 3       = 20
ally_entity_dim   = 11 + 4                    = 15
enemy_entity_dim  = 11                        = 11
mask_dim          = 4 + 4 + 4 + 4             = 16

flat_actor_obs_dim = 20 + 4×15 + 4×11 + 16   = 140
```

## 6. Structured Actor Observation

For attention / entity-encoder use:

- `ego_feature`: (20,)
- `ally_entities`: (4, 15)
- `enemy_entities`: (4, 11)
- `ally_valid_mask`: (4,)
- `ally_alive_mask`: (4,)
- `enemy_valid_mask`: (4,)
- `enemy_alive_mask`: (4,)

Padding rules:
- Missing ally/enemy slots filled with zero vectors.
- Corresponding valid/alive mask entries set to 0.
- Dead agents have alive_mask = 0 but valid_mask = 1 if the slot is occupied.

## 7. Critic State v1

First-version critic state = concatenation of flat actor observations for all
red controlled agents, padded to `max_red`:

```
flat_actor_obs_dim = 140
max_red            = 5
critic_state_dim   = 140 × 5 = 700
```

Vacant red slots are filled with zeros; a `red_valid_mask` of length 5
indicates which slots are active.

## 8. Why enemy_types / enemy_roles Are Excluded in v1

- In the TAM-HAPPO paper-style setting, blue aircraft are primarily
  homogeneous UAVs.
- `enemy_types` / `enemy_roles` provide constant information that does not
  vary across blue agents in the default scenario.
- Excluding them reduces adapter complexity without loss of signal for the
  first baseline.
- The raw environment continues to expose them for future enemy-heterogeneous
  extensions.

## 9. Why Capability Vector Is Postponed

- TAM-HAPPO observation formulas do not contain an explicit capability vector.
- For v1 the role one-hot already distinguishes mav / attack_uav / scout /
  interceptor.
- If scout/interceptor roles prove insufficient to express capability
  differences, a capability vector (missile_left_norm, max_speed,
  radar_range) can be added in v2.

## 10. Why Incoming Missile State Is Postponed

- TAM-HAPPO can treat incoming missiles as entity j with position/velocity.
- The current BRMA environment provides `missile_warning` (0/1) and
  scripted missile evasion.
- v1 uses `missile_warning` only.
- v2 can add `incoming_missile_states` / `incoming_missile_mask` as
  additional entity blocks.

## 11. Implementation Roadmap

1. HeteroObsSpec v1 document (this file).
2. HeteroObsAdapter — pure-function adapter from raw obs to spec.
3. Adapter diagnostics — shape checks, padding tests.
4. Plain MAPPO baseline using the adapter.
5. Zero-shot composition evaluation.
6. Attention / entity-encoder extension.
7. Type-aware / role-aware encoder extension.
8. GRU / HAPPO-style algorithms as follow-up.
