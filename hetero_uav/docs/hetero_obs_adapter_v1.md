# HeteroObsAdapter v1

## 1. Adapter responsibility

The adapter converts HeteroUavCombatEnv raw dict observations into
fixed-dimension actor and critic inputs as defined by HeteroObsSpec v1.
It does **not** modify the environment, reward, missile, PID, termination,
action, or aircraft XML.

## 2. Input raw obs fields (used)

From HeteroUavCombatEnv per-agent obs dict:

| Field | Shape | Usage |
|---|---|---|
| `ego_state` | (11,) | ego feature |
| `ego_role` | (4,) | ego feature |
| `missile_warning` | (1,) | ego feature |
| `altitude` | (1,) | ego feature, /10000 |
| `velocity` | (3,) | ego feature, /600 |
| `ally_states` | (max_allies, 11) | ally entities |
| `ally_roles` | (max_allies, 4) | ally entities |
| `enemy_states` | (max_enemies, 11) | enemy entities |

Fields NOT used by v1 (but present in raw obs):
`ego_type`, `ally_types`, `enemy_types`, `enemy_roles`, `death_mask`.

## 3. Output: flat actor obs (140,)

```
flat_actor_obs = concat(
    ego_feature,       # 20
    ally_entities,     # 4 × 15 = 60
    enemy_entities,    # 4 × 11 = 44
    ally_valid_mask,   # 4
    ally_alive_mask,   # 4
    enemy_valid_mask,  # 4
    enemy_alive_mask,  # 4
)                     # = 140
```

## 4. Output: structured actor obs

- `ego_feature`: (20,)
- `ally_entities`: (4, 15)
- `enemy_entities`: (4, 11)
- `ally_valid_mask`: (4,)
- `ally_alive_mask`: (4,)
- `enemy_valid_mask`: (4,)
- `enemy_alive_mask`: (4,)

## 5. Output: critic state (700,)

```
critic_state = concat(flat_actor_obs for red_0..red_4, padded to max_red=5)
             = 5 × 140 = 700
```

Vacant red slots are filled with zeros.  `red_valid_mask` identifies active slots.

## 6. alive/valid mask definitions

- `valid_mask[i] = 1` if the i-th entity slot corresponds to a real agent
  (not a padding slot) AND its state vector is non-zero.
- `alive_mask[i] = 1` if valid_mask[i] = 1 (non-zero state = alive).
- Dead/missing agents have state = zeros → valid = 0, alive = 0.
- Non-existent padding slots have valid = 0, alive = 0.

## 7. Why v1 only handles red controlled agents

- The first MAPPO baseline will train a red-team policy against the
  existing blue rule-based agent.
- Blue-team policy training is out of scope for v1.
- `adapt_all(controlled_side="red")` explicitly validates this constraint.

## 8. Why the adapter does not modify the environment

The adapter is a pure function from dict → ndarray.  It reads existing
raw observation keys and produces fixed-dim vectors.  It does not alter
the environment's observation_space, step(), or reset().

## 9. Next step

Plain MAPPO baseline using HeteroObsAdapter for actor and critic inputs.
