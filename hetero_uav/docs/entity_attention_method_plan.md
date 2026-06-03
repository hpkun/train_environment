# Entity Attention Method Plan

## 1. Motivation

Current V2 `mav_shared_geo` observation provides structured per-entity
features with valid/alive/observed masks.  Composition zero-shot transfer
requires generalising across different numbers of agents and entities.
The flat MLP MAPPO baseline does not exploit this structure.

## 2. Method Goal

Replace the flat actor MLP frontend with an entity attention encoder
that handles variable entity counts via max-padding with masks.

## 3. Input Structure (HeteroObsAdapterV2)

| Field | Shape |
|---|---|
| ego_feature | (12,) |
| ally_entities | (4, 9) |
| enemy_entities | (4, 7) |
| ally_valid_mask | (4,) |
| ally_alive_mask | (4,) |
| enemy_valid_mask | (4,) |
| enemy_alive_mask | (4,) |
| enemy_observed_mask | (4,) |

## 4. Attention Design v1

- Shared entity MLP for ally and enemy entities
- Role/source info retained in per-entity features
- Masked single-head attention or pooling
- Invalid slots masked (valid=0)
- Dead entities masked (alive=0)
- Unobserved enemies (alive=1, observed=0): zero geometric feature,
  retain mask-level context

## 5. Actor Design

1. Encode ego_feature
2. Encode ally entities (shared MLP)
3. Encode enemy entities (shared MLP)
4. Masked attention / pooling over ally and enemy sets
5. Concat ego embedding + ally context + enemy context
6. Output Gaussian action distribution via small MLP head

## 6. Critic Design

- Initial implementation reuses V2 flat critic_state (480-dim) for
  baseline comparison
- Attention critic can be added after actor is validated

## 7. Experimental Sequence

| Stage | Method | Goal |
|---|---|---|
| A | V2 MAPPO flat | baseline |
| B | V2 MAPPO + entity attention actor | actor improvement |
| C | V2 MAPPO + entity attention actor/critic | full attention |
| D | Role-aware ablation | if needed |
| HAPPO | | only after baselines stable |

## 8. What Not to Change

- Action space (3-dim high-level)
- Missile / evasion logic
- Reward / termination
- Aircraft XML
- observation_mode semantics
