# Entity Policy Staged Audit

This audit keeps the running flat-observation HAPPO experiment intact. The new
entity path is additive and opt-in; it does not replace `HAPPOReferencePolicy`.

## P0 Current State

The current `hetero_uav` HAPPO reference path is:

- Policy class: `algorithms/happo/happo_policy.py::HAPPOReferencePolicy`
- Actor input: flat 96-dimensional `HeteroObsAdapterV2` actor observation
- Critic input: flat 480-dimensional centralized state
- Actor: MLP MAV head and MLP shared-UAV head
- Critic: MLP centralized critic
- Entity encoder: not present in the flat path
- Attention: not present in the flat path
- GRU/recurrent state: not present in the flat path
- Mask generator: not present in the flat path

## Parent BRMA-MAPPO Module Comparison

| Module | Current `hetero_uav` | Parent `brmamappo` | Missing | Reusable code location | Suggested migration stage |
|---|---|---|---|---|---|
| Flat actor observation | `HeteroObsAdapterV2`, 96 dims | Parent has entity-oriented attention path plus legacy flat paths | Partially | `hetero_uav/uav_env/JSBSim/adapters/hetero_obs_adapter_v2.py` | Existing baseline, keep |
| Centralized critic state | 480-dim concatenated flat actor observations | Centralized critic and global observation in MAPPO/attention scripts | No for current baseline | `train_attention_mappo.py`, `train_vanilla_mappo.py` | Existing baseline, keep |
| MAV/UAV actor split | Separate MAV actor and shared UAV actor | Parent mostly homogeneous MAPPO/attention actors | No for hetero baseline | `hetero_uav/algorithms/happo/happo_policy.py` | Existing baseline, keep |
| Entity set adapter | Added minimal wrapper over V2 structured obs | Entity observations are used by attention models | Previously missing | Parent concept: `attention_models.py`, `algorithm/feature_extractor.py` | P1a |
| Entity observation encoder | Added minimal shared entity MLP in new policy | `EntityObservationEncoder` | Previously missing | `algorithm/feature_extractor.py`, `attention_models.py` | P1b/P2 |
| Multi-head attention | Added only in new opt-in actor | `nn.MultiheadAttention` in entity encoders | Previously missing | `algorithm/feature_extractor.py`, `attention_models.py` | P2 |
| Alive/observed attention mask | Added alive/observed keep mask only | Parent BRMA masking includes attention masks and generated masks | Partially | `brma/mask_generator.py`, `algorithm/feature_extractor.py` | P2 now, richer masks later |
| MaskVectorGenerator | Not implemented | `MaskVectorGenerator` | Yes | `algorithm/feature_extractor.py`, `brma/mask_generator.py` | Later; not P1/P2 |
| Biased random mask | Not implemented | BRMA mask generator and rollout schema support | Yes | `brma/mask_generator.py`, `brma/rollout_schema.py`, `brma/collection.py` | Later; after entity path is stable |
| GRUCell | Not implemented | `nn.GRUCell` in actor/critic nets | Yes | `algorithm/mappo_nets.py`, `attention_models.py`, `train_vanilla_mappo.py` | Later P3 |
| Recurrent hidden state maintenance | Not implemented | Hidden states are carried/reset in recurrent MAPPO runners | Yes | `train_vanilla_mappo.py`, `train_attention_mappo.py` | Later P3 |
| Rollout buffer for recurrent/BRMA fields | Current `HAPPORolloutBuffer` stores flat feed-forward data | Parent has recurrent and BRMA-specific storage | Yes | `train_attention_mappo.py::AttentionRolloutBuffer`, `brma/rollout_schema.py` | Later P3/P5 |
| HAPPO sequential correction | Current reference is simplified HAPPO-style actor split | Not the parent BRMA module; full HAPPO correction is not present here | Yes | Requires a separate scoped design | Later P6 |

## P1a Entity Set Adapter

`uav_env/JSBSim/adapters/entity_set_adapter.py` adds `EntitySetAdapter`.
It converts existing `HeteroObsAdapterV2` structured fields into fixed-width
entity tokens:

- `self_entity`
- `ally_entities`
- `enemy_entities`
- `entity_valid_mask`
- `alive_mask`
- `observed_mask`
- `attention_mask`
- `role_id` and `role_name`
- `entity_type_id`

Only alive and observed masks are used. This stage intentionally does not add
distance masks, visibility masks beyond the existing observed flag, biased
random masks, Gumbel-Softmax, or a mask optimizer.

## P1b/P2 Entity Encoder Actor

`algorithms/happo/entity_policy.py` adds `EntityHAPPOReferencePolicy`.
It is opt-in and separate from `HAPPOReferencePolicy`.

Architecture:

```text
entity set
-> shared entity MLP
-> multi-head attention
-> self-token pooling
-> MAV actor head / shared UAV actor head
```

The critic remains the existing 480-dimensional MLP critic. The action space
remains the existing 3-dimensional high-level action. There is no GRU, no
random mask, no biased mask, and no full HAPPO sequential correction in this
stage.

## Why Flat Observation Is Not Enough

The 96-dimensional flat actor observation is useful for the current fixed
3v2-to-5v4 baseline, but by itself it does not guarantee robust zero-shot scale
generalization:

- slot order is encoded directly in the MLP input;
- padding is present but the flat MLP does not naturally ignore padded entities;
- ally and enemy sets are not permutation-invariant;
- 5v4 works only within the fixed `max_red=5`, `max_blue=4` capacity;
- there is no learned entity interaction layer or attention mask generator.

The new entity path addresses only the first structural step: explicit entity
tokens and an attention keep mask. It is still fixed-capacity and should not be
described as full BRMA-MAPPO.

## Deliberately Deferred

- GRU is deferred because recurrent hidden state storage/reset and sequence GAE
  need a coherent rollout/trainer change.
- Biased random mask is deferred because it requires mask-generator outputs,
  rollout storage, and training losses; adding it before a stable entity actor
  would make debugging ambiguous.
- Full HAPPO sequential correction is deferred because the current goal is to
  preserve checkpoint-compatible flat baselines and introduce one structural
  change at a time.
