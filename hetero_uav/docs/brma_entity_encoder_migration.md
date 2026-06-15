# BRMA Entity Encoder Migration

This note records the P2 migration scope for a BRMA-style entity observation
encoder. The new path is opt-in through `--policy-arch brma_entity`; the flat
baseline and the earlier `entity_attention` path remain unchanged.

| Module | Parent Project Location | Migrated | Depends On GRU | Depends On Mask Generator | Current Adaptation |
|---|---|---:|---:|---:|---|
| EntityObservationEncoder | `algorithm/feature_extractor.py`, `attention_models.py` | Yes, adapted | No | No | `BRMAEntityObservationEncoder` maps entity tokens through a shared MLP, applies `nn.MultiheadAttention`, and pools the ego token. |
| MultiheadAttention | `attention_models.py`, `algorithm/feature_extractor.py` | Yes | No | No | Uses PyTorch `nn.MultiheadAttention` with key padding mask. |
| Entity embedding | `attention_models.py` | Yes, adapted | No | No | Shared entity MLP embeds ego, ally, and enemy tokens decoded from the existing 96-dim actor observation. |
| Paper Eq. 33-style output | `attention_models.py` | Yes, adapted | No | No | Actor input concatenates ego entity embedding and ego attention output. |
| Actor GRUCell | `algorithm/mappo_nets.py`, `attention_models.py` | No | Yes | No | Deliberately excluded from P2; planned for a later P3 if the entity encoder path is stable. |
| MaskVectorGenerator | `algorithm/feature_extractor.py`, `brma/mask_generator.py` | No | No | Yes | Deliberately excluded from P2; alive and observed masks only are used as attention padding masks. |
| Biased/random mask training | `brma/train_step.py`, `brma/mask_generator.py` | No | No | Yes | Deliberately excluded from P2; no Gumbel, random scale mask, or biased mask objective is added. |
| Centralized critic | Current `hetero_uav` HAPPO reference | Kept | No | No | The critic remains the existing 480-dim MLP critic for checkpoint and training simplicity. |

## Current Adaptation

`BRMAEntityHAPPOReferencePolicy` consumes the existing fixed 96-dim actor
observation and decodes it into entity tokens internally:

- ego token;
- up to four ally tokens;
- up to four enemy tokens;
- valid/alive/observed masks from the V2 observation tail.

The policy keeps the current high-level 3D action output and role-wise heads:

- one MAV actor head;
- one shared UAV actor head;
- centralized 480-dim critic.

This is a BRMA-style entity encoder path, not a full BRMA-MAPPO reproduction.
It does not include GRU, random scale mask, biased random mask, or strict HAPPO
sequential correction.
