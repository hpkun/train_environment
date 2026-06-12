# BRMA Observation Alignment

## 1. BRMA-MAPPO paper observation design

The BRMA-MAPPO paper targets zero-shot scale generalization in multi-UAV air combat. Its observation design is entity-based rather than a single unstructured flat vector.

The paper's core observation ideas are:

- local observation is decomposed into ego/self state, friendly entities, and enemy entities;
- an entity observation encoder based on multi-head attention processes variable battlefield entities;
- dead or unavailable entities are handled through entity masking rather than being treated as normal active observations;
- a centralized critic can use broader/global information than each local actor;
- zero-shot scale generalization is tested by training at one team size and evaluating at larger team sizes;
- biased random mask attention is introduced to improve scale generalization by learning to mask less important entities while retaining key battlefield information.

The biased random mask is not just a padding mask. It is a learned/randomized masking mechanism applied around the attention computation to reduce over-dependence on fixed entity counts and to help the policy focus on important entities when the number of UAVs changes.

## 2. Current HeteroObsAdapterV2 design

The current mainline observation path uses `observation_mode = "mav_shared_geo"` and `HeteroObsAdapterV2`.

The actor observation is a fixed 96-dimensional vector built from:

- `ego_feature`: ego geometric state, ego role one-hot, and missile warning;
- `ally_entities`: padded ally relative geometry plus ally role;
- `enemy_entities`: padded enemy relative geometry plus track source;
- `ally_valid_mask`;
- `ally_alive_mask`;
- `enemy_valid_mask`;
- `enemy_alive_mask`;
- `enemy_observed_mask`.

The adapter uses:

- `max_red = 5`;
- `max_blue = 4`;
- actor observation dimension `96`;
- critic state dimension `480`, formed as five padded actor-observation slots.

This means 3v2 and 5v4 use the same actor input schema. In 3v2, missing red slots are zero-padded and marked invalid in `red_valid_mask`. In 5v4, `red_1` through `red_4` all receive the same 96-dimensional UAV actor input format.

## 3. Alignment table

| item | status | current implementation |
|---|---|---|
| entity-style observation | paper-aligned | ego, ally entities, and enemy entities are separated before flattening |
| padding | paper-aligned | V2 pads to `max_red=5`, `max_blue=4` |
| valid/alive mask | paper-aligned | red valid mask, ally alive mask, enemy valid mask, and enemy alive mask are explicit |
| enemy observed mask | paper-aligned | enemies can be alive but unobserved; unobserved entities are masked/zeroed |
| centralized critic state | paper-aligned | critic state is fixed at 480 and aggregates padded red actor observations |
| 3v2 to 5v4 zero-shot input compatibility | paper-aligned | both scenarios use the same 96-dimensional actor schema |
| permutation invariance | partially aligned | entities are separated by type, but still flattened in slot order |
| attention-based entity encoder | not implemented | current actor uses flat vectors / MLP-style consumers, not entity attention |
| biased random mask | not implemented | no learned/randomized BRMA mask generator or attention-matrix masking |
| scale generalization beyond `max_red/max_blue` | not implemented | current design is fixed-capacity 3v2 to 5v4, not arbitrary scale |

## 4. What is aligned

The current observation design is aligned with BRMA-MAPPO at the protocol level:

- it uses a BRMA-inspired entity/mask observation;
- it separates ego, allies, and enemies;
- it includes validity, alive, and observed masks;
- it supports a centralized critic input;
- it keeps 3v2 and 5v4 actor observations dimension-compatible;
- it allows 5v4 newly added UAVs to reuse the same shared UAV actor input.

This is enough to continue the main 3v2-to-5v4 zero-shot protocol.

## 5. What is only partially aligned

Permutation handling is only partially aligned. BRMA-MAPPO uses an attention-based entity encoder, which is naturally better suited to set/entity processing. The current V2 adapter preserves entity groups and masks, but then flattens them in deterministic slot order. Therefore, the current implementation is entity-style and mask-aware, but not fully permutation invariant.

The current zero-shot protocol is also narrower than BRMA-MAPPO. It is fixed-capacity 3v2-to-5v4 scale transfer, not open-ended generalization to any larger team size.

## 6. What is not implemented

The current project does not implement:

- the full BRMA-MAPPO observation encoder;
- multi-head attention over entities;
- biased random masked attention;
- a mask vector generator network;
- attention-matrix masking with learned/randomized entity retention;
- permutation-invariant set encoding;
- arbitrary-scale zero-shot generalization beyond the configured capacity.

These are algorithmic extensions, not blockers for the current fixed-capacity experiment.

## 7. Implication for our 3v2 to 5v4 zero-shot experiment

The current observation design is sufficient for a fixed-capacity 3v2 to 5v4 zero-shot scale-transfer experiment:

- the same actor schema is used in both settings;
- 3v2 is represented as a padded subset of the 5v4 capacity;
- 5v4 new UAVs reuse the same shared UAV actor input format;
- masks explicitly identify valid, alive, and observed entities.

However, the thesis/report should describe this as BRMA-inspired entity/mask observation, not as full BRMA-MAPPO. If the future method needs stronger scale generalization claims, the next algorithmic step would be an entity attention encoder and, later, biased random mask attention.

## 8. Safe wording for thesis/report

Safe wording:

> We adopt a BRMA-inspired entity/mask observation schema with fixed maximum team capacity. The actor observation separates ego, ally, and enemy entities and includes valid, alive, and observed masks. This gives a unified input interface for 3v2 training and 5v4 zero-shot evaluation.

Safe limitation:

> The current implementation does not reproduce BRMA-MAPPO's attention-based entity encoder or biased random masked attention. Our zero-shot setting is fixed-capacity 3v2-to-5v4 scale transfer, not arbitrary-scale generalization.

Do not claim:

- complete BRMA-MAPPO observation encoder reproduction;
- biased random masked attention;
- permutation-invariant set encoding;
- arbitrary team-size generalization;
- solved zero-shot combat transfer.

