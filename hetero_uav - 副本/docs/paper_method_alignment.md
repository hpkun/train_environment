# Paper Method Alignment

This document resets the method route for `hetero_uav` to paper-grounded modules only. It does not introduce a new training method, reward term, missile rule, PID change, aircraft XML change, blue rule change, action-space change, or observation-dimension change.

## 1. Goal

The project goal is a heterogeneous MAV/UAV cooperative air-combat experiment with 3v2 training and 5v4 fixed-capacity zero-shot evaluation. The method route should be traceable to:

- BRMA-MAPPO: entity observation, attention, recurrent temporal feature, biased random masked attention, and zero-shot scale generalization.
- HAPPO / heterogeneous policy optimization: heterogeneous decentralized actors, centralized critic, sequential policy update, and advantage decomposition.

Engineering diagnostics such as launch-envelope checks, heading alignment reports, and ACMI inspection are evidence-gathering tools. They should not be presented as method modules.

## 2. BRMA-MAPPO Paper Modules

| BRMA-MAPPO module | Paper role | Current `hetero_uav` status | Comment |
|---|---|---|---|
| Entity observation encoder | Encode ego, allies, and enemies as an entity set rather than a flat slot vector | Partially present | `EntitySetAdapter` and `EntityHAPPOReferencePolicy` provide an entity-token path, but this is a minimal actor-only implementation, not the full BRMA encoder/training path. |
| Multi-head attention | Aggregate variable entity interactions and improve scale transfer | Partially present | `EntityHAPPOReferencePolicy` uses `nn.MultiheadAttention`; it is not yet the migrated BRMA encoder with paper-style mask use and recurrent integration. |
| GRU / temporal feature | Maintain temporal state over partial observations and delayed combat geometry | Missing | Current HAPPO reference path is feed-forward; no recurrent hidden state is stored or updated in the current HAPPO rollout buffer. |
| Biased random mask | Train robustness to missing/ignored entities and support zero-shot scale generalization | Missing | Parent project has candidate mask-generator infrastructure; `hetero_uav` has not wired BRMA masks into rollout or PPO. |
| Scale generalization training | Train under random/masked entity availability and evaluate on larger formations | Missing / partial protocol only | Current 3v2-to-5v4 eval has fixed-capacity padding and masks, but no BRMA scale random mask training. |
| 3v2 train -> 5v4 zero-shot eval | Experimental protocol | Present | The environment/eval protocol supports 3v2 training and 5v4 evaluation with shared observation dimensions. |

## 3. HAPPO Paper Modules

| HAPPO module | Paper role | Current `hetero_uav` status | Comment |
|---|---|---|---|
| Heterogeneous agents | Different roles or types use different policies | Partially present | MAV actor and shared UAV actor exist in HAPPO reference v0. |
| Centralized critic | CTDE value function using global/team information | Present | Current HAPPO reference uses a centralized MLP critic over the 480-dim critic state. |
| Decentralized actors | Each role/agent acts from local observation | Present / partial | MAV and UAV actors act from actor observations; UAV actor is shared across red UAVs. |
| Sequential policy update | HAPPO updates agents/policies in a sequential order | Missing | Current trainer performs role-wise PPO-style updates, not strict HAPPO sequential correction. |
| Multi-agent advantage decomposition | Decompose joint advantage for sequential updates | Missing | Current advantage computation is shared/team-level PPO-style, not strict HAPPO decomposition. |

## 4. Current `hetero_uav` Method State

| Current module | Status | Paper alignment |
|---|---|---|
| Flat MLP baseline | Present | Baseline only; not BRMA-MAPPO and not strict TAM-HAPPO. |
| HeteroObsAdapterV2 flat actor obs | Present | BRMA-inspired fixed-capacity entity/mask observation, flattened for MLP. |
| EntitySetAdapter | Present | Intermediate step toward BRMA entity observation. |
| EntityHAPPOReferencePolicy | Present | Minimal entity-attention actor only; no GRU, no BRMA mask generator, no strict HAPPO correction. |
| MAV actor + shared UAV actor | Present | Aligns with heterogeneous actor idea. |
| Centralized MLP critic | Present | Aligns with CTDE, but not attention/recurrent critic. |
| Role-wise PPO-style update | Present | Useful reference baseline, but not strict HAPPO sequential update. |
| Launch and heading diagnostics | Present | Engineering diagnostics only, not method modules. |

## 5. Current Missing Paper Modules

The current implementation should not be described as full BRMA-MAPPO or TAM-HAPPO. Missing components are:

1. BRMA-style `EntityObservationEncoder` migrated into the actual training path.
2. GRU recurrent actor and recurrent rollout hidden-state handling.
3. Random scale/entity mask training.
4. Biased random mask generation and mask-generator loss/training.
5. Strict HAPPO sequential policy update.
6. Strict multi-agent advantage decomposition.

## 6. Parent `brmamappo` Reusable Code Map

| module | paper correspondence | parent code location | current project state | migrate next? | migration risk |
|---|---|---|---|---|---|
| `EntityObservationEncoder` | BRMA entity encoder + multi-head attention | `algorithm/feature_extractor.py`, `attention_models.py` | Only minimal local entity actor exists | Yes, P2 | Medium: tensor schema differs between parent env and `HeteroObsAdapterV2`; need adapter boundary. |
| `nn.MultiheadAttention` use | BRMA attention encoder | `algorithm/feature_extractor.py`, `attention_models.py` | Present in minimal actor only | Yes, P2 | Medium: must preserve flat checkpoint path and 96-dim compatibility. |
| Actor GRU | Temporal feature module | `algorithm/mappo_nets.py`, `attention_models.py`, `train_ppo.py`, `train_attention_mappo.py` | Missing in HAPPO reference | Yes, P3 | High: rollout buffer must store hidden states and reset on episode/death. |
| Critic GRU | Temporal value module | `algorithm/mappo_nets.py`, `train_ppo.py` | Missing; current critic is 480-dim MLP | Later / optional with P3 | High: critic bootstrap and recurrent sequence batching must be correct. |
| `MaskVectorGenerator` | Candidate biased/random mask generator | `algorithm/feature_extractor.py`, `brma/mask_generator.py` | Not wired | Yes, P4/P5 after entity+GRU | High: mask semantics and loss must be paper-consistent. |
| Random/biased mask helpers | BRMA random/biased entity masking | `brma/mask_generator.py`, `brma/collection.py` | Not wired | Yes, P4/P5 | High: current parent code includes candidate/placeholder pieces; verify exact paper equation before claims. |
| Mask-generator loss / KL loss | BRMA mask optimization | `brma/train_step.py`, `brma/losses.py` | Not wired | P5 only | High: requires actor evaluate path with masked/unmasked policy distributions. |
| Recurrent rollout hidden state | GRU training support | `train_ppo.py`, `train_attention_mappo.py`, tests under parent `tests/` | Missing in current HAPPO buffer | Yes, P3 | High: flat batching would make GRU invalid. |
| Actor forward format | Entity dict/tensor + masks + hidden state | `algorithm/mappo_nets.py`, `attention_models.py` | Current entity actor accepts flat obs or entity tensors without recurrent state | Yes, P2/P3 | Medium-high: preserve opt-in policy arch and checkpoint separation. |

## 7. What Is Not a Paper Method Module

The following must not become the method route or be described as innovations:

- Behavior cloning pretrain.
- Imitation-only pretraining.
- Heading-specific supervised loss.
- Extra imitation loss weight tuning as a method claim.
- Hand-crafted heading reward.
- Extra reward shaping to force firing.
- Relaxing missile launch conditions.
- Changing missile dynamics or target selection to make red fire.
- Treating diagnostic scripts as algorithm components.

Existing direct-chase imitation may remain an experimental initialization or sanity check already run, but it should be reported as an engineering aid, not a paper-aligned contribution.

## 8. Experimental Problem Analysis

Recent diagnostics should be interpreted as evidence about the current implementation, not as a new method:

- The red attack chain is usable: scripted/oracle behaviors can enter the envelope, fire, and hit.
- The learned policies often fail because they do not reliably align heading and AO with the target.
- Easy geometry improves range exposure, but AO remains the dominant launch blocker for learned policies.
- Direct-chase oracle heading is consistent with the environment action decode: `action[1]` is normalized absolute heading.
- Flat and entity-attention short runs did not learn the same heading-to-target behavior.

The paper-consistent response is not to add manual heading loss or reward. The next method work should return to BRMA-MAPPO/HAPPO modules: stronger entity encoder, recurrent temporal feature, and later mask-based scale-generalization training.

## 9. Revised Implementation Stages

| stage | scope | method status | purpose |
|---|---|---|---|
| P0 | Keep flat MLP baseline | Baseline | Preserve existing 10M flat run and checkpoint compatibility. |
| P1 | Keep current minimal entity_attention | Intermediate | Useful smoke path; not final method. |
| P2 | Migrate BRMA-style `EntityObservationEncoder` | Paper-aligned | Replace minimal ad hoc entity pooling with a reusable paper-style encoder. |
| P3 | Add GRU recurrent actor and recurrent rollout buffer | Paper-aligned | Add temporal feature before masks; ensure hidden state is stored/reset correctly. |
| P4 | Add scale/random mask training | Paper-aligned | Introduce scale robustness after entity+GRU path is correct. |
| P5 | Add biased random mask and mask-generator loss | Paper-aligned | Implement BRMA-specific masked attention training only after paper equation and tensor path are verified. |
| P6 | Strict HAPPO sequential update and advantage decomposition | HAPPO-aligned | Add only if the paper narrative requires strict HAPPO rather than HAPPO-style role actors. |

## 10. Immediate Next Step

The next minimum paper-consistent implementation is P2: migrate or adapt the BRMA-style `EntityObservationEncoder` into an opt-in policy path while preserving:

- existing flat baseline behavior;
- current output directories and checkpoints;
- current reward, missile, PID, aircraft XML, blue rule, action space, and observation dimensions.

Do not start BC, heading loss, new reward shaping, or missile-rule changes as the next method step.
