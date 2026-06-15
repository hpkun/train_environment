# BRMA Mask Module Plan

This document records the P4/P5 mask-module scope for `hetero_uav`.
The implementation is opt-in through `policy_arch=brma_recurrent_masked`.
It does not replace the flat baseline, `entity_attention`, `brma_entity`, or
`brma_recurrent`.

## Paper And Parent-Project Alignment

| Module | Paper Role | Parent Code Location | Current Adaptation | Fully Equivalent |
|---|---|---|---|---:|
| Random scale mask | Trains with randomly reduced entity sets to improve scale robustness. | `algorithm/feature_extractor.py`, `brma/mask_generator.py` | `apply_random_scale_mask()` randomly drops valid non-self entities during training only. | No |
| Biased random mask | Learns which entities can be ignored while preserving action distribution. | `algorithm/feature_extractor.py::MaskVectorGenerator`, `brma/mask_generator.py::BRMAMaskGenerator` | `BRMABiasedMaskGenerator` predicts keep probabilities; Top-M low-probability entities are masked. | Partial |
| Mask fusion | Combines alive/padding masks with algorithmic masks before attention. | `brma/mask_generator.py::fuse_brma_masks` | Self is always kept; padding/dead entities remain masked; masked keep tensor is passed to the BRMA entity encoder. | Partial |
| Mask loss / KL / entropy objective | Optimizes mask generator against policy consistency and entropy regularization. | `brma/losses.py`, `brma/train_step.py` | Not connected in this pass; only mask forward/application and logging are connected. | No |

## Current Minimal Adaptation

The active masked path is:

```text
96-dim actor obs
-> flat-to-entity decoder
-> random scale mask and/or biased mask
-> BRMA EntityObservationEncoder
-> GRUCell
-> MAV actor head / shared UAV actor head
-> 3D high-level action
```

The critic remains the centralized 480-dim MLP critic.

## Runtime Semantics

- Random scale mask is enabled by `--brma-random-scale-mask`.
- Biased mask generator is enabled by `--brma-biased-mask`.
- Both are only available for `--policy-arch brma_recurrent_masked`.
- Random masking follows `policy.training`; evaluation uses the full valid observation unless biased mask is explicitly part of the checkpoint behavior.
- The self entity is never masked by random or biased masks.
- Invalid, dead, or padded entities remain masked.
- Checkpoint metadata records `random_scale_mask`, `biased_mask`, and `random_mask_prob`.

## Conservative Paper Wording

This pass implements a BRMA-style masked entity-attention path. It should not be
described as a full BRMA-MAPPO reproduction because the mask KL/objective and
strict BRMA training loop are not fully connected. It is suitable for method
diagrams as an opt-in architecture component, with the above limitation stated.
