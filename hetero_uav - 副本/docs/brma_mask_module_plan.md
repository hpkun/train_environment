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

The active main-training masked path is:

```text
96-dim actor obs
-> flat-to-entity decoder
-> alive / padding / observed mask
-> BRMA EntityObservationEncoder
-> GRUCell
-> MAV actor head / shared UAV actor head
-> 3D high-level action
```

The critic remains the centralized 480-dim MLP critic.

## Runtime Semantics

- Random scale mask code is retained internally, but `--brma-random-scale-mask`
  is rejected by both training entrypoints.
- Biased mask generator is enabled by `--brma-biased-mask`.
- Both are only available for `--policy-arch brma_recurrent_masked`.
- Random masking is not a valid main-experiment setting because the current
  implementation can re-sample masks independently during rollout `policy.act`
  and PPO update `evaluate_actions`, breaking old/new log-probability
  alignment.
- The self entity is never masked by random or biased masks.
- Invalid, dead, or padded entities remain masked.
- Checkpoint metadata records `random_scale_mask`, `biased_mask`, and `random_mask_prob`.

## Random Mask Recovery Routes

The only acceptable routes to restore random/entity mask training are:

1. **Rollout mask replay:** save the exact rollout-time effective entity mask in
   the buffer and force `evaluate_actions` to reuse it during PPO update.
2. **Full BRMA biased mask objective:** implement the paper-aligned mask
   generator objective, including the policy-consistency and entropy terms.

Existing random-mask 500k/2M outputs are therefore diagnostic unsafe-mask runs,
not final main-method results.

## Conservative Paper Wording

This pass implements a BRMA-style entity-attention recurrent path with
dead/padding/observed masks. It should not be described as a full BRMA-MAPPO
reproduction because the mask KL/objective and strict BRMA training loop are
not fully connected. Random scale mask is disabled for main training until one
of the recovery routes above is implemented.
