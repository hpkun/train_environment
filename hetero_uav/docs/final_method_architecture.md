# Final Method Architecture

This document defines the current opt-in BRMA/HAPPO method path and the
wording boundary for paper or report use. It is a code-alignment note, not a
claim that every original-paper component has been reproduced exactly.

## Available Policy Architectures

| Policy arch | Encoder | Recurrent actor | Mask module | Actor heads | Critic |
|---|---|---:|---|---|---|
| `flat` | 96-dim MLP | No | No | MAV head + shared UAV head | 480-dim MLP |
| `entity_attention` | Local entity attention | No | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_entity` | BRMA-style entity attention encoder | No | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_recurrent` | BRMA-style entity attention encoder | GRUCell | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_recurrent_masked` | BRMA-style entity attention encoder | GRUCell | Dead/padding/observed mask; random scale mask rejected by train entrypoints; biased mask is diagnostic/opt-in only | MAV head + shared UAV head | 480-dim MLP |

The default training path remains `flat`. All entity, recurrent, and masked
paths are opt-in through `--policy-arch`.

## Final Method Diagram Text

The most complete implemented path can be drawn as:

```text
Observation
-> Entity construction from fixed 96-dim actor_obs
-> Mask module
   -> alive / padding / observed masks from observation
   -> random scale mask is disabled for main training
   -> optional biased mask generator when explicitly enabled, without full BRMA mask objective
-> BRMA-style EntityObservationEncoder
-> Multi-head attention
-> GRUCell recurrent actor state
-> Role-wise policy heads
   -> MAV actor head
   -> shared UAV actor head
-> Gaussian action distribution
-> JSBSim environment
-> Centralized 480-dim MLP critic for PPO update
```

## `brma_recurrent_masked` Code Path Audit

| Stage | Current implementation | Correctness boundary |
|---|---|---|
| Input | Consumes existing 96-dim actor observation. | Observation dimension and environment schema are unchanged. |
| Entity construction | Decodes self, ally, enemy slots into fixed entity tensors. | Entity capacity is fixed by `max_allies=4`, `max_enemies=4`; this is fixed-capacity 3v2-to-5v4, not arbitrary scale. |
| Alive/padding mask | Uses valid/alive/observed masks from actor observation. | Dead, unobserved, and padding entities are excluded from attention. |
| Self mask | Self token is always kept when actor observation is valid. | Random and biased masks must not drop the self token. |
| Random scale mask | Internal code is retained for future repair, but main training entrypoints reject `--brma-random-scale-mask`. | Current sampling is unsafe for PPO because rollout `policy.act` and update `evaluate_actions` can use different masks for the same transition. Re-enable only with rollout mask replay or a full BRMA biased-mask objective. |
| Biased mask generator | Produces keep probabilities and masks low keep-probability ally/enemy slots when enabled. | Forward mask generation is implemented; full BRMA KL/objective training is not. |
| Entity encoder | Shared entity MLP plus PyTorch multi-head attention with key padding mask. | BRMA-style encoder, not a verbatim full BRMA-MAPPO training stack. |
| GRU | `nn.GRUCell` processes pooled entity embedding before actor heads. | PPO update replays one-step stored hidden states; it is not full TBPTT recurrent PPO. |
| Actor heads | Separate MAV head and shared UAV head. | Heterogeneous role-wise actor structure is implemented. |
| Critic | Centralized MLP critic over 480-dim global state. | Critic is not entity-attention or recurrent. |
| Rollout buffer | Stores flat obs, critic state, actions, log probs, active masks, env id, next value, and optional recurrent hidden state. | Hidden state is grouped by env and reset on episode done during rollout. |
| Checkpoint | `meta.json` stores `policy_arch`, entity dim, critic dim, recurrent size, random mask flag, biased mask flag, and mask probability. | Meta is sufficient to rebuild the implemented opt-in policy. |

## Comparison With Parent `brmamappo`

| Module | Parent project behavior | Current implementation | Consistency | Difference and paper wording |
|---|---|---|---|---|
| EntityObservationEncoder | Shared entity MLP, multi-head attention, death mask / active mask fusion. | Shared entity MLP, multi-head attention, keep mask from alive/padding/observed slots. | Partially aligned | Wording: "BRMA-style entity attention encoder", not full BRMA-MAPPO reproduction. |
| MultiheadAttention | Attention over ego, allies, enemies with key padding mask. | PyTorch `nn.MultiheadAttention` with key padding mask, ego-token pooling. | Aligned at module level | Pooling and input schema are adapted to current 96-dim observation. |
| GRUCell | Recurrent actor/value network paths with runner-managed hidden states. | GRUCell in actor path only; centralized critic remains MLP. | Partially aligned | Wording: "GRU recurrent actor", not full recurrent actor-critic. |
| MaskVectorGenerator | Gumbel/biased mask generation plus mask-related losses and rollout schema. | Biased keep-probability generator plus Top-M mask application and logging. | Partial | Wording: "biased mask generator forward path"; do not claim full biased random mask objective. |
| Random mask | Random friendly/enemy masking for scale robustness. | Internal random non-self entity dropout exists, but training entrypoints reject it. | Not used in main training | Existing random-mask runs are diagnostic unsafe-mask runs, not final results, because the mask is not replayed in PPO updates. |
| RNN state reset | Runner resets hidden state on episode done. | Training loop zeros per-env hidden state on episode reset. | Aligned for episode reset | Does not implement full TBPTT sequence training. |
| Actor heads | MAPPO/HAPPO actor output distribution. | Role-wise MAV actor and shared UAV actor Gaussian distribution. | Aligned with heterogeneous actor goal | Not strict HAPPO sequential correction. |
| Rollout storage | Stores recurrent states and masks for PPO. | Stores recurrent hidden state per transition for one-step replay. | Partial | Good for smoke/probe, not a full recurrent PPO storage design. |

## What Can Be Claimed

Safe claims:

- BRMA-style entity attention encoder.
- GRU recurrent actor.
- Dead/padding/observed entity masks.
- Biased mask generator forward path.
- Role-wise heterogeneous actor with independent MAV head and shared UAV head.
- Centralized critic over global state.
- Opt-in masked recurrent policy path compatible with existing 3v2 and 5v4 fixed-capacity observations.

## What Must Not Be Claimed

Do not claim:

- Complete BRMA-MAPPO reproduction.
- Complete biased random mask objective.
- Full BRMA KL/mask-generator training.
- Random scale mask results as final main-method evidence unless mask replay or the full BRMA objective is implemented.
- Full recurrent PPO or TBPTT.
- Strict HAPPO sequential correction or formal multi-agent advantage decomposition.
- Proof that every module independently improves performance.
- Arbitrary-size zero-shot generalization beyond the configured fixed capacity.

## Diagnostics Are Not Method Modules

Launch-envelope audits, direct-chase oracle checks, heading diagnostics, ACMI
exporters, and plotting utilities are engineering diagnostics. They should not
be presented as algorithmic method components.
