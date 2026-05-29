# BRMA loss formula audit

## Scope

This pass audits the BRMA mask-generator loss formula and adds a standalone
loss helper API. It does not wire BRMA into PPO, does not create a mask-generator
optimizer, does not change rollout action selection, and does not change the
environment, reward, launch, missile, radar, blue policy, or vanilla training.

## CONFIRMED BY PAPER TEXT

- The BRMA mask vector generator is a two-layer MLP with Gumbel-Softmax /
  straight-through sampling.
- The mask generator operates on the agent's local/entity observation for enemy
  UAVs. The extracted text describes relative observations `o_i^j` for enemy
  entities as the mask-generator input, not the centralized critic state.
- The mask generator outputs entity retention probability `p`, continuous
  differentiable mask `msoft`, and discrete biased enemy mask `mB`.
- `p` is described as entity retention probability. The Top-M biased mask
  selects the `MB` entities with the smallest retention probabilities. This
  supports the current lowest-p enemy drop interpretation.
- During training, `MR,train` and `MB,train` are sampled from `Uniform(0, 2)`;
  the maximum masked friendly and enemy UAV count is 2.
- The friendly random mask `mR` randomly masks friendly UAVs. In the paper's
  entity layout this is distinct from the current agent's self entity.
- The final mask fuses random friendly mask, biased enemy mask, and death mask
  as Eq.35 before masked attention.
- Algorithm 1 stores `st`, `ot`, `at`, `rt`, `Vt`, `mB,t`, `p(a|e)`,
  `p(a|emask)`, `pt`, `msoft,t`, `ot+1`, and `st+1`.
- The PPO actor loss remains Eq.27 and the critic loss remains Eq.28.
- The mask generator objective is Eq.41 / Eq.46:
  `D_KL(p_theta(a|e) || p_theta(a|e_mask)) - beta * H(e_mask)`.
- The entropy term is defined over the selected Top-M maskable set `S` from
  `msoft`, rather than over every padded/dead entity. The extracted Eq.45 text
  appears as `- sum_j msoft_j log(msoft_j)` over `S`.
- The paper reports beta selection by fivefold cross validation from
  `(0.05, 0.1, 0.15, 0.2)`, with `0.05` selected in the reported experiments.

## PROJECT INTERPRETATION

- `compute_maskable_set()` excludes self by default, includes valid allies and
  enemies, and excludes invalid/dead/padded entities. This matches the paper's
  intent that self is not randomly removed, but the exact self-index convention
  should still be kept explicit in integration.
- `BRMALossConfig.detach_actor_terms=True` is the default because Algorithm 1
  updates the actor with PPO first, then updates the mask vector generator with
  the mask loss. The standalone helper therefore treats actor log-prob terms as
  fixed inputs unless explicitly configured otherwise.
- `compute_brma_mask_loss()` implements a sampled log-prob proxy for the paper
  KL term:
  `log_prob_unmasked - log_prob_masked`.
  This is not a full Gaussian KL. It is a safe standalone candidate because the
  current dry-run storage exposes dual action log-probs rather than full
  distribution parameters.
- The standalone candidate minimizes
  `proxy_discrepancy - entropy_coef * masked_entropy`.
  This mirrors the confirmed paper sign `KL - beta * entropy`, but should be
  replaced by exact diagonal-Gaussian KL once the actor distribution parameters
  are stored.
- `masked_entropy_loss()` implements Bernoulli entropy over the maskable set to
  satisfy the standalone API requested in this pass. The paper OCR appears
  closer to `-msoft log(msoft)` over the Top-M set, so the exact entropy form
  should be visually verified before PPO integration.
- The mask generator should be updated only through the mask loss. Actor
  parameters should not receive gradients through this helper in the default
  path.

## NEEDS VISUAL PDF VERIFICATION

- Whether the mask-generator MLP input is the raw 10-dim enemy entity feature,
  the Eq.33 entity embedding, or another post-encoder feature needs visual
  verification against the formula symbols. The OCR supports relative enemy
  observations, but exact tensor notation is partially ambiguous.
- The exact attention-mask matrix row/column convention for Eq.35 should be
  checked visually before implementing masked attention behavior.
- Whether BRMA collection samples rollout actions from masked observations or
  unmasked observations needs a dedicated integration audit. Algorithm 1 text
  indicates `oi` and `mi` are put into the encoder before action output, while
  the loss also requires both `p(a|e)` and `p(a|emask)`.
- The paper objective is a KL between full policies. The standalone log-prob
  difference proxy is useful for API shape and gradient checks, but it is not
  visually confirmed as the final training formula.
- The exact entropy formula should be checked visually because the requested
  standalone API uses Bernoulli entropy while extracted Eq.45 appears to omit
  the `(1 - msoft) log(1 - msoft)` term.

## NOT IMPLEMENTED IN THIS PASS

- No PPO update path was changed.
- No BRMA mask-generator optimizer was created.
- No actor rollout action source was changed.
- No exact diagonal-Gaussian KL loss was implemented.
- No `train_attention_mappo.py` default behavior was changed.
- No environment, reward, missile, radar, launch, blue policy, training,
  evaluation, reset, or JSBSim path was run or modified.

## Standalone API Added

- `brma.losses.BRMALossConfig`
- `brma.losses.compute_maskable_set`
- `brma.losses.masked_entropy_loss`
- `brma.losses.compute_brma_mask_loss`

The API is pure PyTorch and intended for static tests and future BRMA PPO
integration only.
