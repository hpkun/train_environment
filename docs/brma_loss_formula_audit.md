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
  the mask loss. The standalone helper therefore treats actor distribution terms
  as fixed inputs unless explicitly configured otherwise.
- `compute_brma_mask_loss()` now defaults to `kl_mode="gaussian"` and uses the
  exact diagonal-Gaussian closed-form KL for the paper divergence term.
- `kl_mode="sample_logprob_proxy"` preserves the earlier static candidate:
  `log_prob_unmasked - log_prob_masked`. This remains a project interpretation
  and should not be treated as the final paper loss.
- The standalone Gaussian candidate minimizes
  `diagonal_gaussian_kl - entropy_coef * masked_entropy`.
  This aligns the KL part with Eq.41 / Eq.46 while keeping training integration
  disabled.
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
- The exact entropy formula should be checked visually because the requested
  standalone API uses Bernoulli entropy while extracted Eq.45 appears to omit
  the `(1 - msoft) log(1 - msoft)` term.

## NOT IMPLEMENTED IN THIS PASS

- No PPO update path was changed.
- No BRMA mask-generator optimizer was created.
- No actor rollout action source was changed.
- The exact diagonal-Gaussian KL loss is standalone only and is not wired into
  PPO or mask-generator updates.
- No `train_attention_mappo.py` default behavior was changed.
- No environment, reward, missile, radar, launch, blue policy, training,
  evaluation, reset, or JSBSim path was run or modified.

## Update: exact diagonal-Gaussian KL API

- `diagonal_gaussian_kl(mu_p, sigma_p, mu_q, sigma_q)` implements
  `KL(N_p || N_q)` for diagonal Gaussian action distributions and returns one
  KL value per batch item.
- `compute_brma_mask_loss(..., kl_mode="gaussian")` uses
  `KL(p(a|e) || p(a|emask)) - beta * H(mask)` with `entropy_coef=0.05` by
  default, matching the paper's reported beta selection.
- The KL term is now paper-aligned for distribution divergence. The entropy form
  still needs visual PDF verification: the current helper uses Bernoulli entropy
  over the maskable set, while extracted Eq.45 appears closer to
  `-msoft log(msoft)`.
- This is still not a complete BRMA training implementation. Future integration
  needs a differentiable masked encoder path so mask-generator parameters can
  receive gradients through `mu_masked` / `sigma_masked`.

## Update: differentiable soft-mask actor API

- `EntityObservationEncoder.forward(..., soft_keep_mask=None)` now supports
  optional float keep weights where 1 means visible and 0 means softly
  suppressed. The default `None` path is unchanged.
- `AttentionActor.evaluate_actions(..., soft_keep_mask=None)` and
  `evaluate_dual_actions(..., masked_soft_keep_mask=None)` can route `msoft`
  into the masked policy path without converting it to a hard bool mask.
- This is a project-interpretation differentiable suppression path: keep weights
  multiply entity embeddings before attention, while hard `entity_mask` still
  handles invalid/dead/padded entities and self is forced visible.
- Exact Gaussian KL is implemented, but full mask-generator training still needs
  PPO/rollout integration and an optimizer. This pass only makes the future
  masked policy path differentiable with respect to soft keep weights.
- Eq.35's exact attention-mask matrix row/column convention still needs visual
  PDF verification before claiming paper-complete masked attention behavior.

## Update: soft collection and Gaussian parameter storage

- `collect_brma_dry_run_step(..., use_soft_mask_path=True)` now evaluates the
  masked policy with original hard entity validity masks plus `msoft`-derived
  soft keep weights only on the selected `mR` / `mB` set. Unselected valid
  entities keep weight 1. The hard BRMA key-padding mask is retained for
  diagnostics and `use_soft_mask_path=False` fallback.
- `BRMARolloutStorage` now stores `mu_unmasked`, `mu_masked`,
  `sigma_unmasked`, and `sigma_masked`, so future BRMA mask loss code can use
  the exact diagonal-Gaussian KL API without reconstructing policy parameters.
- This still does not wire mask loss into PPO, does not create a mask-generator
  optimizer, and does not change default `brma_mode=off` training behavior.
- The selected-set soft keep path better matches Eq.35 and the paper's selected
  Top-M entropy set `S` than applying `msoft` to every valid non-self entity.
- Eq.35's exact row/column convention and the exact entropy form remain visual
  PDF verification items before a paper-complete BRMA training claim.

## Standalone API Added

- `brma.losses.BRMALossConfig`
- `brma.losses.diagonal_gaussian_kl`
- `brma.losses.compute_maskable_set`
- `brma.losses.masked_entropy_loss`
- `brma.losses.compute_brma_mask_loss`

The API is pure PyTorch and intended for static tests and future BRMA PPO
integration only.
