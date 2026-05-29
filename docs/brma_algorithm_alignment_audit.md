# BRMA / MAPPO-Attention algorithm alignment audit

## 1. Scope

This pass is audit-only. It reviews the paper algorithm requirements against the
current attention and BRMA-related code, and it does not change runtime behavior.

No training, evaluation, environment reset, JSBSim execution, reward change,
missile/radar/launch change, or vanilla 500K run file change was performed.

Sources reviewed:

- `Tan 等 - 2026 - Biased random masked attention MAPPO algorithm for zero-shot scale generalization of multi-UAV air c.pdf`
- `_paper_text_tmp.txt` extracted text from the paper PDF
- `attention_models.py`
- `train_attention_mappo.py`
- `brma/mask_generator.py`
- `my_uav_env/alignment/obs_adapter.py`
- `my_uav_env/alignment/global_state.py`
- `configs/experiment_presets.py`
- `docs/current_environment_alignment_status.md`
- `docs/paper_alignment_audit_vanilla_baseline.md`

Assumption: the extracted text is sufficient for the high-level algorithm
requirements listed below, but any garbled OCR around exact matrix indexing,
formula symbols, or Table 3 max-step notation should be visually verified before
implementation.

## 2. Paper algorithm requirements

- MAPPO-Attention is MAPPO plus the entity observation encoder; the paper also
  states MAPPO-Attention uses a GRU layer.
- The entity observation encoder maps each 10-dim entity observation through an
  MLP, applies multi-head attention, concatenates attention-head outputs with
  the original entity embedding as Eq.33, and uses only the first entity output
  as the agent feature.
- The actor feeds the encoded first-entity feature into GRU, then the action
  layer.
- Attention heads: 4.
- Gumbel-Softmax temperature: 0.1.
- The critic uses the same entity observation encoder but does not use the
  biased random mask.
- Training scenario: 6v6.
- Inference / zero-shot generalization scenarios: 8v8 and 10v10.
- During training, `MR,train` and `MB,train` are sampled from `Uniform(0, 2)`.
- Maximum masked friendly and enemy UAVs during training: 2.
- The mask vector generator outputs enemy mask `mB`, continuous mask `msoft`,
  and entity retention probability `p`.
- Friendly mask `mR` is generated randomly.
- Final mask fuses biased enemy mask, random friendly mask, and death mask as
  Eq.35.
- The fused mask is applied to the attention matrix by setting masked positions
  to a large negative value before softmax.
- Buffer stores `st, ot, at, rt, Vt, mB,t, p(a|e), p(a|emask), pt, msoft,t,
  ot+1, st+1`.
- Actor loss follows PPO clipped objective with entropy term, Eq.27.
- Critic loss is MSE to return, Eq.28.
- Mask generator loss follows Eq.41 / Eq.46: KL between `p(a|e)` and
  `p(a|emask)` minus beta times mask entropy.
- Mask generation uses Top-M selection, Gumbel-Softmax, straight-through
  estimator, and an entropy term.
- Inference mask count is based on `Neval - Ntrain`.
- Reported metrics include reward/win rate plus RWR, KD, and attention metrics
  where available: AE, KEAR, PMR.

## 3. Current implementation status

| Requirement | Status | Current implementation | Notes |
|---|---|---|---|
| MAPPO-Attention = entity observation encoder + GRU | PARTIAL | `AttentionActor` uses `EntityObservationEncoder` plus `GRUCell`; `train_attention_mappo.py` uses it. | Actor-side structure exists. Eq.33 concat is not implemented exactly. |
| Entity encoder uses 10-dim paper entity observations | PARTIAL | `attention_models.py` defaults to `entity_dim=11`; training switches to 10 for `paper-placeholder` or `strict`. | Strict path exists, but default is current 11-dim engineering observation. |
| Eq.33 concat attention output with original embedding | MISSING | Encoder returns `encoded_first` from PyTorch attention only. | The code comment explicitly says it keeps `hidden_size` instead of Eq.33 concat. |
| First entity output used as agent feature | MATCH | `encoded_first = encoded[:, 0, :]`. | Matches the paper design at high level. |
| Attention heads = 4 | MATCH | `EntityObservationEncoder(... num_heads=4)` and `AttentionActor(... num_heads=4)`. | No preset override found. |
| Temperature = 0.1 | MISSING | No Gumbel-Softmax mask generator is wired. | Temperature only matters for BRMA mask generation, not plain MAPPO-Attention. |
| Critic uses same entity observation encoder, no BRMA mask | MISSING | `train_attention_mappo.py` uses imported vanilla `CentralizedCritic`; `AttentionCritic` exists but is not wired. | `strict-global` is a flattened strict team state, not the paper's attention critic. |
| Training scenario 6v6 | MISSING | Defaults and presets are 1v1 / 2v2 smoke or 2v2 main. | No 6v6 attention or BRMA preset in `configs/experiment_presets.py`. |
| Inference 8v8 / 10v10 | MISSING | No zero-shot attention / BRMA preset or inference mask-count path found. | Evaluation scripts may evaluate attention checkpoints, but BRMA inference logic is absent. |
| `MR,train`, `MB,train` sampled from `Uniform(0,2)` | MISSING | No BRMA mask-count sampling in `train_attention_mappo.py` or `brma/mask_generator.py`. | |
| Max masked friendly/enemy UAVs during training = 2 | MISSING | No count-constrained BRMA masking path. | |
| Mask generator outputs `mB`, `msoft`, `p` | MISSING | `brma/mask_generator.py` has random keep-mask utilities and a `generate_biased_random_mask` placeholder. | It is not a learned two-layer MLP plus Gumbel-Softmax. |
| Random friend mask `mR` | PARTIAL | `MaskVectorGenerator.generate_random_keep_mask` can randomly drop candidates by probability. | It is not count-constrained `Uniform(Omega_1^NR,MR)` and is not wired into actor training. |
| Fusion with death mask and Eq.35 | PARTIAL | Attention actor keeps hard `entity_mask` for invalid/dead entities and has an optional differentiable `soft_keep_mask` path for BRMA-style suppression. | Full `mR`, `mB`, `mf` fusion is not wired into rollout/PPO. |
| Mask applied to attention matrix | PARTIAL | `nn.MultiheadAttention` uses `key_padding_mask` for invalid/dead entities; optional `soft_keep_mask` multiplies entity embeddings before attention. | Soft embedding suppression is a project interpretation. Paper row/column semantics from OCR should be visually verified before claiming exact Eq.35 behavior. |
| Buffer stores BRMA fields | PARTIAL | `BRMARolloutStorage` stores masks, counts, dual log-probs, entropy, next observations, and Gaussian `mu`/`sigma` params for exact KL. | Not wired into PPO minibatches or optimizer updates. |
| Actor loss Eq.27 | MATCH | PPO clipped loss plus entropy coefficient in `ppo_update_attention`. | This is the current MAPPO-Attention update path. |
| Critic loss Eq.28 | MATCH | MSE loss against GAE returns in `ppo_update_attention`. | The loss form matches; critic architecture does not. |
| Mask generator loss Eq.41 / Eq.46 | PARTIAL | `brma.losses` provides standalone exact diagonal-Gaussian KL minus maskable-set entropy, plus retained sampled log-prob proxy mode. | No optimizer or PPO integration yet; entropy form still needs visual verification. See `docs/brma_loss_formula_audit.md`. |
| Gumbel-Softmax and entropy term | MISSING | Not implemented. | |
| Inference mask count from `Neval - Ntrain` | MISSING | No BRMA inference logic. | |
| RWR and KD metrics | PARTIAL | `train_attention_mappo.py` logs `RWR` and `KD_Red`. | Current `RWR` is red wins divided by total episodes, while the paper defines RWR as red win rate divided by blue win rate. |
| Attention metrics AE / KEAR / PMR | MISSING | Attention weights are returned by actor but not logged or post-processed into AE, KEAR, or PMR. | |
| Strict obs adapter | PARTIAL | `--obs-adapter strict` uses env strict team observations and normalization. `paper-placeholder` truncates 11 to 10. | Strict path depends on env methods at runtime; placeholder is not paper Table 1/Table 2 semantics. |
| Strict global critic | PARTIAL | `--critic-state strict-global` flattens strict team observations with masks. | It is a useful candidate but not the paper's same-encoder critic. |
| Current presets | PARTIAL | Attention smoke presets exist for current, placeholder, strict, and strict-global critic. | Missing 6v6 train and 8v8/10v10 zero-shot evaluation presets. |

## 4. Gaps blocking MAPPO-Attention baseline

Only blockers for the paper MAPPO-Attention baseline:

1. The encoder does not implement Eq.33 exactly because it does not concatenate
   attention-head outputs with the original entity embedding.
2. The training critic is still vanilla `CentralizedCritic`; the paper baseline
   expects the critic to use the same entity observation encoder without BRMA
   mask.
3. The default attention path is not the strict 10-dim Table 1/Table 2 entity
   observation path.
4. There is no 6v6 MAPPO-Attention training preset.
5. There are no 8v8 / 10v10 zero-shot MAPPO-Attention evaluation presets.
6. Attention metrics AE, KEAR, and PMR are not computed.
7. RWR logging does not match the paper definition if blue win rate is the
   denominator.

## 5. Gaps blocking BRMA-MAPPO

Only blockers for full BRMA-MAPPO:

1. No learned two-layer MLP mask vector generator with Gumbel-Softmax,
   temperature 0.1, Top-M selection, straight-through estimator, or outputs
   `mB`, `msoft`, and `p`.
2. No training-time `MR,train` / `MB,train` sampling from `Uniform(0,2)`.
3. No count-constrained random friendly mask `mR`.
4. No Eq.35 mask fusion with death mask.
5. No actor path that computes both unmasked `p(a|e)` and masked
   `p(a|emask)`.
6. No buffer fields for BRMA mask-generator training data.
7. No mask generator loss Eq.41 / Eq.46 or optimizer update.
8. No inference path that masks `Neval - Ntrain` extra entities.
9. No BRMA 6v6 training or 8v8/10v10 evaluation presets.
10. Attention matrix mask semantics need visual paper verification before custom
    implementation because the extracted text is ambiguous about row-vs-column
    expansion while the current PyTorch key-padding path masks keys.

## 6. Safe implementation order

Pass A: attention architecture/static shape audit

- Verify `EntityObservationEncoder` input/output shapes for current, placeholder,
  and strict 10-dim entities.
- Decide the exact Eq.33 output dimension and update actor head dimensions in a
  separate behavior-changing pass.
- Verify custom attention can expose per-head weights needed by AE/KEAR/PMR.

Pass B: critic entity encoder alignment

- Wire an attention critic that uses the same entity observation encoder and no
  BRMA mask.
- Keep `strict-global` flattened critic as a separate ablation, not the paper
  critic.

Pass C: BRMA mask generator API

- Add a learned mask generator API returning `mB`, `msoft`, and `p`.
- Add count-constrained `mR` and `mB` generation with `MR` / `MB`.
- Visual-verify temperature, Top-M smallest retention-probability semantics, and
  attention mask row/column convention against the PDF before coding behavior.

Pass D: buffer extension

- Extend attention rollout storage with `mB,t`, `p(a|e)`, `p(a|emask)`, `pt`,
  `msoft,t`, next observation, and next state fields.
- Keep the vanilla attention buffer path intact until BRMA mode is explicitly
  selected.

Pass E: actor dual log-prob path `p(a|e)`, `p(a|emask)`

- Compute unmasked and masked action distributions during collection.
- Store the distributions or sufficient Gaussian parameters for the mask loss.

Pass F: mask generator loss

- Implement diagonal-Gaussian KL plus entropy over `msoft` with beta from the
  paper's selected value.
- Add a separate mask generator optimizer using learning rate 0.0005.

Pass G: 6v6 preset

- Add paper-labeled 6v6 MAPPO-Attention and BRMA presets only after the static
  architecture and buffer paths are in place.

Pass H: BRMA loss formula audit and standalone loss API

- Audit Eq.41 / Eq.46, Algorithm 1, mask-generator inputs/outputs, Top-M
  direction, entropy set, and PPO relationship.
- Add a pure PyTorch standalone loss API without PPO wiring or optimizer
  creation.
- Use exact diagonal-Gaussian KL where dual actor distribution parameters are
  available; keep any log-prob proxy loss marked as a project interpretation.
- Add a differentiable soft-mask actor API so future masked policy outputs can
  depend on `msoft` without changing default actor behavior.
- Extend BRMA dry-run collection to use the soft masked policy path and store
  Gaussian policy parameters for exact KL.

Pass I: 8v8/10v10 zero-shot evaluation

- Add evaluation presets that load a 6v6-trained policy and set mask counts from
  `Neval - Ntrain`.
- Report reward, win rate, RWR, KD, AE, KEAR, and PMR where available.

## 7. Tests needed

Pure tests first:

- Static encoder shape test for Eq.33 output dimension and GRU input dimension.
- Static attention mask test for death mask and BRMA mask semantics, including
  row/column convention once verified.
- Static strict observation adapter shape test for 1v1, 2v2, 6v6, 8v8, and
  10v10 fake observations.
- Static strict global state dimension test for 6v6, 8v8, and 10v10 fake team
  observations.
- Static mask generator test for exact `MR` / `MB` counts, death-mask fusion,
  `mB` / `msoft` / `p` shapes, and deterministic seeded behavior.
- Static buffer schema test that BRMA fields are stored and minibatched with the
  same time/env/agent indexing as actions and values.
- Static loss test for diagonal-Gaussian KL and mask entropy with known tensors.
- Static standalone BRMA loss smoke test for `brma.losses` config validation,
  maskable-set selection, entropy safety, log-prob proxy loss, and detached
  actor terms.
- Static soft-mask actor API test for `soft_keep_mask` gradients, self keep
  protection, hard mask compatibility, dual evaluation, and `paper_eq33`.
- Static soft collection/storage test for soft path, hard fallback, Gaussian
  params, msoft gradient reachability, and storage shape validation.
- Static metric tests for paper RWR definition, KD, AE, KEAR, and PMR.
- Static preset test that paper-labeled presets use 6v6 training and 8v8/10v10
  evaluation scales.

Smoke tests requiring env, JSBSim, reset, training, or evaluation:

- 1v1 attention actor forward smoke through one env reset.
- 1v1 attention critic forward smoke through strict observations.
- 1v1 BRMA collection smoke with `MR=0`, `MB=0`.
- 2v2 BRMA collection smoke with nonzero masks.
- Short 6v6 MAPPO-Attention smoke run.
- Short 6v6 BRMA smoke run.
- 8v8 and 10v10 zero-shot evaluation smoke after a compatible checkpoint exists.

These smoke tests were not run in this audit pass because they create or reset
environments and would violate the requested constraints.
