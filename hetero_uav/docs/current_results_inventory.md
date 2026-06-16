# Current Results Inventory

This inventory lists existing outputs that can be used as evidence or figure
sources. It intentionally avoids overclaiming final algorithm performance.

## 1. Flat 10M Baseline

| Field | Value |
|---|---|
| Output directory | `outputs/full_10m_normal_geometry_max1000_env1` |
| Completeness | Still running / current partial state |
| Latest observed train step | `2743808` |
| Latest train red_win | `0.1000` |
| Latest train blue_win | `0.6000` |
| Latest train MAV survival | `0.8000` |
| Latest train red missiles fired | `0` |
| Latest train missile hits | `0` |
| Last available eval highlight | At `2501120`, 3v2 red_win `0.4`, 5v4 red_win `0.2`, both timeout dominated |

Use for:

- flat baseline curve and comparison;
- showing that the MLP baseline is unstable and does not consistently learn
  attack behavior.

Do not use for:

- final superiority claims;
- claiming solved zero-shot combat transfer.

## 2. `brma_recurrent_masked` Partial 479k Probe

| Field | Value |
|---|---|
| Output directory | `outputs/brma_recurrent_masked_500k_probe` |
| Completeness | Incomplete partial run |
| Requested steps | `500000` |
| Actual completed steps | `479232` |
| Reason incomplete | Foreground tool timeout before final save |
| `latest/model.pt` | Missing |
| `best/model.pt` | Present |
| Summary files | `partial_probe_summary.csv`, `partial_probe_summary.md` |

Use for:

- evidence that `brma_recurrent_masked` can train, log masks, and produce some
  attack events;
- identifying early good / later degradation behavior.

Do not use for:

- final 500k result;
- final paper main result without rerun or checkpoint selection.

## 3. `brma_recurrent_masked` Best Checkpoint Eval

| Field | 3v2 seen | 5v4 zero-shot |
|---|---:|---:|
| Output directory | `outputs/eval_brma_recurrent_masked_best_3v2` | `outputs/eval_brma_recurrent_masked_best_5v4` |
| Episodes | 20 | 20 |
| red_win_rate | 0.25 | 0.55 |
| blue_win_rate | 0.00 | 0.10 |
| draw_rate | 0.75 | 0.35 |
| timeout_rate | 0.90 | 1.00 |
| red_elimination_win_rate | 0.10 | 0.00 |
| red_timeout_alive_advantage_rate | 0.15 | 0.55 |
| red_missiles_fired_mean | 3.65 | 6.95 |
| red_missile_hits_mean | 0.35 | 0.80 |
| blue_dead_mean | 0.35 | 0.80 |
| MAV survival | 0.00 | 0.00 |

Use for:

- preliminary best-checkpoint evidence;
- showing attack events can occur in the masked recurrent path.

Do not use for:

- final result table without more episodes or a complete run;
- MAV survival claims.

## 4. Random Mask Smoke

| Field | Value |
|---|---|
| Output directory | `outputs/debug_brma_random_mask_smoke` |
| Completeness | 1024-step smoke completed |
| policy_arch | `brma_recurrent_masked` |
| random_scale_mask | `true` |
| biased_mask | `false` |
| NaN | `false` |

Use for:

- proving random scale mask path can run, save, and log.

Do not use for:

- performance claims.

## 5. Biased Mask Smoke

| Field | Value |
|---|---|
| Output directory | `outputs/debug_brma_biased_mask_smoke` |
| Completeness | 1024-step smoke completed |
| policy_arch | `brma_recurrent_masked` |
| random_scale_mask | `false` |
| biased_mask | `true` |
| NaN | `false` |

Use for:

- proving biased mask forward path can run and save.

Do not use for:

- claiming full biased random mask objective or performance benefit.

## 6. Oracle Launch Envelope Diagnostics

| Field | Value |
|---|---|
| Output directory | `outputs/debug_launch_envelope_oracle` |
| Purpose | Verify environment fire chain and launch envelope |

Use for:

- demonstrating that the environment can produce launch/hit/death events under
  a direct-chase or rule/oracle behavior.

Do not use for:

- algorithmic method claims;
- learned policy performance claims.

## 7. Heading / Action Alignment Diagnostics

| Field | Value |
|---|---|
| Output files | `outputs/heading_alignment_diagnostics_summary.csv`, `outputs/heading_alignment_diagnostics_summary.md` if present |
| Purpose | Analyze heading action, target bearing, AO, and learned policy alignment |

Use for:

- diagnosing why learned policies may enter range but fail AO/launch quality.

Do not use for:

- proposing heading loss as a paper method unless explicitly separated as an
  engineering diagnostic.

## 8. Approach-Fire Easy Curriculum Diagnostics

| Field | Value |
|---|---|
| Output directory | `outputs/approach_fire_curriculum_50k` |
| Summary | `outputs/approach_fire_curriculum_summary.csv`, `.md` |
| flat_easy_imitation | 50k, no fire/hit, AO blocked |
| entity_easy_imitation | 50k, no fire/hit, AO blocked |

Use for:

- showing approach-and-fire remains a bottleneck under short diagnostic runs.

Do not use for:

- final method comparison;
- adding non-paper imitation/heading losses as the main method.

