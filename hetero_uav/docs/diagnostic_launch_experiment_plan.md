# Diagnostic Launch Experiment Plan

This plan prepares launch-condition diagnostic experiments for `tam_brma_paper_aligned_v1`. These configurations are not paper-aligned main results. They are controlled probes to identify which launch condition blocks UAV firing opportunities.

Do not treat a diagnostic improvement as final method evidence. A successful diagnostic only shows that the modified condition is a likely bottleneck.

## Config Order

Run in this order:

1. baseline original config
2. `diagnostic_range15`
3. `diagnostic_ao60`
4. `diagnostic_range15_ao60`
5. `diagnostic_mav_rank`
6. `diagnostic_range15_ao60_mav_rank`
7. `tam_interval25`
8. `diagnostic_ta60` only after the above fail to produce launch opportunities

## Config Matrix

| label | 3v2 config | 5v4 config | changed | unchanged | purpose |
|---|---|---|---|---|---|
| baseline | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml` | none | reward, missile dynamics, hit model, geometry, PID, blue rule, action/obs spaces | reference behavior |
| diagnostic_range15 | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15.yaml` | launch range 10 km -> 15 km | reward, missile dynamics, hit model, AO, TA, target ranking | tests whether 10 km range is the launch bottleneck |
| diagnostic_ao60 | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ao60.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ao60.yaml` | AO 45 deg -> 60 deg | reward, missile dynamics, hit model, range, TA, target ranking | tests whether AO is too strict |
| diagnostic_range15_ao60 | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60.yaml` | launch range 15 km and AO 60 deg | reward, missile dynamics, hit model, TA, target ranking | tests coupled range/AO bottleneck |
| diagnostic_mav_rank | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_mav_rank.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_mav_rank.yaml` | red target ranking closest -> `mav_threat_rank` | reward, missile dynamics, hit model, range, AO, TA | tests whether MAV information helps target choice without bypassing launch gates |
| diagnostic_range15_ao60_mav_rank | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60_mav_rank.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60_mav_rank.yaml` | launch range 15 km, AO 60 deg, MAV-aware ranking | reward, missile dynamics, hit model, TA | tests MAV ranking when launch opportunities are less narrow |
| tam_interval25 | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml` | attack interval 0.5 s -> 25 s | reward, missile dynamics, hit model, range, AO, TA, ranking | TAM-HAPPO parameter consistency probe, not a first-launch fix |
| diagnostic_ta60 | `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ta60.yaml` | `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ta60.yaml` | TA 90 deg -> 60 deg | reward, missile dynamics, hit model, range, AO, ranking | second-stage diagnostic only; deviates from BRMA rear-hemisphere condition |

## Judging Criteria

### 2K Smoke

Use only to verify:

- the run starts;
- no NaN;
- no severe action saturation;
- launch diagnostic fields are present;
- no config loading or checkpoint failure.

Do not use 2K smoke to judge learning performance.

### 50K Probe

Use to compare bottleneck indicators:

- `range_ok`;
- `ao_ok`;
- `ta_ok`;
- `track_available`;
- `direct_track_available`;
- `mav_shared_track_available`;
- `final_launch_allowed`;
- `red_missiles_fired`.

50K cannot prove final performance. It can only show whether a modified launch condition increases launch opportunities.

### 500K Probe

Use to compare:

- red launches;
- red hits;
- blue alive / blue dead;
- red alive;
- MAV survival;
- timeout rate;
- win rate;
- whether increased launch opportunity translates into hit quality.

If launch opportunities increase but hits remain low, analyze missile launch quality next. If launch opportunities remain zero, test TA or reward-event variants later, but keep them separate from the first diagnostic batch.

## Command Templates

Activate the environment first:

```bash
conda activate brmamappo
cd /mnt/c/Users/HPK/Desktop/train_environment/hetero_uav
```

Replace `{label}`, `{config}`, and `{out}` for each diagnostic.

### 2K Smoke

```bash
python -u scripts/train_happo_reference.py \
  --config {config} \
  --output-dir outputs/{out}_2k_smoke \
  --total-env-steps 2048 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch pure_happo \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 0 \
  --heartbeat-log outputs/{out}_2k_smoke/heartbeat.log \
  --heartbeat-every-steps 50
```

### 50K Probe

```bash
python -u scripts/train_happo_reference.py \
  --config {config} \
  --output-dir outputs/{out}_50k_probe \
  --total-env-steps 50000 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch pure_happo \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 25000 \
  --keep-checkpoints 3 \
  --enable-rich-logging \
  --rich-log-dir outputs/{out}_50k_probe/rich_logs \
  --heartbeat-log outputs/{out}_50k_probe/heartbeat.log \
  --heartbeat-every-steps 50
```

### Launch Diagnostics

```bash
python -u scripts/eval_policy_launch_diagnostics.py \
  --output-dir outputs/{out}_50k_probe \
  --checkpoint latest \
  --episodes 20 \
  --scenario 3v2 \
  --diagnostic-output-dir outputs/{out}_50k_probe/launch_diag_3v2 \
  --max-steps 1000
```

For 5v4 zero-shot diagnostics, use `--scenario 5v4`.

### 500K Probe

```bash
python -u scripts/train_happo_reference.py \
  --config {config} \
  --output-dir outputs/{out}_500k_probe \
  --total-env-steps 500000 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch pure_happo \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --eval-during-training \
  --eval-interval-steps 50000 \
  --train-eval-episodes 5 \
  --checkpoint-interval-steps 50000 \
  --keep-checkpoints 5 \
  --enable-rich-logging \
  --rich-log-dir outputs/{out}_500k_probe/rich_logs \
  --heartbeat-log outputs/{out}_500k_probe/heartbeat.log \
  --heartbeat-every-steps 50
```

## Diagnostic Script Field Check

`scripts/eval_policy_launch_diagnostics.py` already distinguishes the required core fields:

- `range_ok`;
- `ao_ok`;
- `ta_ok`;
- `track_available`;
- `direct_track_available`;
- `mav_shared_track_available`;
- `final_launch_allowed`;
- `actual_missiles_fired_this_step`;
- `red_missiles_fired`;
- `missile_hits`.

The detail CSV does not currently include a literal boolean column named `actual_fire`; it can be derived as `actual_missiles_fired_this_step > 0`. If later report templates require a literal `actual_fire` field, add it as a small diagnostics-only logging patch.

## Recommended First Three Runs

1. Baseline original config:
   - establishes the current launch opportunity rates under the unchanged BRMA-style launch gate.
2. `diagnostic_range15`:
   - directly tests the most likely bottleneck if UAVs approach but remain outside 10 km.
3. `diagnostic_ao60`:
   - directly tests whether heading/attack-angle alignment is blocking launch even when range is acceptable.

Run `diagnostic_range15_ao60` only after range and AO are understood individually. Run `mav_rank` after confirming that target availability and geometry are not trivially zero. Keep `tam_interval25` as parameter-consistency analysis, not as a fix for early non-launch. Keep `diagnostic_ta60` last because it deliberately relaxes the BRMA rear-hemisphere condition.

