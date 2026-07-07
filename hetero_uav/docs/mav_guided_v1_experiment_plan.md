# MAV-Guided v1 Experiment Plan

`mav_guided_v1` is the current main 3v2 training environment for checking whether
`pure_happo + num_envs=4` can learn a basic UAV launch and hit loop in JSBSim
heterogeneous air combat. This stage is a learnability check, not the final
zero-shot scale-transfer stage. GRU, mask, and entity-attention methods should be
added only after the attack loop is open.

## Current Main Config

The 3v2 and 5v4 `mav_guided_v1` configs are updated in place. They keep reward,
missile dynamics, hit model, PID, aircraft XML, blue rule, action space, and
observation dimension unchanged.

Current settings:

- `red_uav_track_policy: mav_preferred_when_alive`
- `red_target_selection_mode: closest`
- `missile_launch_range_m: 14000.0`
- `missile_launch_ao_deg: 60.0`
- `missile_launch_ta_deg: 90.0`
- `missile_launch_min_range_m: 500.0`
- `missile_attack_interval_sec: 25.0`
- `hetero_reward_mode: tam_brma_paper_aligned_v1`
- `observation_mode: mav_shared_geo`

Rationale:

- `mav_preferred_when_alive` uses MAV shared information when it is available,
  but keeps direct fallback so early training is not completely starved of
  launch opportunities.
- `closest` target selection reduces target-ranking complexity while validating
  environment learnability.
- `range=14km`, `AO=60deg`, and `attack interval=25s` are learnability settings
  for opening the early pure-HAPPO attack loop.
- TA90 / 3-9 line is retained.
- Range/AO/TA/lock/deconfliction remain active; MAV shared track does not bypass
  launch geometry.

## Observation Fix

`enemy_track_source` uses two bits: `[direct, mav_shared]`. Direct and MAV-shared
tracks can now both be true:

- direct only: `[1.0, 0.0]`
- MAV shared only: `[0.0, 1.0]`
- direct and MAV shared: `[1.0, 1.0]`
- neither: `[0.0, 0.0]`

The observation shape and observation space are unchanged.

## Key Metrics

Primary learnability metrics:

- `red_missiles_fired > 0`
- `missile_hits > 0`
- `red_launch_with_mav_shared_track > 0`
- `red_hit_with_mav_shared_track > 0`
- `red_launch_unknown_source_count` should stay near 0
- `red_hit_unknown_source_count` should stay near 0
- `dominant_block_reason` should not remain stuck at `out_of_range`

Source-specific diagnostics:

- `red_launch_direct_count`
- `red_launch_mav_shared_count`
- `red_launch_direct_and_mav_shared_count`
- `red_launch_unknown_source_count`
- `red_hit_direct_count`
- `red_hit_mav_shared_count`
- `red_hit_direct_and_mav_shared_count`
- `red_hit_unknown_source_count`
- `red_launch_with_mav_shared_track`
- `red_hit_with_mav_shared_track`

Launch/hit source counts are computed from actual launch-quality records emitted
by the environment, not inferred from pre-step visibility flags. Pre-step track
flags are only envelope diagnostics. Hit source is inherited from the missile's
launch-quality record and deduplicated by `missile_id`.

## Command Templates

Activate the environment first:

```bash
conda activate brmamappo
cd /mnt/c/Users/HPK/Desktop/train_environment/hetero_uav
```

### 2K Smoke

```bash
python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir outputs/mav_guided_v1_pure_happo_env4_3v2_2k_smoke \
  --total-env-steps 2048 \
  --rollout-length 256 \
  --num-envs 4 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch pure_happo \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 0
```

### 50K Probe

```bash
python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir outputs/pure_happo_mav_guided_v1_env4_3v2_50k_probe \
  --total-env-steps 50000 \
  --rollout-length 256 \
  --num-envs 4 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch pure_happo \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 25000 \
  --keep-checkpoints 3 \
  --enable-rich-logging \
  --rich-log-dir outputs/pure_happo_mav_guided_v1_env4_3v2_50k_probe/rich_logs
```

### Launch Diagnostics

```bash
python -u scripts/eval_policy_launch_diagnostics.py \
  --output-dir outputs/pure_happo_mav_guided_v1_env4_3v2_50k_probe \
  --checkpoint latest \
  --episodes 20 \
  --scenario 3v2 \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --diagnostic-output-dir outputs/pure_happo_mav_guided_v1_env4_3v2_50k_probe/launch_diag_3v2 \
  --max-steps 1000
```

## Reporting Boundary

Safe wording:

- "`mav_guided_v1` repairs dual-source track representation and prefers MAV
  shared information when available."
- "Direct fallback remains enabled to test early training learnability."
- "UAVs still satisfy range/AO/TA/lock/deconfliction before launch."

Do not claim:

- MAV directly launches missiles;
- MAV shared track bypasses launch geometry;
- the method removes TA90 / 3-9 line;
- this stage proves final zero-shot scale transfer;
- random masks or recurrent/entity methods are part of this pure-HAPPO probe.
