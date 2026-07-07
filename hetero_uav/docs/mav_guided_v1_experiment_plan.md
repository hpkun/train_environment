# MAV-Guided v1 Experiment Plan

`mav_guided_v1` is the main-method configuration for heterogeneous MAV/UAV zero-shot scale transfer. It is not a fire-control diagnostic config.

## 1. Method Positioning

`mav_guided_v1` is intended for the main experiment line:

- train in 3v2;
- evaluate zero-shot transfer in 5v4;
- keep the MAV as battlefield information / mission guidance node;
- keep attack UAVs responsible for approach, lock, launch, and attack.

The central mechanism is:

```text
MAV shared observation
-> red_uav_track_policy=mav_required_when_alive
-> red_target_selection_mode=mav_threat_rank
-> UAV still must satisfy range/AO/TA/lock/deconfliction
-> scripted BRMA-style launch
```

This means the MAV does not fire and does not press a fire button for UAVs. It constrains and guides the UAV attack chain through shared target information and target ranking.

## 2. TAM-HAPPO Alignment

Aligned elements:

- MAV carries no missiles.
- MAV provides battlefield information / mission guidance.
- UAVs carry missiles and execute lock/launch/attack.
- MAV reward remains the existing `tam_brma_paper_aligned_v1` safety/support/event structure.
- UAV reward remains the existing BRMA-style flight/advantage/terminal trunk in `tam_brma_paper_aligned_v1`.
- MAV team credit still comes from UAV kills.

`red_uav_track_policy=mav_required_when_alive` makes MAV information operationally visible: when the MAV is alive and observes a target, a red UAV's launch track for that target must come from MAV-shared information. If the MAV is dead or cannot observe that target, the UAV can fall back to direct track.

## 3. BRMA-MAPPO Alignment

Preserved elements:

- scripted missile launch;
- 10 km short-range missile contract;
- 0.25 s lock delay;
- 0.5 s launch cooldown;
- same-target deconfliction;
- TA90 rear-hemisphere / 3-9 line condition;
- UAV must still maneuver into the launch envelope.

`missile_launch_ao_deg=60.0` is a fire-control calibration for early training closure, not the main method contribution. The main contribution is the coupling of MAV shared observation, MAV-required track source, and MAV-aware target ranking.

The method does not use `range15`, `ta60`, no-TA, or geometry bypassing.

## 4. Configs

Training config:

```text
uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml
```

Zero-shot eval config:

```text
uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml
```

Both configs keep:

- `hetero_reward_mode: tam_brma_paper_aligned_v1`;
- `observation_mode: mav_shared_geo`;
- F16 MAV surrogate dynamics with F22 visual;
- `red_0` as MAV with zero missiles;
- attack UAVs with two missiles;
- `sim_freq: 60`;
- `agent_interaction_steps: 12`;
- `action_trim_by_role.mav.pitch: 0.0`;
- `mav_observation_range_m: 80000`;
- `uav_direct_observation_range_m: 10000`;
- `missile_launch_range_m: 10000.0`;
- `missile_launch_min_range_m: 500.0`;
- `missile_launch_ta_deg: 90.0`;
- `missile_attack_interval_sec: 0.5`.

## 5. Key Metrics

Primary MAV-guidance metrics:

- `red_launch_mav_shared_count`;
- `red_hit_mav_shared_count`;
- `first_red_mav_shared_launch_step`;
- `first_red_mav_shared_hit_step`;
- `red_launch_direct_count`;
- `red_hit_direct_count`;
- `red_launch_unknown_source_count`;
- `red_hit_unknown_source_count`;
- `red_launch_with_mav_shared_track`;
- `red_hit_with_mav_shared_track`.

Combat metrics:

- `red_missiles_fired`;
- `missile_hits`;
- `blue_alive_final`;
- `red_alive_final`;
- `mav_survival`;
- `timeout`;
- `red_win`;
- 5v4 zero-shot transfer metrics.

The rich missile log already records `launch_track_source`. `eval_policy_launch_diagnostics.py` summarizes direct vs MAV-shared launch and hit counts from the actual `launch_quality` launch / termination records emitted by the environment, not from pre-step visibility flags.

Diagnostic source semantics:

- direct/shared launch and hit counts are computed from actual launch-quality records.
- pre-step track flags are only used for envelope diagnostics and may be `mixed` when direct and MAV-shared visibility both exist.
- actual launch source is the `launch_track_source` recorded by the environment at missile launch time.
- hit source is inherited from the missile's launch-quality record and deduplicated by `missile_id`.
- unknown source counts are reported separately if a launch or hit record lacks a clean `direct` or `mav_shared` source.

## 6. Command Templates

Activate the environment first:

```bash
conda activate brmamappo
cd /mnt/c/Users/HPK/Desktop/train_environment/hetero_uav
```

### 2K Smoke

```bash
python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir outputs/mav_guided_v1_3v2_2k_smoke \
  --total-env-steps 2048 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch brma_recurrent_masked \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 0 \
  --heartbeat-log outputs/mav_guided_v1_3v2_2k_smoke/heartbeat.log \
  --heartbeat-every-steps 50
```

### 50K Probe

```bash
python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir outputs/mav_guided_v1_3v2_50k_probe \
  --total-env-steps 50000 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch brma_recurrent_masked \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --checkpoint-interval-steps 25000 \
  --keep-checkpoints 3 \
  --enable-rich-logging \
  --rich-log-dir outputs/mav_guided_v1_3v2_50k_probe/rich_logs \
  --heartbeat-log outputs/mav_guided_v1_3v2_50k_probe/heartbeat.log \
  --heartbeat-every-steps 50
```

### 500K Probe

```bash
python -u scripts/train_happo_reference.py \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --output-dir outputs/mav_guided_v1_3v2_500k_probe \
  --total-env-steps 500000 \
  --rollout-length 256 \
  --num-envs 1 \
  --max-steps 1000 \
  --device cuda \
  --policy-arch brma_recurrent_masked \
  --opponent-policy brma_rule \
  --reward-mode tam_brma_paper_aligned_v1 \
  --eval-during-training \
  --eval-interval-steps 50000 \
  --train-eval-episodes 5 \
  --checkpoint-interval-steps 50000 \
  --keep-checkpoints 5 \
  --enable-rich-logging \
  --rich-log-dir outputs/mav_guided_v1_3v2_500k_probe/rich_logs \
  --heartbeat-log outputs/mav_guided_v1_3v2_500k_probe/heartbeat.log \
  --heartbeat-every-steps 50
```

### 3v2 Launch Diagnostics

```bash
python -u scripts/eval_policy_launch_diagnostics.py \
  --output-dir outputs/mav_guided_v1_3v2_50k_probe \
  --checkpoint latest \
  --episodes 20 \
  --scenario 3v2 \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --diagnostic-output-dir outputs/mav_guided_v1_3v2_50k_probe/launch_diag_3v2 \
  --max-steps 1000
```

### 5v4 Zero-Shot Launch Diagnostics

```bash
python -u scripts/eval_policy_launch_diagnostics.py \
  --output-dir outputs/mav_guided_v1_3v2_50k_probe \
  --checkpoint latest \
  --episodes 20 \
  --scenario 5v4 \
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml \
  --diagnostic-output-dir outputs/mav_guided_v1_3v2_50k_probe/launch_diag_5v4 \
  --max-steps 1000
```

## 7. Reporting Boundary

Safe wording:

- "`mav_guided_v1` makes MAV shared information part of the UAV launch track contract and target ranking."
- "UAVs still satisfy BRMA-style range/AO/TA/lock/deconfliction before launch."
- "AO60 is a fire-control calibration; the preserved core rear-hemisphere condition is TA90."

Do not claim:

- MAV directly launches missiles;
- MAV shared track bypasses launch geometry;
- the method removes 3-9 line;
- diagnostic range/AO/TA variants are part of this main method;
- `mav_guided_v1` is a complete reproduction of TAM-HAPPO.
