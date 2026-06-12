# F-16 MAV Surrogate HAPPO Reference 1M Validation Results

## Experiment Setup

This run is a HAPPO reference v0 validation run, not a TAM-HAPPO reproduction.

- Environment family: heterogeneous MAV/UAV 3v2 reference setting.
- Training config: `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml`.
- MAV surrogate: F-16 dynamics/model used for the MAV slot.
- MAV role: preserved as `mav`.
- MAV armament: unarmed.
- Observation: MAV shared geometry observation preserved.
- Action: high-level `[pitch, heading, speed]` retained.
- Missile and evasion: scripted environment logic retained.
- Reward: `happo_ref_v0`.
- Opponent: `brma_rule`.
- Algorithm: separate MAV/UAV actors, centralized critic, simplified sequential HAPPO-style update.
- GRU: not implemented.
- Attention: not implemented.

This run must not be described as full TAM-HAPPO, a paper action-space reproduction, or an attention/temporal reproduction.

## Training Rollout Result

The 1M training run completed.

- `total_env_steps_actual`: 1,000,000.
- Episodes: 15,625.
- NaN detected: false.
- Latest training row:
  - `avg_return`: 12.9876.
  - `red_win`: 1.0.
  - `blue_win`: 0.0.
  - `timeout`: 1.0.
  - `mav_survival`: 1.0.
  - `red_alive_final`: 3.0.
  - `blue_alive_final`: 2.0.
  - `red_missiles_fired`: 0.
  - `blue_missiles_fired`: 0.
  - `missile_hits`: 0.

The training rollout indicates stable survival and timeout alive advantage. It does not indicate an effective combat or kill strategy because there is no missile firing and no missile hit in the rollout log.

## Independent Checkpoint Evaluation

The 100-episode checkpoint evaluation was run with `--checkpoint-mode all`. The saved records are in:

- `outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/checkpoint_eval/happo_3v2_checkpoint_eval.json`
- `outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/checkpoint_eval/happo_3v2_checkpoint_eval.md`

The evaluation records use:

- `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml`
- `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml`

### Best Checkpoint

3v2:

- `avg_return`: -67.8468.
- `avg_length`: 1000.0.
- `red_win_rate`: 0.03.
- `blue_win_rate`: 0.57.
- `draw_rate`: 0.40.
- `timeout_rate`: 1.0.
- `red_elimination_win_rate`: 0.0.
- `blue_elimination_win_rate`: 0.0.
- `red_timeout_alive_advantage_rate`: 0.03.
- `blue_timeout_alive_advantage_rate`: 0.57.
- `mav_survival_rate`: 0.0.
- `red_alive_final_mean`: 1.39.
- `blue_alive_final_mean`: 1.93.
- `blue_dead_mean`: 0.07.
- `red_missiles_fired_mean`: 0.07.
- `red_missile_hits_mean`: 0.07.
- `kill_death_ratio`: 0.0435.

5v4:

- `avg_return`: -276.3950.
- `avg_length`: 1000.0.
- `red_win_rate`: 0.33.
- `blue_win_rate`: 0.02.
- `draw_rate`: 0.65.
- `timeout_rate`: 1.0.
- `red_elimination_win_rate`: 0.0.
- `blue_elimination_win_rate`: 0.0.
- `red_timeout_alive_advantage_rate`: 0.33.
- `blue_timeout_alive_advantage_rate`: 0.02.
- `mav_survival_rate`: 0.0.
- `red_alive_final_mean`: 3.18.
- `blue_alive_final_mean`: 2.77.
- `blue_dead_mean`: 1.23.
- `red_missiles_fired_mean`: 1.72.
- `red_missile_hits_mean`: 1.24.
- `kill_death_ratio`: 0.6758.

### Latest Checkpoint

3v2:

- `avg_return`: -102.2243.
- `avg_length`: 373.64.
- `red_win_rate`: 0.0.
- `blue_win_rate`: 1.0.
- `draw_rate`: 0.0.
- `timeout_rate`: 0.0.
- `red_elimination_win_rate`: 0.0.
- `blue_elimination_win_rate`: 1.0.
- `red_timeout_alive_advantage_rate`: 0.0.
- `mav_survival_rate`: 0.0.
- `red_alive_final_mean`: 0.0.
- `blue_alive_final_mean`: 2.0.
- `blue_dead_mean`: 0.0.
- `red_missiles_fired_mean`: 0.0.
- `red_missile_hits_mean`: 0.0.
- `kill_death_ratio`: 0.0.

5v4:

- `avg_return`: -245.1705.
- `avg_length`: 396.2.
- `red_win_rate`: 0.0.
- `blue_win_rate`: 1.0.
- `draw_rate`: 0.0.
- `timeout_rate`: 0.0.
- `red_elimination_win_rate`: 0.0.
- `blue_elimination_win_rate`: 1.0.
- `red_timeout_alive_advantage_rate`: 0.0.
- `mav_survival_rate`: 0.0.
- `red_alive_final_mean`: 0.0.
- `blue_alive_final_mean`: 3.97.
- `blue_dead_mean`: 0.03.
- `red_missiles_fired_mean`: 0.05.
- `red_missile_hits_mean`: 0.03.
- `kill_death_ratio`: 0.0060.

## Best vs Latest Decision

The best checkpoint is preferable to latest.

Reasons:

- Latest collapses in both 3v2 and 5v4, with `blue_win_rate = 1.0`.
- Latest has `red_alive_final_mean = 0.0` in both evaluation configs.
- Best at least reaches timeout in both configs and has nonzero red timeout alive advantage.
- Best has some red missile hits and blue deaths, especially in 5v4, but this is not stable enough to count as a combat baseline.

The best checkpoint is still not a strong combat baseline because 3v2 has only `red_missile_hits_mean = 0.07`, `blue_dead_mean = 0.07`, `red_win_rate = 0.03`, and `mav_survival_rate = 0.0` in the 100-episode evaluation.

## ACMI Observation

ACMI files exported:

- Best: `outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/acmi/best_3v2_episode0.acmi`
- Latest: `outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/acmi/latest_3v2_episode0.acmi`

Best episode0 summary:

- Outcome: timeout.
- Steps: 1000.
- `red_alive_final`: 2.
- `blue_alive_final`: 2.
- `mav_alive`: true.
- Red deaths: `red_2`.
- `red_missiles_fired`: 0.
- `blue_missiles_fired`: 1.
- `red_missile_hits`: 0.
- `blue_missile_hits`: 1.
- `mav_action_saturation_rate`: 0.0.

Latest episode0 summary:

- Outcome: timeout.
- Steps: 1000.
- `red_alive_final`: 1.
- `blue_alive_final`: 2.
- `mav_alive`: true.
- Red deaths: `red_1`, `red_2`.
- `red_missiles_fired`: 0.
- `blue_missiles_fired`: 1.
- `red_missile_hits`: 0.
- `blue_missile_hits`: 1.
- `mav_action_saturation_rate`: 0.4467.
- `uav_action_saturation_rate`: 0.4995.

The ACMI episode0 exports are consistent with a timeout survival pattern rather than an effective red attack pattern. In both exported episodes, red does not fire or hit; blue fires and hits once.

## Final Conclusion

Decision: B. HAPPO surrogate 1M is only a survival baseline.

It is not a usable combat baseline. The training rollout is stable and NaN-free, but it mainly shows timeout survival. The independent evaluation shows that latest collapses, and best is only partially better. Best has small 3v2 red missile hit and blue death rates, but not enough to claim a stable combat strategy.

Do not continue longer training as the next step. The next step should be a minimal combat-oriented reward/curriculum before new algorithm design.
