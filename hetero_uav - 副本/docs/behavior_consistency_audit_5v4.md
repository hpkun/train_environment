# 5v4 Behavior Consistency Audit

## Purpose

The 5v4 zero-shot metrics are successful, but the fixed ACMI rollout raises
behavior-level questions:

- the MAV appears to fly forward and does not visibly turn back;
- blue aircraft look slow to turn and chase;
- it is unclear whether blue targets the MAV or treats all red aircraft equally.

This audit is read-only. It does not change reward, missile logic, PID,
aircraft XML, action space, observation dimensions, or training code.

## Blue Target Selection

The active blue opponent is `brma_rule`, which delegates to the parent
`rule_based_agent.blue_coordinated_actions`. Its target assignment does include
`red_0` because it iterates over all alive red slots from `death_mask`.

The blue rule policy does not implement explicit role priority:

- no MAV priority;
- no armed-UAV priority;
- no reward-aware target assignment;
- alive red tracks are scored by track quality, range, angle, and target aspect;
- assignments are greedily deconflicted so multiple blue aircraft avoid
  selecting already engaged targets.

The environment missile launch logic is also role-agnostic. For each blue
shooter it scans `self.red_planes`, so `red_0` MAV can be locked and shot if it
is alive, unengaged, closest among eligible targets, and inside range/AO/TA
launch conditions.

## 20-Episode 5v4 Audit Result

Input:

- checkpoint: `outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt`;
- config: `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_happo_ref_v0_f16_mav_surrogate.yaml`;
- episodes: `20`;
- output: `outputs/happo_geometry_curriculum_100k/normal_50k/behavior_audit_5v4/`.

Blue target statistics:

- nearest red distribution:
  - `red_1`: 10231
  - `red_2`: 6507
  - `red_0`: 5587
  - `red_3`: 7964
  - `red_4`: 3200
- inferred heading-aligned target distribution:
  - `red_1`: 2495
  - `red_2`: 773
  - `red_3`: 17708
  - `red_4`: 8763
  - `red_0`: 3750
- missile target distribution:
  - `red_1`: 5
  - `red_2`: 14
- `time_blue_nearest_target_is_mav_rate`: `0.1668`;
- `time_blue_chosen_target_is_mav_rate`: `0.1120`;
- `blue_missile_target_mav_count`: `0`;
- `blue_missile_target_uav_count`: `19`;
- classified `blue_targeting_behavior`: `uav_targeted`.

Interpretation: blue can target the MAV by code path, but in these 20 episodes
its actual missile targets were red UAVs, not `red_0`.

## Episode-0 ACMI Consistency

The fixed ACMI summary does not store target IDs, so the same checkpoint/config
was re-run with seed `0` using the behavior audit script.

Episode-0 inferred result:

- nearest red distribution includes MAV, but not dominantly:
  - `red_0`: 341
  - `red_1`: 513
  - `red_2`: 133
  - `red_3`: 756
- inferred heading-aligned target distribution:
  - `red_0`: 78
  - `red_1`: 146
  - `red_2`: 29
  - `red_3`: 1091
  - `red_4`: 399
- blue missile target distribution:
  - `red_2`: 1
- `blue_missile_target_mav_count`: `0`;
- `blue_missile_target_uav_count`: `1`;
- classified `blue_targeting_behavior`: `uav_targeted`.

Thus, in the ACMI-like episode, blue did not fire at the MAV. The visible
blue pursuit is better interpreted as rule-based pursuit of red aircraft
according to nearest/geometry scoring, with missile fire directed at a UAV.

## MAV Behavior

20-episode MAV statistics:

- `mav_frontmost_rate`: `0.1384`;
- mean distance from MAV to nearest blue: `29256.9 m`;
- minimum distance from MAV to nearest blue: `780.9 m`;
- mean distance from MAV to red UAV formation center: `22846.8 m`;
- `mav_heading_to_blue_rate`: `0.2181`;
- `mav_turn_back_rate`: `0.0`;
- `mav_action_mean`: `[0.2330, -0.0164, 0.0422]`;
- `mav_action_saturation_rate`: `0.00028`;
- `mav_survival_rate`: `1.0`;
- classified `mav_role_behavior`: `inconclusive`.

The MAV is not frontmost most of the time under the audit definition, but it
also does not behave like a clean rear support orbit. The low turn-back rate
and low heading-to-blue rate match the observed "keeps going forward" behavior.
This is better described as forward-survival / loose support behavior, not a
strict paper-style rear support trajectory.

## Blue Pursuit Behavior

20-episode blue pursuit statistics:

- `blue_heading_to_nearest_red_rate`: `0.6031`;
- mean heading error to nearest red: `61.94 deg`;
- mean turn rate: `0.83 deg/env-step`;
- `blue_action_saturation_rate`: `0.3762`;
- mean speed: `315.9 m/s`;
- classified `blue_pursuit_behavior`: `inconclusive`.

For seed-0 episode audit:

- `blue_heading_to_nearest_red_rate`: `0.7510`;
- mean heading error to nearest red: `39.70 deg`;
- mean turn rate: `0.71 deg/env-step`;
- classified `blue_pursuit_behavior`: `responsive`.

The visually slow turn is mainly consistent with the current BRMA rule and PID
dynamics: blue pursuit uses limited delta-heading style commands and bank/GCAS
safety dampening. It is not evidence that blue cannot see or select the MAV.

## Relation To The Heterogeneous MAV/UAV Paper

Existing project paper-grounded notes state:

- 3v2 red is `1 MAV + 2 UAV`; 5v4 red is `1 MAV + 4 UAV`;
- MAV carries no missiles;
- MAV provides battlefield information and mission guidance while prioritizing
  its own safety;
- UAVs attack and launch missiles;
- representative paper behavior has the MAV rearward or circling safely while
  UAVs engage.

The paper-grounded notes do not define a hard trajectory constraint that forces
the MAV to remain behind every UAV at all times. Therefore:

- current 5v4 metrics can still be reported as zero-shot attack-transfer
  success;
- current MAV trajectory should not be claimed as a strict reproduction of the
  paper's rear-support behavior;
- this is a behavior-consistency limitation, not a failure of the measured
  5v4 zero-shot indicators.

## Conclusion

The 5v4 zero-shot metrics remain valid:

- `red_win_rate=0.93`;
- `mav_survival_rate=0.99`;
- `red_missiles_fired_mean=2.59`;
- `red_missile_hits_mean=2.38`;
- `blue_dead_mean=2.33`.

Behavior-level limitation:

The learned MAV policy currently shows forward-survival / loose support rather
than a clearly paper-aligned rear support orbit. Blue does include MAV in its
candidate target set, but actual blue missile targets in the audit were UAVs.
