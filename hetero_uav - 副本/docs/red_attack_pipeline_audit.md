# Red Attack Pipeline Audit

## Why This Audit Was Needed

The F-16 MAV surrogate HAPPO reference 1M run completed without NaN and learned a survival-like rollout pattern, but the training log showed `red_missiles_fired = 0` and `missile_hits = 0`. The independent checkpoint evaluation and ACMI exports also showed weak or absent red attack behavior. Before changing reward or continuing training, the red attack chain had to be checked directly.

This audit does not modify reward, missile dynamics, PID, action space, observation dimensions, aircraft XML, or HAPPO policy/trainer code.

## Red vs Blue Fire Chain

The environment fire-control scan iterates over `self.agent_ids`, so red and blue agents are both scanned by `_check_missile_launch`.

Runtime missile configuration is correct:

- `red_0`: MAV, F-16 surrogate, 0 missiles.
- `red_1`: attack UAV, F-16, 2 missiles.
- `red_2`: attack UAV, F-16, 2 missiles.
- `blue_0`: attack UAV, F-16, 2 missiles.
- `blue_1`: attack UAV, F-16, 2 missiles.

The environment does not require a policy-supplied `selected_target`. Fire-control selects the closest unengaged enemy internally, subject to range, AO, TA, lock delay, cooldown, target alive, and missile count.

Initial red observations contain enemy tracks:

- `red_0`: enemy observed mask sum 2.0.
- `red_1`: enemy observed mask sum 2.0.
- `red_2`: enemy observed mask sum 2.0.

Logging fields are present for red:

- `info[agent]["missiles_fired_this_step"]`.
- `info["__missile_term__"]`.
- `info["__launch_diag__"]`.
- `info["__launch_quality_step__"]`.

Conclusion: the basic red fire-control and logging chain is wired.

## Scripted Red Oracle

Four oracle cases were tested for 20 episodes each.

`red_brma_rule_vs_blue_zero` and `red_brma_rule_vs_blue_brma_rule` did not fire. This indicates the observation-rule policy used in the script is not a sufficient red attack oracle under the current action semantics.

`red_direct_chase_vs_blue_zero`:

- `red_missiles_fired_mean`: 2.0.
- `red_missile_hits_mean`: 2.0.
- `blue_dead_mean`: 2.0.
- `red_win_rate`: 1.0.
- `first_red_fire_step_mean`: 143.0.

`red_direct_chase_vs_blue_brma_rule`:

- `red_missiles_fired_mean`: 2.25.
- `red_missile_hits_mean`: 2.0.
- `blue_dead_mean`: 2.0.
- `red_win_rate`: 1.0.
- `first_red_fire_step_mean`: 257.0.

Conclusion: a scripted red direct-chase oracle can fire and hit. The red fire chain is not globally broken.

## Launch Envelope Probe

A forced initial state placed `red_1` behind `blue_0`, within launch range and with favorable aspect.

Probe result:

- `red_fired_total`: 2.
- `red_hits_total`: 2.
- `any_launch_envelope_satisfied`: true.
- `any_fire_while_envelope_satisfied`: true.
- `red_uav_can_fire_in_theoretical_envelope`: true.

Conclusion: a red UAV can fire when it is explicitly placed inside the launch envelope. No unit or normalization issue was found by this probe.

## HAPPO Engagement Geometry

The 1M best/latest checkpoints were evaluated for 20 episodes.

Best checkpoint:

- `launch_range_rate`: 0.4717.
- `launch_angle_rate`: 0.0.
- `launch_envelope_rate`: 0.0.
- `red_missiles_fired_total`: 0.
- `action_saturation_rate`: 0.0.
- `speed_action_mean`: -0.4392.
- Conclusion: policy avoids or never reaches engagement.

Latest checkpoint:

- `launch_range_rate`: 0.7216.
- `launch_angle_rate`: 0.0.
- `launch_envelope_rate`: 0.0.
- `red_missiles_fired_total`: 0.
- `action_saturation_rate`: 1.0.
- `speed_action_mean`: -0.9453.
- Conclusion: policy avoids or never reaches engagement.

Both checkpoints sometimes enter launch range, but neither satisfies the launch angle/aspect condition, so neither enters the full launch envelope. This explains why red fire remains zero for HAPPO despite the environment being able to fire red missiles.

## Final Decision

`environment_attack_pipeline_status`: `red_fire_chain_working_policy_not_engaging`

Primary issues:

- `policy_avoids_engagement`
- `initial_geometry_too_hard`

Next action:

- `D`: add combat reward/curriculum

This should be a minimal combat-oriented reward/curriculum step, not longer training and not a new algorithm yet.

