# Approach-and-Fire Curriculum Plan

## Current Diagnosis

The current bottleneck is approach-and-fire behavior, not a missing GRU or mask
module.

- The direct-chase oracle reliably fires and hits. In the latest launch-envelope
  diagnostic it fired about two red missiles per 3v2 episode and killed both
  blue aircraft.
- The flat long best checkpoint can fire, but hit quality is unstable and the
  action is highly saturated.
- The flat 50k policy can fire, but diagnostic eval showed zero hits.
- The entity_attention 50k policy is mostly blocked by `out_of_range`, so it has
  not yet learned to approach the engagement envelope.

This means the next useful experiment should make approach-and-fire easier to
learn before adding recurrent memory, random scale masks, or biased masks.

## Easy Geometry Config

New config:

`uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_approach_fire_easy_f16_mav_surrogate.yaml`

It keeps the same:

- action space `[pitch, heading, speed]`;
- missile launch range, AO, TA, lock delay, cooldown and missile dynamics;
- reward mode `happo_ref_v0`;
- blue rule behavior;
- F-16 MAV surrogate and F-16 UAV dynamics;
- observation schema and dimensions;
- max_steps = 1000.

The only curriculum change is initial geometry:

- normal config places blue at latitude `60.20`;
- easy config places blue at latitude `60.07`;
- red UAV positions and headings stay aligned toward blue;
- MAV is kept slightly behind/above as in the existing easy-combat setup.

This is curriculum because it reduces initial approach distance and makes early
launch-envelope exposure more frequent. It does not relax missile rules.

## Short Experiments

Runner:

`python scripts/run_approach_fire_curriculum.py`

Default experiments:

1. `flat_easy_imitation`
   - `policy_arch=flat`
   - `total_env_steps=50000`
   - easy geometry config
   - direct-chase oracle dataset
   - `uav_imitation_coef=0.15`
   - rich logging enabled

2. `entity_easy_imitation`
   - `policy_arch=entity_attention`
   - same steps, config, imitation and diagnostics

Both experiments run the same number of steps so their launch-envelope metrics
are directly comparable.

## Required Metrics

The summary tracks:

- range_ok_rate;
- AO_ok_rate;
- TA_ok_rate;
- lock_ready_rate;
- launch_allowed_rate;
- launch_block_reason distribution;
- red_missiles_fired;
- missile_hits;
- blue_dead;
- action_saturation_rate.

Output:

- `outputs/approach_fire_curriculum_summary.csv`
- `outputs/approach_fire_curriculum_summary.md`

## Decision Rule

If easy imitation improves range/AO/TA rates and produces non-zero fire and
hits, the next step is a 500k curriculum run.

If the policy remains dominated by `out_of_range`, inspect the imitation dataset
and action decoding before adding GRU or masks.

If the policy fires but does not hit, inspect launch quality, TA/AO at fire time,
and target maneuvering before changing reward.
