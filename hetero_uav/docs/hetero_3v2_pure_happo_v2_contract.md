# Heterogeneous 3V2 Pure HAPPO V2 Contract

## 1. Scope

`hetero_3v2_pure_happo_v2` is a paper-aligned, not exact-reproduction,
heterogeneous 3V2 environment contract. It reuses the validated V1 physical
path and changes only the red actor observation and role reward.

The environment contains one unarmed red MAV, two armed red attack UAVs, and
two rule-controlled blue attack UAVs. The blue opponent remains
`PaperGreedyOpponent`.

## 2. Fixed simulation and action contract

| Item | Value | Classification |
|---|---:|---|
| JSBSim aircraft dynamics | 6-DoF | PUBLISHED |
| Physics frequency | 60 Hz | PUBLISHED |
| Decision frequency | 5 Hz | PUBLISHED |
| Physics frames per action | 12 | PUBLISHED |
| Episode decisions | 1000 | PUBLISHED |
| Approximate episode time | 200 s | DERIVED |
| Red/blue composition | 1 MAV + 2 UAV vs 2 UAV | PUBLISHED |
| Action | target pitch, target heading, target speed | PUBLISHED |
| Low-level control | PID | PUBLISHED |
| Action dimension | 3 | DERIVED |

V2 does not change action decoding, PID gains, aircraft XML, aircraft model,
initial geometry, termination implementation, or Pure HAPPO.

## 3. Observation contract

The V1 68-dimensional observation is retained in the same order:

1. ego state: 11;
2. two ally slots: 22;
3. two fixed enemy slots (`blue_0`, `blue_1`): 28;
4. nearest incoming missile: 7.

V2 appends five red-known fire-control values:

1. `missile_cooldown_remaining_norm`;
2. `fire_ready`;
3. `own_inflight_any`;
4. `team_inflight_to_enemy_slot_0`;
5. `team_inflight_to_enemy_slot_1`.

The actor dimension is therefore 73. The shared critic concatenates the three
red observations in fixed `red_0`, `red_1`, `red_2` order, so its input is
`73 * 3 = 219` (DERIVED).

Enemy slots never reorder. A dead or unobserved enemy keeps its slot and its
existing alive/valid masking semantics. The new fields use only red launch
times, red ammunition and the red missile list. They do not read blue policy
memory, blue actions, future trajectories, or critic-only truth.

The MAV uses the same input shape. Its own cooldown, ready and own-inflight
values are zero; team target-occupancy values remain available.

## 4. MAV reward

The active role structure is:

```text
R_MAV = R_safety + R_support + R_event
R_safety = 0.5 R_dist + 0.3 R_threat + 0.2 R_aspect
R_support = 0.6 R_pos + 0.4 R_aware
```

The two weight ratios are PUBLISHED.

- `R_dist` is a bounded nearest-threat distance score. The 8 km danger and
  15 km safe normalization distances are DESIGN_CHOICE.
- `R_threat` penalizes an incoming missile, or a blue aircraft already inside
  the unchanged formal launch geometry. Its bounded normalization is
  DESIGN_CHOICE.
- `R_aspect` penalizes the most dangerous blue heading toward the MAV. The
  45-degree normalization is DESIGN_CHOICE.
- `R_pos` is the mean bounded MAV-to-live-attack-UAV support-distance score.
  Its 0.75/4/15/30 km trapezoid is DESIGN_CHOICE.
- `R_aware` is a continuous MAV-to-blue angular awareness score inside the
  existing 80 km MAV sensing range. Its normalization is DESIGN_CHOICE.

`mav_shared_information_metric` remains diagnostic only. It is not included
in `R_support` or `mav_total`.

When the MAV is dead, every dense component is zero. Only the one-step V1
death/event settlement remains.

## 5. Attack UAV reward

The active reward is:

```text
R_UAV =
  10 R_height
  + 10 R_speed
  + 15 R_angle
  + 10 R_distance
  + 30 R_dodge
  + R_event
```

The `10:10:15:10:30` weights are PUBLISHED.

- `R_height` uses 750 m as the published minimum. The 6000 m nominal altitude
  and 10000 m upper normalization are DESIGN_CHOICE.
- `R_speed` uses the published piecewise target/own-speed formula:

```text
1                              if V_blue < 0.5 V_red
2 - 2 V_blue / V_red           if 0.5 V_red <= V_blue <= 1.5 V_red
-1                             if V_blue > 1.5 V_red
```

- `R_angle` uses the formal ATA and TA definitions and the TAM angle form
  `1 - (ATA + AA) / pi`, with `AA = pi - TA`.
- `R_distance` uses the existing TAM distance form: 1 at no more than 5 km,
  exponential decay from 5 to 10 km, and -1 from 10 km onward.
- `R_dodge = R_AM + R_SM`, where `R_AM = -cos(lambda)` and
  `R_SM = (V_M(t-1) - V_M(t)) / V_norm`. `V_norm=1000 m/s` is DESIGN_CHOICE.
  The term is zero without a live incoming missile.

Angle and distance always use `selected_targets[agent_id]`, the same target
used by the unchanged fire gate and automatic launch path.

When an attack UAV is dead, all dense terms are zero. The death transition
retains its one-step event reward.

## 6. Missile and fire-control contract

| Item | Value | Classification |
|---|---:|---|
| Attack range | 14 km | PUBLISHED |
| Per-aircraft attack interval | 25 s | PUBLISHED |
| Attack interval at 5 Hz | 125 decisions | DERIVED |
| PN gain | 3 | PUBLISHED |
| Missile overload limit | 30 g | PUBLISHED |
| Aircraft overload limit | 9 g | PUBLISHED environment specification |
| Aircraft minimum altitude | 750 m | PUBLISHED environment specification |
| Aircraft maximum speed | 400 m/s | PUBLISHED environment specification |
| ATA gate | 60 degrees | DESIGN_CHOICE |
| TA gate | 90 degrees | DESIGN_CHOICE |
| Hit radius | 300 m | DESIGN_CHOICE |
| Scripted missile speed | 600 m/s | DESIGN_CHOICE |
| Arming time | 0.2 s | DESIGN_CHOICE |
| Missile lifetime | 60 s | DESIGN_CHOICE |

V2 does not change the V1 missile kinematics, hit logic, target selection,
duplicate-target suppression, blue opponent, or automatic launch behavior.
The published 9 g, 750 m and 400 m/s values above are reference environment
limits. V2 does not add a new overload, altitude or speed termination/clamping
path; their concrete aircraft behavior and enforcement remain inherited from
V1 and JSBSim.

## 7. Events and termination

V2 retains the V1 event and termination path. Kill, UAV death, MAV death and
out-of-zone constants are DESIGN_CHOICE because the papers do not publish the
exact constants used here. An event is settled once through `newly_dead` and
the step missile event list.

## 8. V1 to V2 changes

V2 changes only:

1. actor observation from 68 to 73 by appending known fire-control state;
2. critic input from 204 to 219;
3. MAV reward from shared-information occupancy shaping to the published
   safety/support role structure;
4. UAV category weights and speed/dodge formulas to the published TAM forms.

V1 remains available through
`uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml`.

## 9. Checkpoint contract

V2 checkpoint metadata is:

```text
formal_contract = hetero_3v2_pure_happo_v2
reward_contract = paper_aligned_role_reward_v3
observation_contract = formal_entity_fire_state_v2
algorithm_contract = pure_happo_sequential_v2
policy_distribution = tanh_squashed_gaussian_raw_action
critic_contract = centralized_shared_scalar_v
gae_contract = separated_termination_truncation
credit_mode = shared_alive_team_mean
actor_obs_dim = 73
critic_state_dim = 219
action_dim = 3
num_agents = 3
```

V1 and V2 checkpoints are intentionally incompatible. Evaluation rejects a
formal-contract, reward-contract, observation-contract or dimension mismatch;
there is no silent truncation or zero-padding.
