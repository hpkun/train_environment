# BRMA-TAM Role-Situation v3 Reward Specification

## Research scope

`brma_tam_role_situation_v3` is a paper-inspired heterogeneous reward for 3V2 training and 5V4 zero-shot scale transfer. It combines TAM-HAPPO's MAV/UAV role separation with BRMA-MAPPO-inspired multi-entity offensive and defensive situation summaries. It is not an exact reproduction of either paper.

The environment contract fixes MAV shared geometry observations, a missile-free MAV, attack UAVs, and the isolated TAM missile protocol: 14 km launch range, 25 s attack interval, proportional navigation with gain 3, and a 30 g numerical overload limit. The missile implementation remains a paper-aligned kinematic approximation.

## Reward identity

For every alive-before red agent,

```text
r_total = r_common + r_role_encoded + r_flight_encoded
r_common = r_attrition + r_terminal
```

The team state is

```text
J = blue_loss_fraction
    - attack_uav_loss_weight * attack_uav_loss_fraction
    - mav_loss_weight * mav_loss_indicator
r_attrition = attrition_scale * delta(J)
```

The terminal component is applied once. Mutual elimination is neutral; decisive elimination and timeout advantage use the explicit YAML values. Episode reset clears loss deltas and terminal state.

## UAV situation

Each alive attack UAV evaluates every alive blue entity. Pair quality multiplies angle, distance, and attacker-speed modulation. Local offense and local threat use temperature-controlled softmax aggregation. Team coverage averages the best UAV pressure per blue entity; team exposure averages the worst blue pressure per UAV. The role signal is

```text
S_uav = local_weight * (local_offense - threat_weight * local_threat)
        + team_weight * (team_coverage - team_exposure)
```

Coverage and exposure are means, not entity-count sums, which prevents automatic doubling when 3V2 is expanded to an equivalent 5V4 state. Nearest-target progress was removed because it creates target-switch discontinuities and does not express multi-entity coverage.

## MAV role

MAV marginal information counts shared-but-not-direct tracks and weights them by combat relevance. Support position combines a distance band around the attack-UAV centroid with a rearward projection relative to the blue centroid. MAV threat aggregates blue-to-MAV pair quality and missile warning. The configured weighted sum is clipped before role scaling.

No launch, hit, lock, fire-control gate, or per-step survival signal is directly rewarded. Those outcomes remain consequences of geometry, policy action, and the fixed environment protocol.

## Flight and role encoding

Flight uses the existing pitch, roll, and velocity components, clipped before role-specific scaling. Role contributions are encoded using alive-before counts so their Pure HAPPO team mean retains the intended role contribution when agents die on the current transition:

```text
MAV encoded = N_red_before * MAV scaled role
UAV encoded = N_red_before / N_uav_before * UAV scaled role
```

The same rule applies to role-specific flight. A current-step death is included; an agent dead before the step is excluded.

## Logging contract

Raw fields describe unscaled geometry or role quantities. Scaled fields apply YAML coefficients. Encoded fields include alive-before role-size compensation.

Iteration `effective_*` values are alive-before sample means, not per-step means with absent roles filled by zero. The logger maintains a separate sum and count for every field. Common/task/total fields use all alive-before red samples, UAV fields use only alive-before attack-UAV samples, and MAV fields use only alive-before MAV samples. A role that has no samples in a rollout is written as numeric `0.0`. Counts are reset at every rollout boundary.

`episode_reward_components.csv` contains one row per red agent. Sums include only that agent's alive-before transitions. `final_j_combat` is the final valid v3 value, while `max_abs_identity_error` is a maximum rather than a sum. Accumulators are isolated by environment, episode, and agent.

## Static and scale criteria

Static checks require favorable tail geometry to outrank side, head-on, distant, and tailed geometry; concentrated exposure to be penalized; distributed coverage to be preferred; marginal MAV information and safe rear support to be positive; dangerous forward support to be negative; abnormal flight to be penalized; and terminal outcomes to follow the configured ordering.

The deterministic UAV ordering contract is `tail > side_rear > head_on > far_neutral > tailed`, with a negative tailed situation. Team tests compare distributed and concentrated offense matrices and local versus team-wide exposure matrices. The MAV ordering contract is `safe_shared > safe_no_shared > far_no_support > dangerous_forward`; the dangerous-forward value must be negative. Complete task return, rather than terminal bonus alone, is ordered across decisive win, partial attrition, timeout, partial red loss, and red elimination.

The scale test constructs a symmetric 3V2 geometry and duplicates its positions, velocities, headings, and track visibility into 5V4. It compares pair quality, local offense/threat, normalized coverage/exposure, UAV situation, MAV information/support/threat, and role/flight contributions after Pure HAPPO alive-before team averaging. Exact symmetric quantities require absolute error at most `1e-6`. In non-identical entity sets, softmax aggregation can change because normalization is over a different multiset; such cases must report the measured error rather than claim exact invariance.

## 200K probe gate

A multi-seed 200K probe is allowed only after contract-negative tests, static ordering tests, 3V2/5V4 scale tests, reset tests, a current-HEAD PPO update smoke, short multi-episode logging smoke, 5V4 construction smoke, and full-mode CSV validation all pass. All required v3 cells must be present, numeric, finite, and satisfy reward identity within `1e-6`.
