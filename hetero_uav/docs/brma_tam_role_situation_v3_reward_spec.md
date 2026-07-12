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

Iteration `effective_*` values are means over actual rollout transitions: common fields over all alive-before red agents, UAV fields over alive-before attack UAVs, and MAV fields over alive-before MAVs. Empty groups are numeric zero.

`episode_reward_components.csv` contains one row per red agent. Sums include only that agent's alive-before transitions. `final_j_combat` is the final valid v3 value, while `max_abs_identity_error` is a maximum rather than a sum. Accumulators are isolated by environment, episode, and agent.

## Static and scale criteria

Static checks require favorable tail geometry to outrank side, head-on, distant, and tailed geometry; concentrated exposure to be penalized; distributed coverage to be preferred; marginal MAV information and safe rear support to be positive; dangerous forward support to be negative; abnormal flight to be penalized; and terminal outcomes to follow the configured ordering.

Equivalent 3V2 and 5V4 constructions must preserve corresponding local offense/threat, normalized coverage/exposure, MAV role magnitude, and alive-before team-mean role contributions within numerical tolerance.

## 200K probe gate

A multi-seed 200K probe is allowed only after contract-negative tests, static ordering tests, 3V2/5V4 scale tests, reset tests, a current-HEAD PPO update smoke, short multi-episode logging smoke, 5V4 construction smoke, and full-mode CSV validation all pass. All required v3 cells must be present, numeric, finite, and satisfy reward identity within `1e-6`.
