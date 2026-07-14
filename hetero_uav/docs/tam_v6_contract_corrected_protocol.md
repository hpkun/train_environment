# TAM v6 Contract-Corrected Learnability Protocol

## Scope

The v6 profile is a contract-corrected learnability experiment derived from the
frozen `tam_happo_paper_formula_v5` profile. It does not change the v5 reward
formula, coefficients, missile model, aircraft model, initial geometry, or
Pure HAPPO update hyperparameters.

## Contract Corrections

- Red scripted missile-evasion action replacement is disabled (`teams: none`).
- Every environment step reports requested action, executed control target,
  override status/reason, and selected incoming-missile diagnostics.
- Overridden active transitions remain critic samples but are excluded from
  Actor loss, entropy, KL, ratio, and clip-fraction estimates.
- The observation adds a seven-value incoming-missile state plus a validity
  mask. Its checkpoint contract is `mav_shared_geo_v3_incoming_missile` and is
  intentionally incompatible with legacy 140/700-dimensional checkpoints.
- Reward, fire-candidate, lock, and launch diagnostics use the shared
  paper-assessment target implementation. Match rates are conditional on a
  comparable fire candidate, lock, or launch being present.

## Incoming-Missile State

The physical vector is
`[relative_speed_mps, relative_altitude_m, distance_m,
aircraft_to_missile_ATA_rad, missile_to_aircraft_AA_rad,
closing_speed_mps, t_go_sec]`. Positive relative altitude means the missile is
above the aircraft. ATA is the angle from aircraft velocity to the
aircraft-to-missile line of sight. AA is the angle from missile velocity to the
missile-to-aircraft line of sight. Only a live, assigned, closing missile is
valid; among valid threats the minimum positive time-to-go is selected. No
valid threat produces seven zeros and mask zero.

Normalization divisors are stored in YAML and checkpoint metadata: 1000 m/s,
10000 m, 20000 m, pi, pi, 1000 m/s, and 60 s respectively. Adapter values are
clipped to bounded ranges.

## Method Boundaries

Paper-related elements include role-specific decentralized Actors, a
centralized value function, sequential HAPPO-style policy correction, and the
existing TAM reward categories. The current control interface is a
three-dimensional continuous high-level PID adaptation
`[target_pitch, absolute_target_heading, target_speed]` with a tanh-squashed
Gaussian distribution. It is not the paper's exact four-dimensional,
40-bin action space.

Both MAV and attack UAV use F-16 dynamics in this profile. Heterogeneity is in
role, sensor access, payload, and reward, not aircraft dynamics. MAV constants
not disclosed by the paper remain documented project assumptions. Pure HAPPO
has no GRU, attention critic, TAM network module, or network-failure mask; the
dead-agent active mask is retained.

The default credit mode remains `shared_alive_team_mean`. The optional
`role_local_vector_critic` uses one centralized-state value per controlled
agent and local reward/done/GAE targets. It is an experimental credit-assignment
comparison, not a claim that the shared mode is incorrect.
