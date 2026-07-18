# Formal 3V2 Reward V5 Contract

## Scope

`task_aligned_shared_potential_reward_v5` is an independent reward contract
for `hetero_3v2_pure_happo_v2`. It does not replace V4. The V4 and V5 YAML
files select their reward contract explicitly, and their checkpoints cannot be
used interchangeably for resume training.

V5 leaves the 73-dimensional actor observation, 219-dimensional critic state,
3-dimensional action, JSBSim dynamics, PID, sensing, target selection, fire
gate, missile, `PaperGreedyOpponent`, scenario geometry and combat-capability
termination unchanged.

## Shared team reward

Every red actor receives the same scalar:

```text
team_reward = shared_event_reward
            + terminal_reward
            + potential_shaping_reward
```

Pure HAPPO still applies `fixed_three_agent_team_mean`; the mean of three
identical rewards is exactly `team_reward`.

## Shared events

```text
shared_event_raw = 200 * red_kill_count
                   - 200 * red_attack_death_count
                   - 200 * mav_death_count
                   - 100 * out_of_zone_count
shared_event_reward = shared_event_raw / 200
```

A red out-of-zone transition receives only -100 and is not also counted as a
-200 death. A blue loss counts as a red kill only when a real red missile hit
event identifies it. Event magnitudes are PUBLISHED TAM role-event values;
sharing them across the red team and dividing by 200 are DESIGN_CHOICE.

## Terminal reward

```text
outcome_bonus = +1 red_win, -1 blue_win, 0 otherwise

terminal_retention_raw = 30 * (
    (2 - blue_attack_alive)
    - (2 - red_attack_alive)
    - (1 - mav_alive)
)

terminal_reward = outcome_bonus + terminal_retention_raw / 200
```

The BRMA final-loss-difference idea is ADAPTED to heterogeneous 3V2. The
explicit outcome bonus and common 200 scale are DESIGN_CHOICE. Terminal reward
is zero on non-boundary transitions and is settled once. An invalid numeric
episode has no win/loss bonus.

## Role potential

V5 reuses the V4 role formulas without changing their thresholds or weights:

```text
phi_mav = clip((R_safety + R_support) / 1.8, -1, 1)
phi_uav_i = clip((10 R_height + 10 R_speed + 15 R_angle
                  + 10 R_distance + 30 R_dodge) / 75, -1, 1)
phi_team = (phi_mav + phi_uav_1 + phi_uav_2) / 3
```

Dead roles have zero potential. MAV safety/support and UAV category weights
are PUBLISHED; unpublished normalizations remain the same DESIGN_CHOICE as V4.
The shared-information metric remains diagnostic and is not an active reward.

## Potential shaping

```text
potential_gamma = 0.99
potential_beta = 0.25
potential_shaping_reward = 0.25 * (
    0.99 * phi_team_next_effective - phi_team_previous)
```

At reset, `phi_team_previous` is initialized from the initial state using a
fresh, current-state-only potential target map. Attack-UAV potential targets
use the same observable/alive target ranking as formal target selection, but
the map is computed independently from the fire-control
`env.selected_targets` cache. Potential evaluation does not mutate fire gates,
launch state, missiles, controllers or reward settlement state.

V5 dodge potential uses only current incoming-missile geometry. With the
formal fixed-speed missile, `dodge_speed=0`; unlike V4 reward settlement, V5
potential evaluation does not read or write historical missile speed.

A normal transition uses the real next potential and then advances the cache.
A true termination, invalid outcome or numeric anomaly uses
`phi_team_next_effective=0`. A pure time-limit truncation preserves
`phi_team_next_effective=phi_team_next`, matching truncation bootstrap
semantics. Timeout terminal retention is still settled. Gamma, beta and this
potential-shaping choice are DESIGN_CHOICE. The training entry rejects a PPO
gamma different from 0.99.

For a trajectory of `T` transitions, the audit checks the discounted identity:

```text
sum_t gamma^t * shaping_t
  = beta * (-phi_initial + gamma^T * phi_final_effective)
```

## Version isolation

V5 checkpoint metadata records:

```text
reward_contract = task_aligned_shared_potential_reward_v5
potential_gamma = 0.99
potential_beta = 0.25
event_scale = 200.0
shared_team_reward = true
```

Resume and normal evaluation require an exact reward-contract match. The
ranking audit may load compatible actor parameters from V4 or V5 checkpoints,
but never loads their critic, optimizers or training state and labels the run
as a cross-reward actor-only audit.

The ranking audit uses the same paired initial perturbations for every policy,
reports discounted and undiscounted component returns separately, and treats
discounted team return as the primary mathematical metric. Its sorted policy
list is observational, not a hard-coded success order. `red_hits` comes from
missile events while `red_kills` comes from V5 event settlement.

Audit criteria are separated by purpose:

- **Task reward checks** use
  `undiscounted_task_return = undiscounted_shared_event_return +
  undiscounted_terminal_return`. Structural loss comparisons are made only
  between episodes with the same seed and paired initial perturbation. A check
  with no eligible samples or pairs reports `pass=null` (`N/A` in Markdown),
  never a vacuous pass.
- **Policy performance** uses mean discounted team return for checkpoint
  comparison and observational ranking.
- **Potential consistency** uses the discounted telescoping error only.

Every audit episode must have a unique stable-JSON perturbation signature.
The JSON payload reports scenario count, unique perturbation count and the
resulting uniqueness flag; any duplicate perturbation aborts the audit.

Legacy `raw_mav_reward`, `normalized_mav_reward`, `raw_uav_reward` and
`normalized_uav_reward` columns remain for compatibility. Under V5 they are
role-potential diagnostics, not separate actor training rewards. The explicit
V5 columns `mav_dense_diagnostic`, `mav_phi_diagnostic`,
`uav_dense_diagnostic`, `uav_phi_diagnostic` and `shared_training_reward`
should be used for interpretation.

## Classification summary

- PUBLISHED: TAM role weights and attack-UAV event magnitudes.
- ADAPTED: BRMA final loss-difference idea for heterogeneous 3V2.
- DESIGN_CHOICE: shared events, outcome bonus, event scale, potential shaping,
  beta, gamma and unpublished normalizations.
- UNCHANGED: dynamics, action/observation contracts, PID, sensing, fire gate,
  missile, opponent, initial geometry and termination rules.
