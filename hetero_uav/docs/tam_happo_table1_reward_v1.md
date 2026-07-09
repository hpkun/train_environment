# TAM-HAPPO Table 1 Reward v1

`tam_happo_table1_v1` is a diagnostic reward mode for checking whether the
current heterogeneous JSBSim 3V2/5V4 environment can learn kill-based
cooperation under a reward structure aligned with TAM-HAPPO Table 1.

## Active Terms

Attack UAV reward:

```
R_UAV = w_H R_H + w_V R_V + w_A R_A + w_D R_D + w_DM R_DM + R_E
```

- `R_V` follows the Table 1 piecewise speed reward with `reference_speed_mps`
  from YAML.
- `R_A = 1 - (ATA + AA) / pi`; the reward target is selected by best
  `0.6 * R_A + 0.4 * R_D` over alive blue aircraft.
- `R_D` follows the Table 1 distance reward in kilometers:
  `1` for `R <= 5`, `exp(-0.921 * (R - 5))` for `5 < R < 10`, and `-1`
  for `R >= 10`.
- `R_DM = R_AM + R_SM` is active only when an incoming missile or missile
  warning is available. Missing missile geometry logs
  `tam_table1_uav_missing_dodge_geometry=1` and contributes zero.
- UAV event terms use the Table 1 explicit values in YAML:
  `kill_enemy=+200`, `death=-200`, and `first_out_of_zone=-100`.

MAV reward:

```
R_MAV = R_safety + R_support + R_event
```

- `R_safety = 0.5 R_dist + 0.3 R_threat + 0.2 R_aspect`.
- `R_support = 0.6 R_pos + 0.4 R_aware`.
- `R_event = -I(MAV death) * C_d + capped team contribution`.

`C_d`, `team_credit_per_kill`, and `team_credit_cap` are implementation
settings for the symbolic Table 1 MAV event terms. They are configured in YAML;
this implementation does not claim the paper provides these exact constants.

## JSBSim Adaptation Boundaries

`R_H` is a JSBSim safety adaptation of Table 1 `R_H = P_V + P_H`.
Normal safe flight returns zero, while unsafe altitude or excessive vertical
speed receives negative shaping. This avoids a persistent positive dense reward
from normal flight.

`R_pos` uses a battlefield-center support anchor by default. This is a JSBSim
adaptation of the Table 1 support-position concept and avoids using the blue
centroid as an anchor, which can pull the MAV toward the enemy formation.

## Exclusions

This reward does not use BRMA-MAPPO active reward terms. BRMA `r_adv` and
`r_end` are logged only as `tam_table1_*_brma_*_log`.

This reward does not add:

- BRMA terminal `30 * (N_red - N_blue)`;
- v7 role-weighted terminal;
- red win timeout bonus;
- survival advantage bonus;
- MAV alive terminal bonus;
- launch, fire, MAV-shared launch, or MAV-shared hit bonuses.

## Non-Modified Environment Mechanics

This mode does not modify missile dynamics, hit model, missile launch gate,
range/AO/TA/lock/cooldown/deconfliction logic, PID, aircraft XML, blue rule,
action space, observation space, or training algorithm logic.
