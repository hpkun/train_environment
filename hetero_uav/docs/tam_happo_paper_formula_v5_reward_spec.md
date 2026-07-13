# TAM-HAPPO Paper Formula v5

## Claim

`tam_happo_paper_formula_v5` is a **TAM-HAPPO paper-formula implementation with documented JSBSim adaptations; not an exact reproduction of unpublished constants**.

The main reward contains TAM terms only. BRMA-MAPPO Eq. 20-23 is an independent log-only reference and is not added to the v5 total.

## UAV Reward

TAM-HAPPO Table 1 defines

`R_uav_raw = 10 R_H + 10 R_V + 15 R_A + 10 R_D + 30 R_DM + R_E`.

The implementation applies one global numerical scale, `1/200`, to the complete raw value. This gives `+1`, `-1`, and `-0.5` for the published `+200` kill, `-200` death/crash, and `-100` first out-of-zone events. The published ratio `10:10:15:10:30` is not calibrated from episode sums.

- Height, Table 1: `R_H=P_V+P_H`. The paper does not publish enough closed-form detail to map `P_V/P_H` exactly to this JSBSim interface. v5 therefore reuses the existing tested finite-envelope implementation and labels it `jsbsim_height_adaptation`; it records `P_V`, `P_H`, and `R_H` separately.
- Speed, Table 1: target-relative piecewise `R_V`, using the independently selected paper target. A non-finite or zero own speed returns `-1` with an invalid flag.
- Angle, Table 1: `R_A=1-(ATA+AA)/pi` for the same target used by speed and distance.
- Distance, Table 1: `1` through 5 km, `exp[-0.921(R-5)]` for 5-10 km, and `-1` from 10 km. This is independent of the 14 km missile launch protocol.
- Dodge, Table 1: `R_DM=-cos(lambda)+(V_M(t-1)-V_M(t))/V_norm`. It is zero without a real `under_missiles` threat. Missile-speed state resets every episode. `V_norm` is not specified by the paper and is a documented JSBSim normalization assumption.
- Event, Table 1: unique shooter kill, alive-before to dead-after death/crash, and first real horizontal OOB. Dead-before agents receive zero on every subsequent step.

No pitch, roll, generic flight, survival-step, v3 attrition, or v3 terminal reward enters the v5 total.

## Target Selection

TAM-HAPPO Sec. 2.4, Eq. 10-12 defines an auxiliary target assessment. For every alive blue entity, v5 computes angle, distance, relative-height, and relative-speed terms and applies fixed weights `0.35:0.25:0.20:0.20`. These weights select the reward target only; they do not alter Table 1 reward weights or fire control.

The paper does not publish the relative-height and relative-speed normalization constants. The YAML labels these values as project assumptions. Reward target, closest target, lock target, launch target, and match flags are logged independently.

## MAV Reward

TAM-HAPPO Table 1 defines:

- `R_mav = R_safety + R_support + R_event`
- `R_safety = 0.5 R_dist + 0.3 R_threat + 0.2 R_aspect`
- `R_support = 0.6 R_pos + 0.4 R_aware`
- `R_event = -I(death) C_d + capped team contribution`

`R_dist`, real missile warning, and the blue-attacker-to-MAV aspect sum follow Table 1. Both aspect raw sum and per-blue mean are logged; the published sum is used for training.

The paper describes `d_b` relative to the evolving battlefield center but does not provide a complete JSBSim construction. v5 uses the alive attack-UAV centroid and alive-blue centroid to place a dynamic ideal support point behind the UAV formation. The published `R_pos(d_b)` piecewise function is then applied to distance from that point. Rear projection, lateral offset, and support distance are logged. This is a **paper-trajectory-semantic JSBSim adaptation**, not an exact paper formula.

Awareness is counted only for valid UAV-blue pairs that are not directly visible to the UAV but are visible through the alive MAV. MAV survival alone produces no awareness reward.

`C_d`, team contribution per kill, the contribution cap, and support-position adaptation distances are **not specified by the paper**. They are the only values eligible for empirical constraint selection. They must not be presented as paper coefficients.

## BRMA Reference

BRMA-MAPPO Sec. 2.5, Eq. 20-23 is implemented separately:

- Eq. 20: `T_a(q_LOS)` with boundaries 4, 15, and 35 degrees and the published piecewise values.
- Eq. 21: `T_d=1` through 15 km, otherwise `exp(1-D/15)`.
- Eq. 22: `r_adv_i=sum_j[T_a(i,j)T_d(i,j)-0.8T_a(j,i)T_d(j,i)]` over all alive enemies.
- Eq. 23: `r_end=0` when alive counts match; otherwise `30(N_red-N_blue)`.

The key entity is the pair with maximum absolute Eq. 22 contribution. Raw sum and per-target mean are both recorded. `brma_overlay_enabled=0` in the main v5 YAML.

## State, Identity, and Scale

Each step records `alive_before`, `alive_after`, `death_transition`, and `dead_before`. Dense and event terms are computed only for alive-before agents. Episode accumulators are separated by environment, episode UID, and agent in the existing runner/logger path.

The reward path total and the independently reconstructed scaled component sum are computed separately. `identity_error=total-reconstructed_sum` must remain within `1e-8`; changing a stored component without changing the reward path makes the identity check fail.

Paper raw sums are preserved. Per-target and per-role means are additional cross-scale diagnostics; they do not modify 3V2 or 5V4 rewards. Consequently, 3V2 and 5V4 raw totals are not expected to be equal.

## Timeout and Censoring

Environment timeout is recorded only when the environment reaches 1000 decisions. A collector that stops before an environment terminal writes `censored=1`, `environment_timeout=0`, and `terminal_observed=0`. A true early terminal is not censored.

## v4 Correction

v4 estimated top-level scales from short episode accumulated magnitudes. That mixed published step coefficients with episode-duration-dependent totals. v5 keeps all published TAM coefficients fixed, uses a single `1/200` linear scale, and limits calibration to unpublished MAV constants through explicit feasibility constraints.

## Protocol Boundaries

The 3V2 and 5V4 v5 configs preserve the existing geometry, 5 Hz decision rate, 1000-decision episode, 14 km/25 s launch protocol, PN gain 3, and 30 g overload limit. Reward, fire-control target selection, and missile protocol remain separate concepts.
