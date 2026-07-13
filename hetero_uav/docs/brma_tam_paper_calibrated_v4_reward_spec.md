# BRMA-TAM Paper-Calibrated v4

This mode is a **paper-grounded heterogeneous reward, not exact paper reproduction**.

## Paper-grounded internal ratios

The TAM-HAPPO UAV dense terms use `Height:Speed:Angle:Distance:Dodge = 10:10:15:10:30`, divided by 75. UAV events use normalized `kill=+1`, `death/crash=-1`, and first out-of-zone `-0.5`. Angle and distance each use BRMA-inspired offense minus `0.8 * reverse threat`; they are aggregated over all alive entities independently and are never multiplied together. MAV safety uses `0.5:0.3:0.2` for distance, missile warning/threat, and aspect. MAV support uses `0.6:0.4` for position and shared-not-direct awareness.

## Empirical scales

The papers do not publish a top-level Task/UAV/MAV/Flight mixture for this JSBSim horizon, nor exact normalized MAV death and team-credit constants. Therefore `uav_dense_scale`, `uav_event_scale`, `terminal_scale`, `mav_safety_scale`, `mav_support_scale`, `mav_event_scale`, and `flight_scale` are calibration parameters, never paper claims. Bootstrap YAMLs use unit scales only to collect raw components.

`calibrate_paper_grounded_v4_reward.py` evaluates random actions and supplied checkpoints under the v4 environment, records unscaled episode sums and outcome variables, reports quantiles/nonzero/saturation statistics, and emits safety-dominant, balanced, and event-dominant candidates. Dense capacity is referenced to the paper's `75/200=0.375` relation; MAV top-level alternatives remain explicitly empirical.

## Events and terminal

Kill, death, first OOB, MAV death, capped team credit, and terminal are episode-state guarded. Terminal is `+1` for red elimination win, `-1` for blue elimination win, and zero for mutual elimination. Timeout uses final normalized force difference; MAV-dead/no-blue-loss is capped at a negative result. There is no per-step survival reward.

## Scale transfer and logging

Angle/distance and awareness use means over valid UAV-blue pairs, avoiding linear growth from 3V2 to 5V4. Every raw, scaled, total, component-sum, identity, and outcome field is logged independently. Summary runs retain episode and iteration aggregates; per-step reward CSV remains full-mode only.

Candidate promotion requires finite identity-safe training, nonzero approach geometry, improved blue loss/launch evidence across seeds, no timeout-only local optimum, and MAV safety moving consistently with survival. Otherwise calibration is inconclusive.
