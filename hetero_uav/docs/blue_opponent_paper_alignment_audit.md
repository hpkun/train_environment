# Blue Opponent Paper Alignment Audit

## Purpose

This audit is part of environment completion, not training. The goal is to
check whether the blue rule opponent matches the intended BRMA-MAPPO baseline
environment and the TAM-HAPPO inspired heterogeneous MAV/UAV setting.

The current `greedy_fsm` 50-step rollout stays in patrol under paper-aligned
3v2/5v4 initial geometry. That is not enough evidence to change initial
geometry. First we need to define what the blue opponent should be able to do,
then audit the current implementation against that expected logic.

`greedy_fsm` is not final opponent behavior for paper results.

## BRMA-MAPPO Blue Opponent Expectations

Repository evidence from `my_uav_env/env.py` and the copied BRMA-style
`uav_env/JSBSim/env.py` supports these expectations:

- rule-based opponent: confirmed by the BRMA-style scripted environment path.
- target acquisition: confirmed through environment observation and missile
  launch scans.
- target assignment or target selection: partially confirmed. The environment
  has engaged-target deconfliction, but the current script-layer opponent does
  not assign targets across blue aircraft.
- pursuit / intercept maneuver: confirmed as high-level action intent, but the
  exact original blue maneuver controller is not fully recovered here.
- missile launch handled by environment fire-control or scripted condition:
  confirmed by `_check_missile_launch`.
- missile evasion: confirmed as an environment automatic layer for both teams.
- GCAS / safety handling: confirmed for Blue in BRMA-style env as
  `enable_gcas_for_blue`.
- boundary / altitude safety: confirmed at environment level.
- no trainable blue policy during red training: consistent with current
  scripted-opponent setup.

Items not directly traceable to a complete original blue controller are marked
as uncertain in the gap table.

## TAM-HAPPO Blue Opponent Expectations

Repository notes describe the heterogeneous target setting as red MAV/UAV
composition transfer against mostly homogeneous blue UAVs. Based on available
project documentation and current code, the expected blue-side properties are:

- greedy / rule-based / finite-state controller: expected, but the exact paper
  controller is uncertain in the repository.
- blue side is homogeneous UAVs: implemented in paper-aligned configs.
- blue does not have MAV shared observation: implemented in V2, where MAV
  shared tracks are red-side support only.
- blue should still have enough local situation information to select
  maneuvers: expected, but current visibility diagnostics show this is not yet
  guaranteed.
- blue chooses actions from high-level maneuver candidates or finite-state
  rules: expected, but candidate maneuver scoring is not implemented.
- blue UAVs carry missiles: implemented through attack UAV missile counts.
- blue should be a stable adversary, not passive patrol-only behavior:
  expected; currently unresolved because no blue direct track appears in the
  100-step paper-aligned visibility diagnostic.

No original heterogeneous paper text was found in the current repository that
fully specifies the blue finite-state controller. Therefore several TAM-HAPPO
items remain uncertain and must not be fabricated.

## Current Implementation

- `zero`: script-layer no-op opponent for smoke/debug.
- `random`: random high-level actions in `[-1, 1]`.
- `rule_nearest`: selects nearest non-zero enemy observation and points toward
  it with fixed attack speed.
- `greedy_fsm`: diagnostic finite-state opponent with `evade`,
  `recover_altitude`, `attack_mav_priority`, `attack_nearest`,
  `search_acquire`, and legacy `patrol`.
- Environment missile/fire-control: automatic BRMA-style launch logic.
- Environment evasion: automatic missile warning response layer.
- Environment GCAS: Blue GCAS exists in the BRMA-style environment path.
- Blue observation source: own local observation only.
- V2 blue visibility: direct-only; MAV shared observation supports red UAVs,
  not blue UAVs.
- Current visibility diagnostic result: blue never observed red in 100-step
  paper-aligned diagnostics.
- Visibility asymmetry is currently possible: red may benefit from MAV shared
  tracks while blue remains direct-only.

## Gap Analysis Table

| expected capability | BRMA evidence | TAM-HAPPO evidence | current implementation | gap | recommended action |
|---|---|---|---|---|---|
| target visibility/acquisition | confirmed by observation/fire-control scan | expected but exact paper text uncertain | blue direct-only visibility; current paper-aligned diagnostic has no blue track | yes | audit geometry/range before training |
| target assignment | engaged-target deconfliction exists in fire-control | expected/uncertain | no script-layer target assignment | yes | add target assignment after branch diagnostics |
| target prioritization | nearest target selection exists in fire-control | MAV priority plausible but uncertain | `greedy_fsm` can prioritize MAV if roles/types observed | partial | validate role/type availability in blue obs |
| pursuit/intercept | high-level action and PID support it | expected | rule_nearest and greedy_fsm produce pursuit intent | partial | test controlled visible-target cases |
| patrol/search behavior | boundary helpers exist in original BRMA code | expected | `search_acquire` is an initial high-speed contact intent | partial | validate with horizon-sweep visibility diagnostics |
| missile launch interface | confirmed automatic fire-control | blue UAVs carry missiles | environment owns launch | no low-level gap | keep scripted launch layer unchanged |
| missile evasion | confirmed environment layer | expected | environment handles evasion; greedy_fsm only adds high-level intent | no low-level gap | keep unchanged |
| altitude/GCAS safety | Blue GCAS confirmed | safety expected/uncertain | BRMA-style Blue GCAS exists | no current change | do not add MAV GCAS here |
| boundary handling | battlefield boundary exists | expected | environment handles bounds; script has no explicit boundary search | partial | audit before adding search behavior |
| finite-state transition | uncertain | expected | initial greedy_fsm states exist | partial | add transition diagnostics |
| candidate maneuver scoring | situation reward utilities exist, not blue policy | expected/uncertain | not implemented | yes | defer until target acquisition is clear |
| diagnostic observability | launch/visibility diagnostics exist | required | scripts now report coverage and gaps | partial | continue adding controlled branch tests |
| paper-aligned geometry compatibility | not directly specified | required for protocol | blue has no visible red in 100-step diagnostic | unresolved | record as protocol decision, do not blindly change geometry |

## Decision Principles

- Do not immediately make `greedy_fsm` the default opponent.
- Do not immediately run training.
- Do not change initial geometry solely because `greedy_fsm` is patrol-only.
- First verify whether blue has reasonable search/acquisition logic.
- If blue has no perception, the policy should eventually have search/contact
  logic instead of passive patrol-only behavior.
- If initial geometry prevents blue visibility for 100 steps, record it as an
  environment protocol issue before changing initial states or ranges.
- Changes should enter diagnostic scenarios first, not training scenarios.
- `greedy_fsm` remains not final opponent behavior until validation is complete.

## Recommended Staged Fix

B1. Blue opponent alignment audit.

B2. Controlled branch tests for `greedy_fsm`.

B3. Target acquisition/search behavior and horizon-sweep validation.

B4. Paper-aligned geometry/range decision.

B5. Target assignment / candidate maneuver scoring.

B6. Only after validation, decide whether to use `greedy_fsm` for training.
