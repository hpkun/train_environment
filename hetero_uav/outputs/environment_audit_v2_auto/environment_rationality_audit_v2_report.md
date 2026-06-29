# Environment Rationality Audit V2 Report

## Run Scope
- requested episodes: 2
- max_steps: 1000
- jitter: True
- jitter_seeds: 3
- No RL training was run.
- No reward, missile dynamics, launch gate, PID, blue rule, action/observation dimension, aircraft XML, engine XML or trainer logic was modified.

## 1. Full-state chase vs obs-limited chase
- full-state straight chase: {'episodes': 18, 'red_fire': 28, 'red_hit': 24, 'blue_fire': 4, 'blue_hit': 4}
- obs-limited chase with MAV shared track: {'episodes': 18, 'red_fire': 0, 'red_hit': 0, 'blue_fire': 0, 'blue_hit': 0}
- obs-limited chase against zero-action blue: {'episodes': 18, 'red_fire': 20, 'red_hit': 20, 'blue_fire': 6, 'blue_hit': 6}
- obs-limited action decisions used target_source counts: {'': 85658, 'mav_shared': 20346, 'direct': 12330}. The empty source rows are mostly MAV loiter or no-observed-target fallback rows.
- If full-state succeeds while obs-limited fails, prioritize observation schema, MAV shared track usability or policy representation before missile dynamics.

## 2. True oracle launch-window vs straight chase
- true oracle launch-window vs blue rule: {'episodes': 18, 'red_fire': 24, 'red_hit': 24, 'blue_fire': 2, 'blue_hit': 2}
- true oracle launch-window vs blue zero: {'episodes': 18, 'red_fire': 24, 'red_hit': 24, 'blue_fire': 4, 'blue_hit': 4}
- Oracle policy explicitly scores launch-window metrics and uses altitude-safe pitch logic; compare its crash/fire/hit rows with straight chase in CSV.

## 3. Where RL non-firing is more likely to be blocked
- red largest blocked counter: track_unobserved_blocked=4832608.0
- blue largest blocked counter: engaged_blocked=5597212.0
- red MAV-shared track candidate sum: 2184484.0
- red direct track candidate sum: 852420.0
- Use `launch_gate_by_side_v2.csv` and `blocked_reason_by_side_v2.csv`; missing fields are blank, not silently zero-filled.

## 4. Red/blue launch-gate symmetry
- The gate code is shared, but actual counts differ because red has MAV role blocking and MAV-shared track paths while blue has different observations and rule policy behavior.
- V2 records real `range_ok_pairs`, `ao_ok_pairs`, `ta_ok_pairs`, `boresight_ok_pairs`, `geometry_ok_pairs`, track candidates and block counters.

## 5. MAV shared track
- MAV shared track participation is measured by `red_mav_shared_track_candidates` and obs-limited action decisions with `target_source=mav_shared`.
- In this run, red launch diagnostics counted `red_mav_shared_track_candidates=2184484` and `red_direct_track_candidates=852420`; MAV shared tracks exist in the launch candidate stream.
- If these remain zero while full-state chase succeeds, the shared-track observation path is suspicious for RL learnability.

## 6. Blue rule target preference
- See `blue_rule_strength_v2.md`, `blue_target_distribution_v2.csv` and `blue_first_launch_geometry_v2.csv`.
- The report separates passive red, level-flight red, obs-limited red, full-state chase red and oracle red policies.
- Blue launch target distribution in this run: red_2=36, red_1=26, red_0/MAV=12. Blue does sometimes target MAV, but most recorded launches target red UAVs.
- Blue is effective in weak/passive red policies and the all-attack symmetric policy; it is less decisive against full-state/true-oracle red where red gets first useful attack opportunities.

## 7. Crash and action coupling
- See `crash_preceding_window.csv` and `crash_action_coupling_summary.md`.
- This audit does not enable red GCAS or change action range; it only reports whether low-altitude crash risk persists.
- Crash_LowAlt is still present: death_reason_by_side.csv records red Crash_LowAlt rows, and crash_preceding_window.csv contains 3600 preceding-step samples. This remains a flight-envelope/action-safety risk, especially for scripted chase variants.

## 8. All-attack homogeneous interpretation
- The all-attack config is included to isolate homogeneous reward/control sanity. If it still crashes under true oracle, suspect action/PID/GCAS/flight-envelope. If only scripted chase crashes, suspect the scripted chase policy and initial geometry.
- In this run, `red_rule_vs_blue_rule_symmetric_all_attack` produced 6/6 blue_win with blue_fire=14 and blue_hit=12, so all-attack failure is not just a missing red missile-chain issue; blue pressure plus red flight/control behavior must be inspected together.

## 9. Conditional conclusion
- Current environment is not bottom-layer unreachable if full-state or true-oracle rows show missile-launch-hit reachability.
- Whether it is friendly to RL depends on obs-limited chase, launch blocked reasons and crash coupling.
- If obs-limited fails while full-state succeeds, prioritize observation/policy representation.
- If obs-limited succeeds while RL fails, prioritize reward/algorithm.
- If true oracle still crashes frequently, prioritize action/PID/GCAS/flight-envelope diagnostics.
- If blue-zero succeeds while blue-rule fails, prioritize blue-rule pressure diagnostics.

## 10. Forbidden overstatements
- Do not claim the environment has no problem.
- Do not claim the algorithm is definitely wrong.
- Do not claim reward or blue rule is definitely wrong without the conditional evidence above.
