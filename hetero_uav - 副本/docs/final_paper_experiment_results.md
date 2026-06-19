# Final Paper Experiment Results

## 1. Experiment Goal

The final experiment target is heterogeneous MAV/UAV cooperative air combat
with fixed-capacity zero-shot transfer from 3v2 training to 5v4 evaluation.

The concrete protocol is:

- train on 3v2: red `1 MAV + 2 UAV` versus blue `2 UAV`;
- evaluate on seen 3v2;
- evaluate zero-shot on 5v4: red `1 MAV + 4 UAV` versus blue `4 UAV`;
- do not fine-tune on 5v4.

This is not a claim of arbitrary-scale generalization. It is a fixed-capacity
3v2-to-5v4 transfer setting using padded observation slots and masks.

## 2. Method Mainline

The final mainline contains:

- BRMA-inspired V2 unified observation;
- padding and masks for fixed-capacity `max_red=5`, `max_blue=4`;
- independent MAV actor for `red_0`;
- shared UAV actor reused by all red UAV slots;
- wrapped-heading oracle imitation for circular heading targets;
- UAV imitation anchor during HAPPO fine-tuning;
- easy -> medium -> normal geometry curriculum;
- seen 3v2 evaluation plus 5v4 zero-shot evaluation.

The actor observation dimension is `96`; the critic state dimension is `480`.

## 3. Key Fixes

Several fixes were necessary before the final result became usable:

- The F-22 MAV branch was visually closer to the intended MAV but unstable
  under the current high-level `[pitch, heading, speed]` action plus PID path.
- F-16 MAV surrogate dynamics were used to make the method trainable and to
  isolate heterogeneous policy/observation questions from aircraft instability.
- Heading is a circular variable. Oracle imitation was fixed by replacing
  plain heading MSE with wrapped heading loss.
- ACMI visualization labels `red_0` as `MAV/F22 visual`, while the current
  dynamics remain F-16 surrogate for trainability.
- Missile objects are exported from real in-environment missile positions, so
  Tacview/ACMI missile tracks are visible.

No final result depends on modifying reward, missile dynamics, PID, aircraft
XML, action space, or observation dimensions during the final packaging step.

## 4. Main Results Table

| experiment | scenario | checkpoint | red_win_rate | red_elimination_win_rate | mav_survival_rate | red_missiles_fired_mean | red_missile_hits_mean | blue_dead_mean | blue_win_rate | conclusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Shared MLP baseline | 3v2 / 5v4 | best | low / timeout-draw dominated | 0.00 | 0.00 | not reliable | not reliable | 0.00 | low | weak baseline only |
| HAPPO reference survival baseline | F-16 MAV surrogate 1M, 3v2 | latest train | 1.00 train rollout | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 train rollout | survival without combat |
| Oracle-pretrain normal direct attempt | normal 3v2 | latest fast | timeout alive advantage | 0.00 | 1.00 | 0.05 | 0.00 | 0.00 | 0.00 | direct jump to normal failed |
| Easy combat 50k | easy 3v2 | latest, 50 episodes | 1.00 | not primary | 1.00 | 1.50 | 1.48 | 1.48 | 0.00 | first learned easy-task combat result |
| Geometry curriculum normal 3v2 | normal 3v2 | best, 50 episodes | 0.92 | 0.52 | 0.94 | 1.82 | 1.56 | 1.52 | 0.00 | final seen-task combat checkpoint |
| 5v4 zero-shot | 5v4 | best, 100 episodes | 0.93 | 0.14 | 0.99 | 2.59 | 2.38 | 2.33 | 0.00 | final zero-shot transfer result |

## 5. Final 3v2 Result

Final seen-task result uses the geometry-curriculum normal 3v2 best checkpoint,
evaluated for 50 episodes:

- `red_win_rate=0.92`;
- `red_elimination_win_rate=0.52`;
- `mav_survival_rate=0.94`;
- `red_missiles_fired_mean=1.82`;
- `red_missile_hits_mean=1.56`;
- `blue_dead_mean=1.52`;
- `blue_win_rate=0.00`.

This is the first learned normal-geometry checkpoint that combines MAV survival
with red UAV firing, hits, and blue aircraft deaths.

## 6. Final 5v4 Zero-Shot Result

Final zero-shot result uses the same best checkpoint from normal 3v2 geometry
curriculum and evaluates it on 5v4 for 100 episodes:

- `red_win_rate=0.93`;
- `blue_win_rate=0.00`;
- `draw_rate=0.07`;
- `timeout_rate=0.86`;
- `red_elimination_win_rate=0.14`;
- `red_timeout_alive_advantage_rate=0.79`;
- `mav_survival_rate=0.99`;
- `red_alive_final_mean=3.76/5`;
- `blue_alive_final_mean=1.67/4`;
- `blue_dead_mean=2.33`;
- `kill_death_ratio=1.88`;
- `red_missiles_fired_mean=2.59`;
- `red_missile_hits_mean=2.38`.

Interpretation:

The 5v4 policy transfers UAV attack behavior and preserves MAV survival in the
larger fixed-capacity scenario. Most wins are still timeout alive-advantage
wins, with some red elimination wins.

## 7. Behavior Consistency Notes

The 5v4 behavior audit checked the fixed ACMI behavior and blue targeting logic.

Key findings:

- Blue target selection includes `red_0` MAV.
- Blue largely treats all alive red slots as equivalent candidate targets.
- There is no explicit blue MAV-priority or armed-UAV-priority rule.
- In 20 audited 5v4 episodes, blue missile targets did not include MAV:
  `blue_missile_target_mav_count=0`, `blue_missile_target_uav_count=19`.
- Blue missile targets were mainly UAVs: `red_2:14`, `red_1:5`.
- MAV behavior is better described as forward-survival / loose support, not a
  strict rear-support orbit.
- Current MAV trajectory cannot be claimed as a full reproduction of the
  heterogeneous paper's rear support behavior.
- This behavior limitation does not invalidate the reported 5v4 zero-shot
  attack-transfer metrics.

## 8. What Can Be Claimed

The paper/report can safely claim:

- The project completes a heterogeneous MAV/UAV 3v2-to-5v4 fixed-capacity
  zero-shot transfer experiment.
- The V2 unified observation with padding and masks supports the same actor
  schema in 3v2 and 5v4.
- A MAV independent actor plus shared UAV actor closes the heterogeneous
  policy loop.
- UAV attack behavior transfers to 5v4 without fine-tuning.
- MAV survival remains high in the final 5v4 zero-shot evaluation.
- The geometry-curriculum checkpoint achieves both attack behavior and MAV
  survival in seen 3v2.

## 9. What Cannot Be Claimed

The paper/report must not claim:

- full BRMA-MAPPO reproduction;
- full TAM-HAPPO reproduction;
- BRMA biased random masked attention;
- GRU or attention implementation;
- exact paper action-space reproduction;
- exact MAV rear-support trajectory reproduction;
- arbitrary-scale generalization beyond the fixed-capacity 3v2-to-5v4 setup;
- that the latest checkpoint is always best, since checkpoint selection is
  required.

## 10. Final Recommendation

Stop the current experiment here and use the geometry-curriculum best
checkpoint as the final reported model.

Do not continue normal 200k for the current paper experiment. The current
result already supports the main claim: fixed-capacity 3v2-to-5v4 zero-shot
transfer with heterogeneous actors, unified observation, red UAV attack
transfer, and high MAV survival.

Future work can be written as:

- stronger MAV support trajectory constraints;
- explicit MAV safety/support reward;
- blue opponent target-priority design;
- attention/recurrent modules for a closer TAM-HAPPO-style extension.
