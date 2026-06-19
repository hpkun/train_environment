# Current Experiment Conclusion and Next Plan

## 1. Current Goal

The current goal is to finish a defensible heterogeneous MAV/UAV cooperative air-combat experiment mainline. This is not a full TAM-HAPPO reproduction, not low-level flight-control reinforcement learning, and not an open-ended engineering effort to keep auditing every environment component.

The experiment should answer a narrower question: under the current heterogeneous 3v2 train and 5v4 transfer setting, can a learned red MAV/UAV team acquire basic engagement behavior, fire missiles, score hits, and preserve the MAV well enough to support zero-shot composition experiments?

## 2. Completed Experiments

### Shared MLP MAPPO 1M

The shared MLP MAPPO 1M run is the current weak baseline. Its best 100-episode checkpoint mostly survived until timeout but did not produce reliable combat:

| eval config | red_win | blue_win | draw | timeout | MAV survival | blue elimination |
|---|---:|---:|---:|---:|---:|---:|
| 3v2 | 0.05 | 0.00 | 0.95 | 1.00 | 0.00 | 0.00 |
| 5v4 | 0.15 | 0.12 | 0.73 | 1.00 | 0.00 | 0.00 |

Interpretation: shared MLP MAPPO is useful only as a lower-bound baseline. It does not establish heterogeneous combat competence.

### HAPPO Reference v0 With F-22 MAV

The F-22 MAV branch exposed a control and stability problem under the current high-level action interface. Training logs could show apparent survival or timeout behavior, but independent evaluation and ACMI inspection showed that this was not a robust combat policy. Earlier analysis concluded that F-22 should not remain the immediate blocker for the paper experiment.

Interpretation: F-22 is closer to the intended MAV appearance, but in the current environment it is not the right short-term vehicle for validating learning.

### HAPPO Reference v0 With F-16 MAV Surrogate 200k

The F-16 MAV surrogate fixed the immediate survival stability issue enough to validate the algorithm/environment path. The 200k surrogate evaluation produced real attack signals in the latest checkpoint:

| checkpoint | red_win | blue_win | timeout | MAV survival | red fire | red hit | blue death |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | 0.96 | 0.00 | 1.00 | 1.00 | 0.02 | 0.02 | 0.02 |
| latest | 0.80 | 0.10 | 0.56 | 0.86 | 1.48 | 1.20 | 1.20 |

Interpretation: this is the strongest evidence so far that the simplified setup can support learned red attacks, but it is not yet a stable final result.

### HAPPO Reference v0 With F-16 MAV Surrogate 1M

The 1M F-16 surrogate training run completed without NaN and showed stable survival in the training log, but final checkpoint evaluation did not confirm a strong combat policy:

- Latest train row: return about +12.99, red_win 1.00, MAV survival 1.00, but red missiles fired 0 and missile hits 0.
- Best 100-episode 3v2 eval: red_win 0.03, blue_win 0.57, draw 0.40, MAV survival 0.00, red hits 0.07.
- Best 100-episode 5v4 eval: red_win 0.33, blue_win 0.02, draw 0.65, MAV survival 0.00, red hits 1.24.
- Latest checkpoint eval collapsed to blue elimination wins in both 3v2 and 5v4.

Interpretation: F-16 MAV surrogate solved the immediate aircraft-survival problem, but HAPPO reference v0 is still mainly a survival/timeout baseline, not a reliable combat method.

### Red Attack Pipeline Audit

The red attack pipeline audit verified that the environment can support red missile attacks:

- Red attack UAVs have missiles.
- Red and blue fire-control are both active.
- Red observations include enemy tracks.
- Fire-control does not require policy-selected targets; it can select valid targets from range, angle, lock, cooldown, alive, and ammunition constraints.

Interpretation: the missing learned behavior is not caused by a broken red fire-control chain.

### Red Direct Chase Oracle Sanity Check

The direct chase oracle proved that a simple scripted red policy can close the attack geometry, fire, hit, and kill:

| scenario | red fire | red hit | blue death | red win |
|---|---:|---:|---:|---:|
| direct chase vs blue zero | 2.00 | 2.00 | 2.00 | 1.00 |
| direct chase vs blue BRMA rule | 2.25 | 2.00 | 2.00 | 1.00 |

Interpretation: the environment can produce red kills when the policy enters the launch envelope. The learned HAPPO policy mainly fails before launch: it does not close correctly, does not satisfy launch angle, and does not remain in the launch envelope.

## 3. Core Conclusions

- Shared MLP MAPPO should only be treated as a weak baseline.
- F-22 MAV is not a good immediate experimental vehicle under the current high-level control interface.
- F-16 MAV surrogate improves survival stability, but HAPPO reference v0 remains mostly a survival baseline.
- The red fire chain is working.
- The direct chase oracle can fire, hit, and destroy blue aircraft.
- The learned HAPPO policy's main failure is tactical engagement: it does not reliably approach, align, and satisfy launch range/angle constraints.

## 4. Stop Doing

The following work should remain paused:

- Do not continue long training runs just to see whether latest improves.
- Do not continue repairing F-22 as the main blocker.
- Do not add more audit scripts unless they directly answer the next experimental decision.
- Do not implement full TAM-HAPPO now.
- Do not blindly tune reward values without first making the policy learn basic engagement.

## 5. Two Possible Next Directions

### A. Minimal Algorithm Direction

Use the direct chase oracle as a simple prior for learning. This can be done through imitation pretraining, action guidance, or a short behavior-cloning warm start so that the learned red UAVs first acquire approach-and-align behavior.

Rationale: the oracle already demonstrates the missing behavior. The learning problem should be shaped around acquiring that behavior before adding temporal attention or more complex heterogeneous modules.

Risk: if implemented too heavily, the result may become a scripted-policy demonstration rather than an RL result. The prior must be minimal and clearly reported.

### B. Minimal Environment-Task Direction

Construct an easy combat task by reducing initial distance and adjusting initial headings so that the policy can encounter the launch envelope more often.

Rationale: current trajectories often fail before any valid launch opportunity. A simpler task can test whether the existing learner can discover firing and hit behavior when the geometry is not too hard.

Risk: this must be presented as a curriculum or diagnostic task, not as the final zero-shot transfer scenario.

## 6. Recommended Next Step

Prioritize the easy combat task or direct-chase oracle imitation route. The immediate next milestone should be: a learned policy fires and hits in 3v2 under controlled, easier initial geometry. Do not spend the next cycle on large-scale engineering audits or full TAM-HAPPO implementation.
