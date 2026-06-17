# Remaining Experiments Todo

This list is intentionally minimal. It is for finishing the paper experiment,
not for expanding the system.

## Priority Legend

- Must: required for the current paper/report experiment.
- Optional: useful if time permits.
- Can drop: not necessary for the current paper/report.

## Experiment Checklist

| ID | Item | Priority | Purpose | Notes |
|---|---|---|---|---|
| A | Complete `brma_recurrent_masked` no-random-mask training to 500k or 1M | Must | Final main-method candidate with safe PPO log-prob replay | Use non-time-limited terminal; preserve intermediate eval/checkpoints. |
| B | Optional biased-mask 500k diagnostic | Optional | Ablation for biased mask generator forward path | Run only if explicitly needed; it is not a full BRMA mask objective. |
| C | Final 3v2 seen eval | Must | Seen-scenario result | Use best checkpoint selection; at least 50 episodes, preferably 100. |
| D | Final 5v4 zero-shot eval | Must | Transfer result | No fine-tuning; same checkpoint as 3v2 eval. |
| E | Representative trajectory export | Must | Figure material | Select episodes with red fire/hit/blue death where available. |
| F | Training curve plot | Must | Learning process figure | Use reward, win/draw/timeout, mask stats, missile/hit stats. |
| G | Win-rate curve plot | Must | Seen/zero-shot trend | Plot 3v2 and 5v4 eval over training steps. |
| H | Missile/hit/blue-dead statistics | Must | Combat-behavior evidence | Include red missiles fired, red hits, blue dead. |
| I | More short architecture probes | Can drop | Engineering validation | Current tests and smokes are enough for paper material preparation. |
| J | Full strict HAPPO sequential correction | Can drop | Algorithmic completeness | Too large for current stage; document as limitation/future work. |
| K | Full BRMA biased mask objective | Can drop | Algorithmic completeness | Future work; do not replace it with unsafe random mask. |

## Recommended Immediate Sequence

1. Finish one clean no-random-mask run with intermediate checkpoint/eval logging.
2. Select best checkpoint by 3v2 seen eval and verify 5v4 zero-shot eval.
3. Export representative ACMI and curves for paper figures.
4. Only then decide whether biased-mask 500k is worth running as a diagnostic.

## Random Mask Status

`--brma-random-scale-mask` is disabled in the main training entrypoints. The
current implementation re-samples masks during rollout and PPO update, so
old/new log probabilities are not guaranteed to use the same entity mask.
Recovery requires either rollout mask replay or a full BRMA biased-mask
objective.
