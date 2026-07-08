# Pure-HAPPO Degradation Audit

This audit package is a read-only diagnostic entry point for staged degradation
in `pure_happo` runs.  It does not modify reward, environment dynamics, policy
architecture, missile logic, or training code.

## Scope

The audit separates four possible explanations:

- Reward semantics: terminal or dense reward favors survival/count advantage
  instead of effective attack.
- Policy parameters: entropy, log standard deviation, KL, action saturation, or
  critic loss show PPO instability.
- Environment pressure: blue pressure or missile events dominate before red
  can enter effective attack geometry.
- Implementation accounting: PPO log-prob replay, tanh action accounting,
  inactive masks, zero-learning-rate updates, or grouped GAE violate invariants.

## Scripts

Run all audits on the default fixed-route pure-HAPPO output:

```bash
bash scripts/run_pure_happo_degradation_audit.sh
```

Override the run directory or device:

```bash
OUT=outputs/my_pure_happo_run DEVICE=cpu bash scripts/run_pure_happo_degradation_audit.sh
```

Individual scripts:

```bash
python scripts/audit_training_degradation.py --output-dir outputs/pure_happo_fixed_route_blue_500k_probe
python scripts/audit_terminal_reward_semantics.py --output-dir outputs/pure_happo_fixed_route_blue_500k_probe
python scripts/audit_pure_happo_update_invariants.py --device cpu
```

## Outputs

For a run directory `OUT`, the offline log audits write:

- `OUT/degradation_audit/degradation_summary.json`
- `OUT/degradation_audit/degradation_summary.csv`
- `OUT/degradation_audit/degradation_summary.md`
- `OUT/degradation_audit/terminal_reward_semantics.json`
- `OUT/degradation_audit/terminal_reward_semantics.csv`
- `OUT/degradation_audit/terminal_reward_semantics.md`

The synthetic PPO accounting audit writes:

- `outputs/audits/pure_happo_update_invariants/summary.json`

## Interpretation

`train_log.csv` is treated as recent-window training telemetry, not formal
evaluation.  A flagged stage is a lead for inspection, not a final conclusion.

- `suspicious_survival_win`: red wins appear tied to timeout/survival rather
  than red missile hits or blue deaths.
- `attack_effective`: red fire/hit/blue-dead indicators suggest actual attack
  progress.
- `regression_from_previous_stage`: return, red win, or red firing dropped
  substantially from the previous stage.
- PPO anomaly flags indicate possible parameter-update instability.

If `audit_pure_happo_update_invariants.py` fails, inspect implementation
accounting before interpreting reward or environment difficulty.  If it passes
while terminal and degradation audits show survival-only wins, prioritize reward
semantics and environment pressure analysis.
