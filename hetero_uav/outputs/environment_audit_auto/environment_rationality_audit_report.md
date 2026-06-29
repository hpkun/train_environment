# Environment Rationality Audit Report

## 1. Executive Summary
- This audit performed static code/config review plus no-training scripted rollouts.
- Current environment has plausible learning signal paths, but important asymmetries exist: blue GCAS, red-only missile evasion, MAV launch block, track-gate differences, and blue scripted rule strength.
- Recommendation: do not continue long training until scripted/oracle reachability and blue-rule strength are inspected in the generated CSVs.

## 2. Environment Mechanism Findings
- Action mapping: `[-1,1]^3` to pitch/heading/speed; pitch target range is aggressive at +/-90 deg.
- PID/GCAS/evasion: blue has GCAS, red has missile evasion, red lacks GCAS.
- Dynamics: main F16-dynamics/F22-visual configs do not audit true F22 dynamics.
- Initial geometry, observation, launch gate, missile and termination details are in companion audit files.

## 3. Red/Blue Symmetry Findings
- Flight action mapping is mostly symmetric at layer 3.
- Asymmetries: blue GCAS, red-only evasion, MAV role launch block, red target selection option, blue rule policy implementation.
- Some asymmetries are intentional experimental design; others can affect early learning and should be ablated diagnostically.

## 4. Blue Rule Strength Findings
- See `scripted_rollouts/first_launch_hit_summary.csv` and `blue_rule_audit.md`.
- In passive-red rollout, blue hit count total observed: 3.

## 5. Scripted/Oracle Reachability Findings
- Oracle red total fire=12, hit=12.
- Red-vs-blue-zero total fire=12.
- See scripted rollout CSVs for per-policy outcomes.

## 6. Crash and Flight-Envelope Findings
- See crash_altitude_speed_summary.csv and termination_death_crash_audit.md.
- If red low-altitude crashes dominate while blue does not, action/PID/GCAS asymmetry is a primary suspect.

## 7. Reward Interface Findings
- Reward signals depend on step-local kill/fire counts and info diagnostics. Rich logging must be enabled for causal post-hoc analysis.
- brma_paper_homogeneous_v1 should be treated only as a diagnostic homogeneous baseline.

## 8. Decision Tree Conclusion
- Oracle/chase-style red can reach at least some launch/hit events; if RL still fails, reward/algorithm or policy representation remains suspicious.
- Blue can hit passive/weak red in scripted rollouts; blue rule strength is non-trivial.

## 9. Concrete Next Steps
1. If oracle red cannot launch/hit, inspect launch/track gate and blue pressure before reward changes.
2. If blue-zero enables red launch but blue-rule suppresses it, run a blue-strength curriculum/ablation diagnostic only.
3. If scripted red works but RL fails, then inspect reward/algorithm with rich logs before long training.