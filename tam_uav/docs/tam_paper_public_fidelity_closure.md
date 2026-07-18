# TAM paper public fidelity closure

This note classifies the frozen v4 environment and the pure feedforward HAPPO
baseline against information publicly stated in the paper.

## A. MATCHED_PUBLISHED

1. The published 2v2, 3v2, and 5v4 initial states.
2. JSBSim physics at 60 Hz.
3. Policy decisions at 5 Hz.
4. Twelve physics frames per policy action.
5. Episode limit of 1000 decision steps.
6. Four-dimensional direct-FCS action control.
7. Throttle range `[0.4, 0.9]`.
8. Control-surface ranges `[-1, 1]`.
9. Forty categorical levels per action dimension.
10. Seven-dimensional self observation.
11. Five-dimensional relative observation.
12. Situation evaluation from Eq. 10-12 with the published weights.
13. Aircraft limits of 400 m/s, 750 m minimum altitude, and 9 g.
14. Missile limits and metadata: 30 g, 14 km, 25 s, 84 kg, 2.87 m, and 0.127 m.
15. Proportional navigation gains `Ky=Kz=3`.
16. Published MAV/UAV roles and missile counts.
17. Published reward formulas and weights.
18. Actor and value learning rates of `5e-4`.
19. PPO clip parameter `0.2`.
20. Gradient clipping at `10`.
21. Entropy coefficient `0.01`.
22. GAE lambda `0.95`.
23. Discount factor gamma `0.99`.
24. Feedforward actor and critic hidden widths `[256, 128]`.
25. Explicit clipped Huber value loss with delta `10`.

## B. PAPER_SILENT_ASSUMPTIONS

1. The specific F22 and F16 JSBSim model identities.
2. Detailed blue-FSM state transitions.
3. Candidate-manoeuvre prediction constants.
4. Detection distance.
5. Radar field of view.
6. Missile initial speed.
7. Missile thrust and powered duration.
8. Effective missile drag.
9. Missile hit radius.
10. Missile timeout.
11. Exact UAV height-reward definitions of `PV` and `PH`.
12. MAV distance thresholds.
13. `Cmax`.
14. Winner resolution at the episode limit.
15. PPO epoch count.
16. Minibatch size.
17. Rollout length.

## C. OUT_OF_SCOPE

1. TAM network reconstruction.
2. GRU.
3. State memory.
4. Attention.
5. Zero-shot mechanisms.
6. Table 8 generalization experiments.
7. Self-play.
8. A learned blue opponent.

`PUBLICLY_SPECIFIED_ENVIRONMENT_COMPONENTS_ALIGNED=true`

`EXACT_PRIVATE_ENVIRONMENT_REPRODUCED=false`

`PAPER_SILENT_ASSUMPTIONS_PRESENT=true`

`PURE_FEEDFORWARD_HAPPO_BASELINE=true`
