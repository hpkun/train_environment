# Paper-Readiness Gap Analysis

This note is a readiness check, not a final result claim. The current evidence shows a working heterogeneous MAV/UAV experiment framework and a 3v2-to-5v4 transfer phenomenon. It does not yet prove algorithm superiority.

## A. Already Supported

| Item | Current evidence | Status |
|---|---|---|
| Fixed-capacity 3v2-to-5v4 zero-shot phenomenon | The geometry-curriculum normal checkpoint evaluates on both seen 3v2 and larger 5v4 without fine-tuning. | Supported as a single-run phenomenon |
| Unified observation and shared UAV actor feasibility | V2 actor observation keeps a fixed 96-dim schema and critic state keeps a fixed 480-dim schema across 3v2 and 5v4. | Supported |
| Wrapped-heading imitation utility | The original oracle action-match MSE was about 0.075882 with closed-loop no-fire behavior; wrapped heading reduced MSE to about 0.010 and enabled closed-loop launches. | Supported as an ablation signal |
| Geometry curriculum utility | Direct normal geometry oracle anchor produced near-zero fire/hit; curriculum normal best produced red launches, hits, and blue deaths. | Supported as an ablation signal |

## B. Not Fully Supported

| Missing or weak evidence | Why it matters |
|---|---|
| Algorithm superiority over strong baselines | Current evidence is mostly single-run and does not include enough statistically comparable baselines. |
| Robust multi-seed stability | A paper-level claim needs repeated seeds or confidence intervals. |
| Complete TAM-HAPPO or BRMA-MAPPO reproduction | The current method does not implement GRU, attention value network, or BRMA biased random masked attention. |
| Strict MAV support behavior | 5v4 ACMI behavior is closer to forward-survival or loose support than a verified rear support trajectory. |
| Environment and opponent realism | Blue policy remains rule-based; no claim should be made about robustness against a learned opponent. |
| Zero-shot quality equal to in-domain training | 5v4 zero-shot win rate is retained, but elimination retention and per-enemy kill efficiency drop. |

## C. Required for Paper-Level Claims

| Evidence category | Minimum required |
|---|---|
| Baseline table | Compare shared MAPPO, HAPPO reference v0, oracle-anchor variants, and geometry curriculum under the same reporting schema. |
| Ablation table | Include wrapped-heading correction, oracle anchor, and geometry curriculum. |
| Transfer-quality table | Report win retention, elimination retention, fire/hit retention, normalized blue-death retention, MAV survival delta, and timeout-dependency delta. |
| 5v4 adaptation upper-bound | Fine-tune from the 3v2 checkpoint on 5v4 to estimate how much zero-shot performance leaves on the table. |
| Multi-seed repeated training/eval | Needed before claiming robust algorithmic superiority. |
| Limitation section | State that this is fixed-capacity 3v2-to-5v4 transfer, not arbitrary-scale generalization or full TAM-HAPPO reproduction. |

## Recommended Interpretation

The current result is enough to motivate a thesis/report section about a working experimental framework, transfer behavior, and key engineering ablations. It is not enough to claim that the algorithm is generally better than MAPPO/HAPPO baselines. The next evidence step is not another large training run; it is a compact evidence matrix plus a 5v4 fine-tune upper-bound and clearer transfer-quality metrics.
