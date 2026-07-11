# TAM Greedy Rule Protocol

`tam_greedy_rule` is a deterministic, paper-aligned greedy-rule opponent for
heterogeneous TAM-HAPPO experiments. It is **not an exact reproduction** because
the paper does not publish the complete candidate magnitudes and score weights.

At each existing 5 Hz environment decision, every blue aircraft evaluates the
fixed maneuver set: level hold, climb, descend, normal/hard left and right
turns, accelerate, decelerate, pursue current target, and left/right break.
The minimum safety set adds explicit return-to-center and hard-deck recovery
candidates.
The selected maneuver maximizes a normalized immediate score combining target
approach (0.30), angle alignment (0.30), speed suitability (0.10), altitude
safety (0.10), boundary safety (0.10), and missile-warning evasion (0.35).

Targets come only from the blue aircraft's current visible observation. Dead
and unobserved slots are excluded. Assignment is nearest-first with deterministic
one-to-one deconfliction, a short 15-step persistence window, and deprioritization
of targets already engaged by friendly missiles. The rule does not read red
actions, rewards, hidden simulator state, or future trajectories.

The minimum safety layer rejects descent near the hard deck, favors headings
toward the battlefield center near the boundary, and keeps target speed above a
conservative floor. Environment automatic missile launch remains unchanged.

Training metadata records `tam_rule_protocol_version`,
`tam_candidate_maneuvers`, `tam_score_weights`, and
`tam_rule_claim=paper_aligned_not_exact_reproduction`. The existing
`brma_rule` behavior is unchanged and remains a separate baseline.

Unlike `brma_rule`, this protocol does not use lead-point pursuit, layered
energy logic, G compensation, or a long-lived tactical state machine. Formal
heterogeneous Pure HAPPO/TAM experiments may select `tam_greedy_rule`; BRMA
alignment and robustness evaluations retain `brma_rule`.
