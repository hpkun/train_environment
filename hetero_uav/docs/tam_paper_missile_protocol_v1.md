# TAM Paper Missile Protocol V1

The isolated `tam_paper_protocol_v1` YAML files preserve the frozen v2 reward,
aircraft, geometry, observation, action, termination, and missile inventory.
They change only the published TAM-HAPPO missile protocol values:

| Paper parameter | Protocol value | Current implementation |
|---|---:|---|
| Attack range | 14,000 m | Active launch-gate configuration |
| Attack interval | 25 s | Active per-shooter launch cooldown |
| Maximum overload | 30 g | Active PN acceleration clamp |
| Navigation gain | 3.0 | Active proportional-navigation gain |
| Missile mass | 84 kg | Metadata only |
| Missile length | 2.87 m | Metadata only |
| Missile diameter | 0.127 m | Metadata only |

Both teams use the same environment `MissileSimulator` and the same guidance
configuration. Mass, length, and diameter are not consumed by the current
constant-speed scripted kinematics. They are retained as protocol metadata and
must not be interpreted as an aerodynamic or propulsion model.

AO, TA, minimum launch range, lock delay, deconfliction, missile speed, hit
radius, lifetime, and hit-probability details are not supplied by the cited
table. The protocol therefore retains the current project values for these
items as project assumptions. The resulting model is a **paper-aligned
kinematic approximation, not exact missile reproduction**.

The original v2 YAML files remain unchanged and identify the completed 100K
pilot. The `_tam_paper_protocol_v1.yaml` suffix identifies all new formal
protocol runs.
