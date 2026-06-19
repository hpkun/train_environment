# Red-Only Missile Evasion

## Current Design

- Red agents use scripted missile evasion.
- Blue agents do not use scripted missile evasion.
- Blue agents still keep the existing GCAS safety net and normal action path.

## Motivation

The experiment models missile-warning / emergency evasion as a red MAV/UAV
formation information advantage. Blue remains a rule-based opponent and does not
receive the same scripted missile evasion layer.

## Paper Relation

The heterogeneous paper supports incoming missile information as an observable
object. It does not require that blue must be unable to sense missiles. This
project treats red-only scripted evasion as an explicit experimental setting to
emphasize the red heterogeneous formation's sensing advantage.

## Limitations

- This is still scripted evasion, not learned evasion.
- It does not use full missile entity observation.
- It does not model time-to-go, missile energy, or a dedicated 3D dodge reward.
- Later work can add learned or hybrid evasion after missile-aware observation
  is available.
