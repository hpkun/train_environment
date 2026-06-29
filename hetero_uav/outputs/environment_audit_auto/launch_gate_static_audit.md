# Launch Gate Static Audit
- `_check_missile_launch()` checks alive, ammo, cooldown, lock delay, track, range, AO, TA, optional boresight, engaged target and target selection.
- BRMA-style launch gate remains unchanged by this audit.
- Red-specific restrictions include MAV role launch block and MAV/shared track dependency.
- Blue uses same geometry gate but may differ in observation fallback and scripted policy behavior.
- Red target selection can be `closest` or `mav_threat_rank`; blue rule target selection is in parent `rule_based_agent.py` via `OpponentPolicy(brma_rule)`.