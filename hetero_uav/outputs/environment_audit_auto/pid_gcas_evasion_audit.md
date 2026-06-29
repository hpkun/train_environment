# PID / GCAS / Evasion Audit
- Blue has GCAS safety net when `enable_gcas_for_blue=true`; red does not.
- Red has scripted missile evasion; blue does not.
- These asymmetries are explicit in `_parse_actions()` and can affect crash rates and survivability.
- Current main F16-dynamics/F22-visual configs use F16 dynamics for MAV and F22 only as ACMI visual label.
- F16-dynamics MAV surrogate means this audit cannot be interpreted as true F22 flight-dynamics rationality.