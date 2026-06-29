# Action Mapping Audit
- Action space is Box [-1,1]^3 for every agent.
- PID path maps act[0] to absolute target pitch +/-90 deg, act[1] to absolute target heading +/-180 deg, act[2] to velocity 102-408 m/s.
- Direct-FCS path maps act[0]/act[1] to elevator/aileron command and act[2] to throttle 0.4-0.9.
- Red and blue share Layer 3 action mapping; team asymmetries occur before action mapping: red missile evasion, blue GCAS.
- Risk: +/-90 deg target pitch is aggressive and can expose red agents without GCAS to low-altitude or over-control crashes.