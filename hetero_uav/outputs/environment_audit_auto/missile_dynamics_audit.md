# Missile Dynamics Audit
- Current MissileSimulator is scripted close-range AAM with fixed `missile_speed_mps=600`, `t_max=60`, `K=3`, hit radius `Rc=300`, arm time 0.15s.
- Legal termination reasons in current code are hit, p_hit_fail, timeout, target_dead, unknown fallback.
- Low-speed and overshoot missile terminations are not expected in the current scripted AAM path.
- Red and blue missiles share the same simulator class; asymmetry is more likely from launch geometry, track, target selection or shooter flight state.