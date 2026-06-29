# Observation / Track Gate Audit
- `brma_sensor` provides classic entity states; `mav_shared_geo` adds geo states, observed masks and `enemy_track_source`.
- Red MAV role is blocked from launch by `_has_launch_track()` returning `role_blocked_mav`.
- Red UAV can use direct track or `mav_shared` track if obs cache is populated.
- Blue launch track checks `enemy_track_source` / `enemy_observed_mask`, then falls back to legacy visible enemy states.
- Potential asymmetry: blue fallback/direct track can be broader than red MAV-shared track path; this must be interpreted with rollout blocked-reason tables.