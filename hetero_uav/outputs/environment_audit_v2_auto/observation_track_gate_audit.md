# Observation / Track Gate Audit
- `mav_shared_geo` exposes `enemy_observed_mask` and `enemy_track_source`.
- V2 adds obs-limited scripted policies that read only actor observation fields.
- If full-state chase succeeds while obs-limited chase fails, observation or representation is more suspicious than missile dynamics.