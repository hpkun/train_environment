# Initial Geometry Audit

- See CSV for per-agent initial lat/lon/alt/speed/yaw and nearest enemy distances.
- Risk to inspect: red_0 MAV in MAV configs starts behind/offset, but blue target selection may still choose it if nearest/visible geometry makes it attractive.
- 3v2 is numerically asymmetric by design: red has 1 MAV + 2 attack UAV vs blue 2 attack UAV; terminal formulas using alive counts can encode team-size bias depending on reward mode.
