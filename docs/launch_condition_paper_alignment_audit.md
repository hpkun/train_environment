# Launch-condition paper alignment audit

## 1. Scope

This pass is audit-only. It reviews whether the current
`my_uav_env/env.py::_check_missile_launch()` gate matches the paper launch
condition description.

No code was changed in this pass. No training, evaluation, env reset, or JSBSim
execution was run.

Reviewed sources:

- Paper extracted text in `_paper_text_tmp.txt`, especially Section 2.1.3 and
  Table 4.
- `my_uav_env/env.py` launch constants and `_check_missile_launch()`.
- Current launch-quality probe summary supplied by the user.

## 2. Paper launch-condition text

The extracted paper text states:

- The missile is a generic infrared short-range AAM.
- Fire-control position data comes from the host aircraft's electro-optical
  sensors.
- Launch condition: electro-optical sensors detect a target and the target
  remains continuously detected within the sensor detection cone for over
  0.25 s.
- After launch, the missile is autonomously guided by its infrared seeker.
- Launch interval: each aircraft observes a 0.5 s interval after firing.
- Missiles are launched when launch conditions are met and no recent missile
  launch events have targeted the same target.
- To improve hit probability, missiles are launched only after the enemy
  aircraft crosses the 3-9 line.
- The 3-9 line is the rear hemisphere of the target aircraft, geometrically the
  180 degree fan-shaped area from target 3 o'clock to 9 o'clock.
- Table 4 gives photoelectric sensor detection range as 10 km.

Important absence in extracted text:

- No explicit `closing_speed > 0` launch gate was found.
- No explicit "target not moving away" launch gate was found.
- No no-escape-zone or missile energy launch gate was found.
- No exact electro-optical cone angle was found.
- No explicit minimum range was found.
- No stricter rear-aspect threshold beyond crossing the 3-9 line was found.

These absences are based on the extracted text only. Exact cone angle and any
appendix/code-only fire-control logic would need visual PDF or released-code
verification.

## 3. Current implementation

Current constants in `my_uav_env/env.py`:

| Item | Current value |
|---|---:|
| `MISSILE_LAUNCH_AO_THRESH` | 45 deg |
| `MISSILE_LAUNCH_RANGE_THRESH` | 10,000 m |
| `MISSILE_LAUNCH_MIN_RANGE` | 500 m |
| `MISSILE_LAUNCH_TA_THRESH` | 90 deg |
| `missile_lock_delay_frames` | `round(0.25 * sim_freq)` |
| `missile_cooldown_frames` | `round(0.5 * sim_freq)` |

Current `_check_missile_launch()` behavior:

- Iterates over all alive agents.
- Skips dead shooters.
- Decrements per-shooter cooldown each physics frame.
- Scans alive enemies.
- Skips targets already in `_engaged_targets`.
- Computes `AO, TA, R = get2d_AO_TA_R(...)`.
- Requires:
  - `R > 500 m`
  - `R < 10 km`
  - `AO < 45 deg`
  - `TA > 90 deg`
- Among eligible targets, picks the closest unengaged target.
- Requires continuous lock for `0.25 s`.
- Requires cooldown to be zero.
- Requires shooter not to be on kill cooldown.
- Launches, hot-adds the target to `_engaged_targets`, then resets lock.

The recently added launch-quality instrumentation records closing speed and
other quality values after the launch decision is already satisfied. It does
not gate launch.

## 4. Requirement-by-requirement alignment

| Requirement / possible requirement | Paper evidence | Current implementation | Status | Notes |
|---|---|---|---|---|
| Electro-optical sensor detects target in cone | Paper says target must be continuously detected within sensor detection cone | `AO < 45 deg` | PARTIAL / NEEDS PAPER TEXT VERIFICATION | Paper extracted text does not state the cone half-angle. Current 45 deg is a project assumption for the cone. |
| Photoelectric sensor max range | Table 4: 10 km | `R < 10,000 m` | MATCH | Uses 2D `R` from `get2d_AO_TA_R`; paper launch range is sensor range, not explicitly 2D vs 3D in extracted text. |
| Minimum launch range | Not found in extracted text | `R > 500 m` | ENGINEERING ADDITION | This prevents point-blank/self-hit behavior. It is not the cause of low Red hit rate at normal launch ranges, but is not paper-text-backed. |
| 3-9 line / rear hemisphere | Paper requires crossing target 3-9 line, rear 180 deg fan | `TA > 90 deg` | MATCH | This is the natural rear-hemisphere interpretation. |
| Stricter rear-aspect shot, e.g. `TA > 120/135/150 deg` | Not found in extracted text | Not required | NOT REQUIRED BY EXTRACTED PAPER | Do not tighten TA based only on hit-rate diagnostics. |
| Lock delay | Paper: target remains detected for over 0.25 s | `lock_timer >= round(0.25 * sim_freq)` | MATCH | Counted in physics frames. |
| Launch cooldown | Paper: 0.5 s launch interval | `missile_cooldown_frames = round(0.5 * sim_freq)` | MATCH | Counted in physics frames. |
| No recent missile launch events targeting same target | Paper explicitly states this | `_engaged_targets` skips in-flight/freshly assigned targets and hot-updates after launch | MATCH / ENGINEERING-STRENGTHENED | Current implementation is at least as strict. It also prevents same-frame double-launch. |
| Closest target | Paper does not state closest target as a launch condition | Chooses closest eligible unengaged target | ENGINEERING TARGET SELECTION | This affects which target is fired at, but not a paper-required quality gate. |
| Closing speed > 0 | Not found in extracted launch conditions | Not required | NOT REQUIRED BY EXTRACTED PAPER | Current launch-quality probe shows this is highly diagnostic, but not a paper launch gate. |
| Target not moving away | Not found in extracted launch conditions | Not required | NOT REQUIRED BY EXTRACTED PAPER | Same as closing speed. |
| No-escape-zone / missile energy condition | Not found in extracted launch conditions | Not required | NOT REQUIRED BY EXTRACTED PAPER | Paper uses hit probability and missile dynamics after launch rather than an explicit pre-launch NEZ gate in extracted text. |
| Hit probability directional factor | Paper defines `P_hit = 0.05 + 0.95 * max(0, Vm dot LOS / (|Vm||LOS|))` | Implemented in missile simulator, not launch gate | MATCH CONCEPTUALLY | This belongs to hit logic, not launch authorization. |

## 5. Launch-quality probe interpretation

Given probe results:

| Metric | Red | Blue |
|---|---:|---:|
| Launches | 391 | 54 |
| Hits | 7 | 50 |
| Hit rate | 1.79% | 92.59% |
| Closing mean | -122.5 m/s | +175.6 m/s |
| Miss closing mean | -125.3 m/s | not provided |
| Hit closing mean | +29.2 m/s | not provided |

Interpretation:

- Red is satisfying the current launch gates often enough to fire, but many Red
  shots are low-quality from an engagement-geometry standpoint.
- The strongest diagnostic is closing speed: Red average launch closing speed is
  negative, while Red hits have positive mean closing speed and Blue launches
  have strongly positive mean closing speed.
- This does not prove a launch-condition mismatch. The paper explicitly uses
  rear-aspect launch and post-launch hit probability; it does not, in the
  extracted text, require pre-launch positive closing speed.
- A negative-closing rear-aspect shot can occur when the shooter is behind the
  target but falling away or unable to close. That is poor learned positioning /
  energy management, not necessarily an invalid launch under the paper text.

## 6. Decision

Do **not** change missile launch conditions based on this audit.

Reason:

- The extracted paper launch-condition requirements are already mostly captured:
  10 km photoelectric range, 0.25 s continuous detection, 0.5 s launch interval,
  same-target deconfliction, and rear hemisphere / 3-9 line.
- The paper text found in this pass does not require `closing_speed > 0`,
  target-not-moving-away, no-escape-zone, a smaller AO threshold, or a larger TA
  threshold as pre-launch gates.
- Red's poor hit rate is therefore better classified as reward/learning quality
  or policy geometry quality, not as a confirmed paper mismatch in launch
  authorization.

Paper mismatch candidates to keep open:

1. **AO cone angle**: current `AO < 45 deg` implements the unspecified sensor
   cone. The extracted paper text does not give the exact cone half-angle.
2. **Minimum range**: current `R > 500 m` is an engineering safety guard not
   found in the extracted paper text.
3. **2D vs 3D launch range**: current `R` comes from `get2d_AO_TA_R()`. The
   paper's photoelectric range is stated as 10 km, but the extracted launch text
   does not clearly specify whether this gate is horizontal or 3D range.

Recommended next action:

- Keep launch conditions unchanged.
- Use the launch-quality diagnostics to analyze Red launch geometry by closing
  speed, AO, TA, range, and action saturation.
- If intervention is needed, prefer reward/learning diagnostics or policy
  quality analysis before any launch-gate ablation.
