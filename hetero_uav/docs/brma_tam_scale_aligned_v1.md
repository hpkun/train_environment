# brma_tam_scale_aligned_v1

## Scope

`brma_tam_scale_aligned_v1` is reward contract revision 3. It is a
paper-semantics/environment-aligned reward, not a verbatim reproduction of
BRMA-MAPPO or TAM-HAPPO. It coexists with and does not change
`brma_tam_scripted_composite_v1`.

The old composite combines TAM dense terms weighted by 10/15/10 with BRMA
flight terms whose coefficients are around 0.01. At the initial 3V2 geometry,
the old distance term is `-10` per attack UAV per decision, producing an
effective team reward near `-6.54`; over 1000 decisions this naturally reaches
roughly `-6500`. This can dominate kill events and rewards early death by
ending the continuing penalty.

## UAV reward

```text
R_UAV = R_flight + R_progress + R_event,UAV + R_terminal
R_flight = r_pitch + r_roll + r_alt + r_bound + r_vel
```

The flight components are the already weighted BRMA base components. BRMA
advantage, base terminal/death, and TAM dodge remain log-only.

For the closest alive blue target (3D range, blue-id tie break):

```text
Phi_D = exp(-max(d_km - 5, 0) / 10)
Phi_A = clip((1 - (ATA + AA)/pi + 1) / 2, 0, 1)
Phi_V = clip((R_V + 1) / 2, 0, 1)
R_progress_raw = 5 Delta_D + 2 Delta_A + Delta_V
R_progress = clip(R_progress_raw, -0.5, 0.5)
```

This replaces a persistent far-range penalty with a potential difference. A
stationary trajectory has zero cumulative progress; approach and retreat are
approximately antisymmetric. The progress baseline is isolated by episode and
agent. It resets without reward on episode start, target switch/death, no
target, invalid geometry/speed, and dead-before state. It updates once per
policy decision, not per JSBSim physics frame.

UAV events are `+10` per kill, `-10` once on death/crash, and `-5` on first
horizontal out-of-zone event. Death takes precedence over same-step OOB.

## MAV reward

```text
R_MAV = R_flight + 0.02 R_role_raw + R_event,MAV + R_terminal
R_role_raw = 0.5 R_dist + 0.3 R_threat
           + 0.2 mean(R_aspect) + 0.6 R_pos + 0.4 mean(R_aware)
```

`R_dist`, `R_threat`, and `R_pos` preserve the audited 3D formulas. Aspect and
awareness divide by the number of alive blue aircraft. Awareness uses alive
blue count, not visible count, so partial and complete coverage differ. This
keeps the active role scale comparable between 3V2 and 5V4 while retaining raw
sum and per-blue mean diagnostics.

MAV death is `-20` once. Team kill credit is `10 / initial_blue_count` per new
attack-UAV kill, capped at 10, so complete enemy elimination has the same
credit in 3V2 and 5V4. A dead-before MAV receives no later credit.

## Team terminal

```text
L_blue = dead_blue / initial_blue
L_red = (initial_attack_uav * mav_dead + dead_attack_uav)
        / (2 * initial_attack_uav)
R_terminal = clip(30 (L_blue - L_red), -30, 30)
```

The same terminal value is assigned once to every red agent alive before the
terminal step. The trainer's alive-before mean therefore preserves the team
terminal scale. In 3V2: no-loss win `+30`, one-UAV-loss win `+22.5`, MAV-loss
win `+15`, no-loss timeout `0`, MAV-loss/no-kill `-15`, and red elimination
without blue loss `-30`. The theoretical range remains `[-30, 30]` in 5V4.

## Configuration provenance

All coefficients and thresholds are explicit in the 3V2 and 5V4 YAML blocks.
The progress construction is an implementation-level scale alignment; BRMA
flight and TAM role geometry supply its semantics. No launch reward, heading
reward, imitation, curriculum, reward clipping wrapper, or normalization
wrapper is used. Missile dynamics/guidance/hit model, launch gate, scripted
launch/evasion, blue rule, PID, JSBSim/XML, action/observation spaces, initial
geometry, termination, GAE, HAPPO sequential correction, and policy network
are unchanged.
