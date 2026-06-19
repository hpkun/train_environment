# TAM-HAPPO Paper-Grounded Specification

This document recalibrates the implementation boundary for HAPPO 3v2
reference validation against Chen et al. 2026, "A deep reinforcement learning
cooperative air combat method with temporal feature and attention enhancement
for heterogeneous flight vehicles." It is a specification document only. It
does not change reward, missile, PID, action space, aircraft XML, observation
dimension, MAPPO, HAPPO, GRU, or attention code.

## 1. What The Paper Actually Does

The paper studies cooperative air combat in heterogeneous
manned-unmanned-missile systems. The main experimental settings are:

| setting | red side | blue side | paper role intent |
|---|---|---|---|
| 2v2 homogeneous | 2 UAV | 2 UAV | homogeneous UAV combat baseline |
| 3v2 heterogeneous | 1 MAV + 2 UAV | 2 UAV | MAV supports and survives, UAVs engage |
| 5v4 heterogeneous | 1 MAV + 4 UAV | 4 UAV | larger heterogeneous transfer/generalization |

For the paper's 3v2 setting, the red team forms an inverted triangle with one
MAV at the rear and two UAVs forward. The MAV is unarmed and is responsible for
battlefield information and mission guidance while ensuring its own safety.
Each UAV carries two missiles. The blue team has two identical UAVs, each also
carrying two missiles. The typical described behavior is that the MAV circles
or stays rearward for support and safety, while the UAVs lock targets, launch
missiles, evade incoming threats, and destroy the blue UAVs.

The paper action is low-level fixed-wing control:

- action vector: `[Ct, Ca, Ce, Cr]`;
- `Ct`: throttle control, range `[0.4, 0.9]`;
- `Ca`: aileron, range `[-1, 1]`;
- `Ce`: elevator, range `[-1, 1]`;
- `Cr`: rudder, range `[-1, 1]`;
- continuous action is represented in multi-discrete form with 40 levels per
  action dimension.

The paper's TAM-HAPPO method contains all of the following core pieces:

- HAPPO for heterogeneous agents, using multi-agent advantage decomposition and
  sequential policy update;
- centralized value function;
- temporal state memory based on GRU;
- temporal replay/buffer organization for sequence consistency;
- inactive-agent masking;
- entropy regularization;
- multi-head attention integrated into the centralized value network.

The reward design is role-specific:

- MAV reward modules: safety, support, and event rewards. The paper emphasizes
  MAV survivability, battlefield awareness, support positioning, missile-warning
  risk, aspect-angle risk, death penalty, and bounded team contribution.
- UAV reward modules: height, speed, angle, distance, dodge, and event rewards.
  The paper emphasizes safe flight, kinematic advantage, favorable missile
  engagement geometry, effective firing distance, missile evasion, kills,
  losses, crashes, and boundary violations.

The paper evaluates reward curves, win/combat behavior, trajectories, attitude
curves, ablations, and larger 5v4 heterogeneous performance. It reports TAM-HAPPO
as stronger than HAPPO and MAPPO in the heterogeneous settings.

## 2. What Our Current Environment Already Matches

- It has the paper-aligned 3v2 composition: red `1 MAV + 2 UAV`, blue `2 UAV`.
- It has the paper-aligned 5v4 composition: red `1 MAV + 4 UAV`, blue `4 UAV`.
- MAV is modeled as a support/command role and is unarmed in the main configs.
- UAVs are modeled as attack aircraft with two missiles.
- Current main configs use F-22 as the MAV engineering approximation and F-16
  for UAVs.
- The observation direction includes MAV-shared geometry for heterogeneous
  cooperation.
- The environment uses JSBSim/PID and automatic missile/fire-control mechanics.
- Dead-agent active masking and team-done handling have already been repaired
  for the MAPPO baseline.

## 3. What Our Current Environment Intentionally Differs From

- The current action remains high-level `[pitch, heading, speed]` with PID,
  not the paper's `[Ct, Ca, Ce, Cr]` multi-discrete low-level control.
- Missile launch remains scripted/environment-owned, not directly controlled by
  the actor.
- Missile evasion remains scripted/environment-owned; red-only evasion is an
  experimental environment choice, not a paper-exact implementation.
- The current observation does not fully expose missile geometry needed for a
  faithful dodge reward.
- Current baseline training uses shared MLP MAPPO unless an explicit HAPPO
  reference path is selected.
- The current role rewards are engineering approximations and must not be
  described as exact paper reward reproduction.

These differences are intentional for the current stage because the immediate
goal is reference validation of a stable heterogeneous environment and method
boundary, not a complete TAM-HAPPO reproduction.

## 4. What HAPPO Reference v0 May Implement

HAPPO reference v0 may reasonably implement only a bounded subset:

- separate or role-separated MAV/UAV actors;
- centralized critic/value;
- active-agent mask;
- team-level done handling;
- HAPPO-style sequential update if implemented;
- high-level action retained;
- scripted missile retained;
- no attention in v0;
- no GRU in v0 unless explicitly implemented later;
- paper-informed 3v2 validation metrics.

This scope can validate whether a heterogeneous-policy baseline is worth
pursuing in the current environment. It is not enough to claim full TAM-HAPPO.

## 5. What Must Not Be Claimed

Do not claim:

- full TAM-HAPPO reproduction;
- paper action-space reproduction;
- attention-enhanced value network;
- temporal GRU module;
- exact paper reward reproduction;
- learned missile evasion;
- paper-identical aircraft dynamics or physical parameters.

## 6. Minimum Paper-Grounded HAPPO Implementation Boundary

Before coding or extending HAPPO, the implementation boundary should be:

1. Keep current high-level `[pitch, heading, speed]` action unless a separate
   low-level-control branch is explicitly planned.
2. Keep scripted missile launch/evasion in v0 and document that this is an
   environment engineering choice.
3. Implement only HAPPO-style sequential update in v0 if the algorithm work is
   resumed.
4. Treat no-GRU HAPPO as a no-temporal HAPPO ablation.
5. Treat no-attention HAPPO as a non-attention HAPPO ablation.
6. Use 3v2 behavior, MAV survival, UAV engagement, missile hits, blue deaths,
   non-timeout outcomes, and 5v4 transfer as validation metrics.

## 7. Checklist Before Coding HAPPO

- Confirm whether the next run is a HAPPO ablation or full TAM-HAPPO work.
- Confirm whether GRU is in scope.
- Confirm whether attention value network is in scope.
- Confirm whether low-level paper action space is in scope.
- Confirm whether missile-aware observation is in scope.
- Confirm whether role reward is a controlled ablation or baseline setting.
- Do not mix action-space, reward, missile, and algorithm changes in one run.
