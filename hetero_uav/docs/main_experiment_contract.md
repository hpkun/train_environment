# Main Experiment Contract

## Purpose

This document records the current paper mainline experiment contract and the
audit used to verify that the code follows it. The audit is not a tuning step,
not a training run, and not a method module.

## Mainline Protocol

- Train config: `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml`
- Eval configs:
  - `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml`
  - `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml`
- Observation: V2 `mav_shared_geo`
- Reward mode: `brma_legacy`
- Blue opponent: `greedy_fsm`
- Algorithm: shared MAPPO baseline
- Action dimension: 3

## Composition

The 3v2 protocol uses one MAV plus two red attack UAVs against two blue attack
UAVs. The 5v4 protocol uses one MAV plus four red attack UAVs against four
blue attack UAVs.

The MAV uses the A-4 model and has `num_missiles=0`. Attack UAVs use the f16
model and have `num_missiles=2`.

## Audit Scope

`scripts/audit_main_experiment_contract.py` checks runner defaults, YAML config
values, observation adapter dimensions, shared MAPPO baseline wiring, and
required summary fields.

The audit does not modify reward, termination, missile, action, evasion, PID,
aircraft XML, attention, HAPPO, GRU, or role-aware algorithm behavior.
