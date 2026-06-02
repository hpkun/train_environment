# JSBSim Configs

JSBSim-specific BRMA and heterogeneous scenario YAML files.

Formal heterogeneous composition configs:

- `hetero_train_2v2_mav_attack.yaml`
- `hetero_test_3v3_mav_2attack.yaml`
- `hetero_test_3v3_mav_attack_scout.yaml`
- `hetero_test_3v3_mav_attack_interceptor.yaml`

`hetero_2v2_mav_attack.yaml` is retained as a debug alias for the 2v2 MAV +
attack-UAV composition.

Use `scripts/diagnose_hetero_compositions.py` and
`tests/test_jsbsim_hetero_compositions.py` to verify these configs before
starting MAPPO environment adaptation.
