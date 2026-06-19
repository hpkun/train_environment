# Paper / Parent / hetero_uav Environment Alignment Audit

## 1. TAM-HAPPO Environment Facts

Based on the TAM-HAPPO paper (Tan et al. 2026) and project documentation.

| Aspect | Paper States | Evidence |
|---|---|---|
| Physical simulation | JSBSim 6-DOF flight dynamics | paper §3.1 |
| Simulation frequency | Not explicitly stated; typical JSBSim default 60 Hz | inferred |
| Decision frequency | Not explicitly stated | inferred from agent_interaction_steps |
| Action space | 3-dim continuous: `[pitch, heading, speed]` | paper §2.4 |
| Action mapping | target_pitch / target_heading / target_velocity (ABSOLUTE targets) | env.py:127-129 |
| Aircraft type for MAV | **Not specified** — paper uses "MAV/UAV heterogeneous" | paper §1, §3 |
| Aircraft type for UAV | **Not specified** | — |
| MAV missiles | MAV does not carry missiles (surveillance role) | paper_experiment_setup.md |
| UAV missiles | UAV carries missiles (attack role) | paper_experiment_setup.md |
| MAV role | Information sharing / survival / command | paper_method_section_draft.md §1 |
| 3v2 scenario | 1 MAV + 2 UAV vs 2 blue UAV | paper_experiment_setup.md |
| 5v4 scenario | 1 MAV + 4 UAV vs 4 blue UAV | paper_experiment_setup.md |
| Initial altitude | 6000m (3v2), 6700m MAV in 5v4 | config files |
| Initial speed | 250 m/s | config files |

**The paper does NOT specify F22 as the MAV aircraft model.** F22 was chosen as an
engineering implementation choice in hetero_uav. The project's own documentation
(`paper_experiment_setup.md`) states: "F-16 MAV surrogate is an engineering choice
for controllability under the current high-level action + PID interface."

## 2. BRMA-MAPPO Environment Facts

| Aspect | Paper / Code States |
|---|---|
| Aircraft platform | JSBSim-backed 6-DOF |
| Control input | High-level target commands → PID → JSBSim surfaces |
| Engine startup | `propulsion.get_engine(j).init_running()` + `propulsion.get_steady_state()` |
| JSBSim model reuse | `reset_to_initial_conditions(0)` reused across episode resets |
| Default aircraft | Parent project `train_ppo.py` uses **F-16** (default Config has no aircraft selection) |
| Engine model specifics | No starter/N1/N2/fuel/warm-start logic exists in parent code |

## 3. Parent JSBSim Implementation (my_uav_env/simulator.py)

AircraftSimulator.reload() (lines 351-401):
```python
self.jsbsim_exec = jsbsim.FGFDMExec(data_dir)
self.jsbsim_exec.set_debug_level(0)
self.jsbsim_exec.load_model(self.model)       # "f16" or "f22"
Catalog.add_jsbsim_props(...)
self.jsbsim_exec.set_dt(self.dt)

# Clear default ICs, apply new initial state
self._clear_default_condition()
for key, value in self.init_state.items():
    self.set_property_value(Catalog[key], value)

success = self.jsbsim_exec.run_ic()

# Engine startup — identical for all models
propulsion = self.jsbsim_exec.get_propulsion()
n = propulsion.get_num_engines()
for j in range(n):
    propulsion.get_engine(j).init_running()
propulsion.get_steady_state()
```

**No model-specific engine startup logic exists.** The parent project uses the
same `init_running()` + `get_steady_state()` sequence for both F-16 and any other
aircraft model. There is no N1/N2/starter/fuel/warm-start handling.

## 4. hetero_uav Implementation (uav_env/JSBSim/simulator.py)

**Byte-for-byte identical** to parent `my_uav_env/simulator.py`. The engine startup
sequence is exactly the same.

However, there are TWO sets of configs with different MAV models:

| Config | MAV model | Rewards | Action trim (MAV pitch) | Purpose |
|---|---|---|---|---|
| `hetero_mav_shared_geo_3v2.yaml` | **f22** | brma_legacy | **0.1** | Legacy/diagnostic |
| `hetero_mav_shared_geo_3v2_happo_ref_v0.yaml` | **f22** | happo_ref_v0 | **0.0** | Main F22 experiment |
| `hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml` | **f16** | happo_ref_v0 | **0.0** | F16 surrogate |

## 5. F22-Specific Assumptions

### What the paper says

The TAM-HAPPO paper describes a "heterogeneous MAV/UAV" setting. The **MAV is a
role, not a specific aircraft model**. The MAV has:
- No missiles (surveillance / information sharing role)
- Shared observation capability (MAV detects enemies and shares tracks with UAVs)
- Survival priority

### What hetero_uav implemented

- F-22 was chosen as the MAV aircraft model in the main config
- F-16 was chosen as the UAV aircraft model
- F-16 surrogate was provided as an alternative when F-22 proved unstable
- The project's own docs (`paper_experiment_setup.md:23-25`) state: "The F-16
  MAV surrogate is an engineering choice for controllability... It should not be
  described as the physical MAV model from the original heterogeneous paper."

### Conclusion

**F22 is NOT a paper requirement.** It is our implementation choice. The paper
does not specify F22. The project's own documentation already recommends F16
surrogate as the main experimental path.

## 6. Engine Startup Evidence

### F22 thrust audit findings

The `check_f22_action_path.py` script showed:
- Engine thrust at frame 1: **0.0 lbs** (engine not producing thrust yet)
- Engine thrust at frame 60: **1,849 lbs** (~8.2 kN)
- This is extremely low compared to F119 rated thrust (~35,000 lbs military, ~156 kN)

### Key finding: check_f22_action_path.py used wrong config

The script hardcodes `MAIN_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"`
(line 22). This is the **old** config with:
- `hetero_reward_mode: brma_legacy` (not `happo_ref_v0`)
- `action_trim_by_role.mav.pitch: 0.1` (not 0.0)

**The 0.1 pitch trim observed in the control chain audit came from this old config,
NOT from the official training config.**

The official training used `hetero_mav_shared_geo_3v2_happo_ref_v0.yaml` which has
`action_trim_by_role.mav.pitch: 0.0`.

### Engine thrust root cause — insufficient evidence

We cannot conclude the F22 engine thrust is a "bug" because:
1. We have not compared F16 vs F22 thrust under identical conditions
2. The control chain audit used a legacy config, not the official training config
3. The parent project does not have any F22-specific engine logic to compare against
4. No paper specifies expected thrust values for F22/F119 in this context

The low thrust could be:
- Correct JSBSim F22 model behavior (F119 has complex engine spool-up dynamics)
- An initialization issue (`init_running()` may not be sufficient for F119)
- A throttle command mapping issue (the PID output may not reach the correct
  JSBSim throttle properties)
- An altitude/density effect (6000m altitude reduces available thrust)
- Or all of the above

**We have insufficient evidence to modify engine initialization.**

## 7. What Must NOT Be Changed Without Explicit Approval

1. F22/F119 engine startup logic in `simulator.py`
2. PID gains or PID structure in `pid_controller.py`
3. Aircraft XML or engine XML files
4. Action trim values in official configs (F22 = 0.0, F16 surrogate = 0.0)
5. Aircraft model selection in official configs
6. `init_running()` / `get_steady_state()` sequence

## 8. Gate Decision: **B**

**The paper does not specify F22 as the MAV platform.**

**The parent project does not have F22-specific engine startup logic.**

**We have insufficient evidence that F22 engine initialization is bugged.**

Therefore:
- **Do NOT modify engine startup.**
- **Do NOT claim F22 is a paper requirement.**
- **Run a focused F16 vs F22 thrust comparison diagnostic** before making any
  further claims about engine behavior.
- **Fix `check_f22_action_path.py`** to accept `--config` parameter and use the
  official config by default.

## 9. Recommended Next Action

1. **Add `--config` parameter to `check_f22_action_path.py`** so it uses the
   official `happo_ref_v0` config (not legacy `hetero_mav_shared_geo_3v2.yaml`).

2. **Run an F16 vs F22 thrust diagnostic** using the official configs, comparing
   engine thrust at frame 60 for both models under identical fixed actions.

3. **If F22 thrust is confirmed low relative to F16 under identical conditions:**
   - This is a JSBSim model issue, not a code bug
   - Options: use F16 surrogate (paper-supported), or investigate F22 JSBSim model
   - Do NOT patch engine startup without explicit review

4. **If F22 thrust is comparable to F16:**
   - The original "F22 uncontrollable" diagnosis was based on wrong config
   - Re-run F22 training with the sanitize fix

5. **Regardless of thrust findings:**
   - Update `paper_experiment_setup.md` to remove any implication that F22 is required
   - Document that the paper does not mandate a specific aircraft model
