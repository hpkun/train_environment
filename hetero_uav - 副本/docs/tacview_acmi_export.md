# Tacview ACMI Export

## Purpose

The ACMI export tool is for environment visualization audit. It is not training,
not an evaluation metric, and not a method component.

Use it to inspect initial geometry, blue search/acquisition behavior, MAV
situation position, and whether paper-aligned scenarios visually match the
intended environment protocol.

## Relationship To Original Project

`hetero_uav` reuses the original project's `TacviewLogger` implementation under
`uav_env/JSBSim/render_tacview.py`. The logger writes Tacview 2.1 text ACMI
files. Aircraft and missile states are read from JSBSim simulator methods such
as `get_geodetic()` and `get_rpy()`.

The original environment imports and can use `TacviewLogger`, but this tool adds
a standalone hetero environment rollout exporter.

## Usage

```powershell
python scripts/export_hetero_tacview_acmi.py `
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml `
  --steps 300 `
  --red-policy zero `
  --blue-policy greedy_fsm `
  --output-acmi outputs/tacview/paper_3v2_greedy_fsm.acmi `
  --output-json outputs/tacview/paper_3v2_greedy_fsm_meta.json `
  --record-missiles
```

Supported audit policies:

- red: `zero`, `random`
- blue: `zero`, `rule_nearest`, `greedy_fsm`, `random`

No MAPPO model is loaded.

## How To Open

Open the `.acmi` file with Tacview. If Tacview is not installed, inspect the
text header and frame markers:

- `FileType=text/acmi/tacview`
- `FileVersion=2.1`
- `ReferenceTime`
- `#0`, `#<seconds>` frame markers

## Interpretation

- `red_0` should be the MAV in paper-aligned hetero configs.
- Blue aircraft should move according to the selected scripted policy.
- If blue never approaches red, use the ACMI to diagnose geometry or policy.
- If missiles appear and `--record-missiles` is enabled, missile tracks are
  exported as separate entities.

## Limitations

- Sensor cones are not recorded.
- Communication links are not recorded.
- Explosion rendering is a basic marker only.
- The exporter does not change environment mechanics.
- The exported rollout is an audit artifact, not final opponent validation.
