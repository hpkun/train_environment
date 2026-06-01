# Models

This debug project keeps aircraft model names such as `A-4` and `F-16` in the
YAML configuration, but the first implementation uses a lightweight kinematic
proxy instead of loading real JSBSim XML aircraft models.

Future work can place JSBSim aircraft and engine model files here and replace
`uav_env.JSBSim.core.aircraft.AircraftPlatform` with a JSBSim-backed platform.
