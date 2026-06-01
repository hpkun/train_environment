# JSBSim Models

This directory contains the local model assets used by the optional JSBSim
flight-dynamics backend.

Current assets:

- `aircraft/A-4/A-4.xml`: A-4 model XML provided from the JSBSim upstream A4
  aircraft directory.
- `aircraft/F-16/F-16.xml`: F-16 model copied from the parent project's local
  aircraft data.
- `engine/J52.xml`: A-4 engine model.
- `engine/F100-PW-229.xml`: F-16 engine model.
- `engine/direct.xml`: direct thruster model used by the copied F-16 data.

The environment still defaults to the lightweight `simple` kinematic backend.
Set `dynamics_backend: "jsbsim"` or pass `dynamics_backend="jsbsim"` to
`make_env` to construct `JSBSimAircraftPlatform` instances. The Python
`jsbsim` package must be installed for that mode.
