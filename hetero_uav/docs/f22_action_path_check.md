# F-22 Action Path Check

## Why This Check

F-22 was selected as the MAV model to align with the heterogeneous paper's
MAV visual profile.  Before continuing with missile audit or training, we
must confirm that high-level actions actually reach JSBSim control
properties and produce distinguishable aircraft responses.

## What This Checks

- Whether `fcs/elevator-cmd-norm`, `fcs/aileron-cmd-norm`,
  `fcs/rudder-cmd-norm`, `fcs/throttle-cmd-norm` are populated
- Whether climb/descend, turn left/right, speed up/down produce
  distinguishable state changes
- Whether MAV action trim (inherited from A-4) interferes with F-22

## What This Does NOT Do

- Modify missile, reward, termination, PID, or aircraft XML
- Run training
- Make performance claims

## Pass Criteria

- FCS control properties are readable and non-zero
- Pitch, heading, and speed responses are directionally correct
- No crash, no NaN

## Next Step

- If passed: continue missile audit
- If failed with trim: try `--disable-mav-trim` diagnostic
- If still failed: reconsider MAV model choice
