# Termination / Death / Crash Audit
- Episode terminates when all blue or all red aircraft are dead; truncates at max_steps.
- Crash/death reasons include missile hit, low altitude, over-G/extreme/non-finite states depending on env checks.
- Red lacks blue GCAS, so red low-altitude crash risk can be higher under aggressive actions.
- Death reasons are exposed through `info[aid]['death_reason']` and `info['death_events']`.