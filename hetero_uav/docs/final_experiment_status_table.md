# Final Experiment Status Table

| experiment | train scenario | eval scenario | aircraft | policy | reward | MAV survival | red fire | red hit | blue death | win type | conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shared MLP MAPPO 1M | 3v2 | 3v2 / 5v4 | F-22 MAV branch + F-16 UAVs | shared MLP MAPPO | brma_legacy | 0.00 in 100-episode best eval | not reliable | not reliable | 0.00 blue elimination | mostly timeout/draw | weak baseline only |
| HAPPO reference v0 with F-22 MAV | 3v2 | 3v2 / 5v4 | F-22 MAV + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | eval 0.00 | unstable | unstable | not robust | latest blue elimination | F-22 unstable under current interface |
| HAPPO reference v0 F-16 surrogate 200k best | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 1.00 | 0.02 | 0.02 | 0.02 | timeout red alive advantage | survival, almost no combat |
| HAPPO reference v0 F-16 surrogate 200k latest | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.86 | 1.48 | 1.20 | 1.20 | mixed timeout and red elimination | strongest learned attack signal, not stable |
| HAPPO reference v0 F-16 surrogate 1M train latest | 3v2 | training rollout | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 1.00 | 0 | 0 | 0 | timeout survival | survival baseline, not combat |
| HAPPO reference v0 F-16 surrogate 1M best eval 3v2 | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.00 | 0.07 hit proxy | 0.07 | 0.07 | mostly blue alive advantage / draw | not usable combat baseline |
| HAPPO reference v0 F-16 surrogate 1M best eval 5v4 | 3v2 | 5v4 zero-shot | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.00 | 1.72 | 1.24 | 1.23 | timeout alive advantage / draw | some transfer signal, MAV survival fails |
| Red direct chase oracle vs blue zero | none | 3v2 sanity | F-16 MAV surrogate + F-16 UAVs | scripted direct chase | environment fire-control | red team survives in sanity case | 2.00 | 2.00 | 2.00 | red elimination win 1.00 | attack chain works |
| Red direct chase oracle vs blue BRMA rule | none | 3v2 sanity | F-16 MAV surrogate + F-16 UAVs | scripted direct chase | environment fire-control | red team survives in sanity case | 2.25 | 2.00 | 2.00 | red elimination win 1.00 | learned policy lacks engagement behavior |
| BRMA observation alignment test | none | 3v2 / 5v4 contract | not aircraft-dependent | V2 adapter contract | none | not applicable | not applicable | not applicable | not applicable | not a combat test | unified observation contract verified |
