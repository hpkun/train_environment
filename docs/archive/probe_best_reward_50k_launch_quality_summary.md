# Launch quality summary

Source: `results/probe_best_reward_50k_launch_quality.csv`

| Group | Launches | Hits | Hit rate | Range mean | Range p25 | Range p50 | Range p75 | AO mean | AO p50 | TA mean | TA p50 | Closing mean | Abs altitude diff mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red all | 342 | 11 | 0.0322 | 5438.7427 | 3348.6014 | 5513.4759 | 7643.8037 | 25.0448 | 29.1992 | 144.9299 | 164.1905 | -101.7396 | 2248.7829 |
| Blue all | 47 | 39 | 0.8298 | 4216.8143 | 1362.0778 | 2287.0134 | 8992.3543 | 19.9437 | 16.3163 | 102.2087 | 94.1879 | 171.7372 | 1458.0717 |
| Red hit | 11 | 11 | 1.0000 | 6550.2598 | 3659.8845 | 8246.7869 | 9484.7706 | 23.0157 | 32.1462 | 117.5297 | 100.7247 | 25.3530 | 1828.8261 |
| Red miss | 331 | 0 | 0.0000 | 5401.8041 | 3350.2287 | 5495.7623 | 7533.7633 | 25.1122 | 29.1524 | 145.8405 | 166.0298 | -105.9632 | 2262.7391 |
| Blue hit | 39 | 39 | 1.0000 | 4740.4109 | 1527.5711 | 2568.5919 | 9926.6114 | 18.5772 | 15.7346 | 103.7874 | 94.1063 | 195.3621 | 1257.3555 |
| Blue miss | 8 | 0 | 0.0000 | 1664.2806 | 917.4771 | 1136.0505 | 2226.7169 | 26.6054 | 33.6502 | 94.5125 | 95.1324 | 56.5656 | 2436.5632 |

## Diagnosis

- Red misses are not farther than Red hits on average.
- Red misses have worse AO than Red hits on average.
- Red misses are closer to the 45 deg AO launch boundary than Red hits.
- Red misses do not show weaker TA than Red hits on average.
- Red misses are not closer to the 90 deg TA launch boundary than Red hits.
- Red misses have worse closing speed than Red hits on average.
- Blue launch quality is better by realized hit rate.
