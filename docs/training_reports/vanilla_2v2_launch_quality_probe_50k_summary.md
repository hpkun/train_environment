# Launch quality summary

Source: `results/vanilla_2v2_launch_quality_probe_50k_launch_quality.csv`

| Group | Launches | Hits | Hit rate | Range mean | Range p25 | Range p50 | Range p75 | AO mean | AO p50 | TA mean | TA p50 | Closing mean | Abs altitude diff mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red all | 391 | 7 | 0.0179 | 6887.7522 | 5610.6307 | 7593.9115 | 8610.4358 | 22.3666 | 19.0785 | 137.6714 | 134.6938 | -122.5038 | 1067.5079 |
| Blue all | 54 | 50 | 0.9259 | 4640.5073 | 1130.3913 | 3681.2735 | 9460.5271 | 25.0489 | 28.5096 | 110.7083 | 96.2222 | 175.5831 | 1174.7043 |
| Red hit | 7 | 7 | 1.0000 | 6613.5322 | 5234.9425 | 7682.0177 | 7965.2080 | 29.9417 | 33.0731 | 107.7852 | 110.5930 | 29.1953 | 1022.8792 |
| Red miss | 384 | 0 | 0.0000 | 6892.7510 | 5615.7438 | 7579.7488 | 8614.2523 | 22.2285 | 18.9688 | 138.2162 | 135.9247 | -125.2691 | 1068.3214 |
| Blue hit | 50 | 50 | 1.0000 | 4948.5659 | 1585.1543 | 4164.4865 | 9815.1557 | 24.3099 | 26.6889 | 111.8085 | 95.9837 | 178.3973 | 1211.2283 |
| Blue miss | 4 | 0 | 0.0000 | 789.7744 | 728.6661 | 821.6160 | 882.7244 | 34.2866 | 40.9882 | 96.9559 | 97.4803 | 140.4059 | 718.1549 |

## Diagnosis

- Red misses are launched farther than Red hits on average.
- Red misses do not have worse AO than Red hits on average.
- Red misses are not closer to the 45 deg AO launch boundary than Red hits.
- Red misses do not show weaker TA than Red hits on average.
- Red misses are not closer to the 90 deg TA launch boundary than Red hits.
- Red misses have worse closing speed than Red hits on average.
- Blue launch quality is better by realized hit rate.
