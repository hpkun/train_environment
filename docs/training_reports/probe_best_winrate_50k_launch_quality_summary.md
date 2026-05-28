# Launch quality summary

Source: `results/probe_best_winrate_50k_launch_quality.csv`

| Group | Launches | Hits | Hit rate | Range mean | Range p25 | Range p50 | Range p75 | AO mean | AO p50 | TA mean | TA p50 | Closing mean | Abs altitude diff mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red all | 765 | 10 | 0.0131 | 4746.8669 | 3158.8943 | 4856.5462 | 6504.7989 | 23.2105 | 24.8231 | 148.7222 | 153.9211 | -151.3554 | 1710.1707 |
| Blue all | 61 | 56 | 0.9180 | 4327.2871 | 1489.5573 | 2239.3644 | 9608.1276 | 20.8299 | 18.1507 | 104.1073 | 94.6228 | 191.7090 | 1111.7734 |
| Red hit | 10 | 10 | 1.0000 | 6143.4446 | 5986.4562 | 6802.7625 | 7920.6402 | 23.8150 | 23.2242 | 145.2253 | 149.4661 | -78.1877 | 1170.5649 |
| Red miss | 755 | 0 | 0.0000 | 4728.3692 | 3158.3492 | 4831.4540 | 6437.5537 | 23.2025 | 24.9656 | 148.7685 | 153.9211 | -152.3245 | 1717.3178 |
| Blue hit | 56 | 56 | 1.0000 | 4653.1063 | 1715.5963 | 2328.6830 | 9923.2130 | 20.0463 | 18.0846 | 104.8204 | 94.3581 | 204.4905 | 1074.5620 |
| Blue miss | 5 | 0 | 0.0000 | 678.1115 | 613.2850 | 685.8224 | 739.9507 | 29.6053 | 36.5744 | 96.1203 | 97.5640 | 48.5569 | 1528.5416 |

## Diagnosis

- Red misses are not farther than Red hits on average.
- Red misses do not have worse AO than Red hits on average.
- Red misses are not closer to the 45 deg AO launch boundary than Red hits.
- Red misses do not show weaker TA than Red hits on average.
- Red misses are not closer to the 90 deg TA launch boundary than Red hits.
- Red misses have worse closing speed than Red hits on average.
- Blue launch quality is better by realized hit rate.
