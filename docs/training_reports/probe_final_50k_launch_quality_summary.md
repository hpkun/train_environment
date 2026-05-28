# Launch quality summary

Source: `results/probe_final_50k_launch_quality.csv`

| Group | Launches | Hits | Hit rate | Range mean | Range p25 | Range p50 | Range p75 | AO mean | AO p50 | TA mean | TA p50 | Closing mean | Abs altitude diff mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red all | 250 | 4 | 0.0160 | 5614.6015 | 3768.3644 | 5841.0765 | 8865.1248 | 31.2931 | 35.9060 | 123.5179 | 115.2173 | -69.5843 | 1666.7794 |
| Blue all | 44 | 38 | 0.8636 | 3738.1279 | 785.6888 | 1404.6872 | 7655.3215 | 24.4174 | 24.9924 | 100.0747 | 94.8414 | 173.4661 | 1342.4752 |
| Red hit | 4 | 4 | 1.0000 | 7564.1222 | 6978.9146 | 7596.7109 | 8181.9185 | 24.2883 | 21.6668 | 125.2636 | 121.8639 | -41.5767 | 1410.8578 |
| Red miss | 246 | 0 | 0.0000 | 5582.9020 | 3666.6677 | 5823.5571 | 8865.1248 | 31.4070 | 36.2043 | 123.4895 | 115.2173 | -70.0397 | 1670.9407 |
| Blue hit | 38 | 38 | 1.0000 | 4201.3885 | 938.8663 | 1771.2003 | 8271.9605 | 23.1388 | 22.9365 | 100.7379 | 94.6718 | 187.7639 | 1297.0357 |
| Blue miss | 6 | 0 | 0.0000 | 804.1442 | 738.5186 | 745.9135 | 852.5049 | 32.5152 | 32.2190 | 95.8748 | 96.2379 | 82.9131 | 1630.2585 |

## Diagnosis

- Red misses are not farther than Red hits on average.
- Red misses have worse AO than Red hits on average.
- Red misses are closer to the 45 deg AO launch boundary than Red hits.
- Red misses have weaker rear-aspect TA than Red hits on average.
- Red misses are closer to the 90 deg TA launch boundary than Red hits.
- Red misses have worse closing speed than Red hits on average.
- Blue launch quality is better by realized hit rate.
