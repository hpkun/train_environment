# PASS: paper_minimal_3v3_v1 audit

## Checks

- PASS: `profile_dimensions`
- PASS: `timing`
- PASS: `initial_mirror`
- PASS: `fingerprint_isolation`
- PASS: `finite`
- PASS: `no_invalid_episodes`
- PASS: `fixed_vs_fixed_both_launch`
- PASS: `fixed_red_can_launch_and_hit_straight_blue`
- PASS: `color_swap_has_fire_control_activity`

## Rule-loop scenarios

| scenario | steps | red launches/hits | blue launches/hits | invalid |
|---|---:|---:|---:|---|
| fixed_vs_fixed | 300 | 5/1 | 5/1 | False |
| fixed_red_vs_straight_blue | 300 | 5/1 | 5/1 | False |
| straight_red_vs_fixed_blue | 300 | 5/1 | 5/1 | False |
