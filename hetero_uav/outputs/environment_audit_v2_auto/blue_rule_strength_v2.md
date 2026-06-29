# Blue Rule Strength V2

## level_flight_red_vs_blue_rule
- blue launches recorded: 14
- MAV target launches: 2
- first launch steps: [523, 523, 888, 888, 523, 523, 888, 888, 741, 741, 284, 284, 936, 936]
- outcomes: {'draw_or_timeout': 18}

## blue_rule_only_strength_probe
- blue launches recorded: 14
- MAV target launches: 2
- first launch steps: [523, 523, 888, 888, 523, 523, 888, 888, 741, 741, 284, 284, 936, 936]
- outcomes: {'draw_or_timeout': 18}

## zero_action_red_vs_blue_rule
- blue launches recorded: 12
- MAV target launches: 0
- first launch steps: [314, 314, 316, 316, 314, 314, 316, 316, 321, 321, 390, 390]
- outcomes: {'draw_or_timeout': 18}

## straight_chase_red_vs_blue_rule
- blue launches recorded: 4
- MAV target launches: 0
- first launch steps: [183, 183, 183, 183]
- outcomes: {'red_win': 12, 'draw_or_timeout': 6}

## oracle_geometry_red_vs_blue_rule
- blue launches recorded: 4
- MAV target launches: 0
- first launch steps: [183, 183, 183, 183]
- outcomes: {'red_win': 12, 'draw_or_timeout': 6}

## red_rule_vs_blue_rule_symmetric_all_attack
- blue launches recorded: 14
- MAV target launches: 4
- first launch steps: [516, 516, 354, 354, 356, 356]
- outcomes: {'blue_win': 6}

## oracle_launch_window_red_vs_blue_rule
- blue launches recorded: 2
- MAV target launches: 2
- first launch steps: [656, 656]
- outcomes: {'red_win': 12, 'draw_or_timeout': 6}

## obs_limited_chase_red_vs_blue_zero
- blue launches recorded: 6
- MAV target launches: 2
- first launch steps: [610, 610, 901, 901]
- outcomes: {'draw_or_timeout': 10, 'red_win': 8}

## oracle_launch_window_red_vs_blue_zero
- blue launches recorded: 4
- MAV target launches: 0
- first launch steps: [756, 756]
- outcomes: {'red_win': 12, 'draw_or_timeout': 6}

## Interpretation Guardrail
- Blue rule strength is conditional on red trajectory and launch geometry; do not label it simply as too strong without the per-policy rows above.