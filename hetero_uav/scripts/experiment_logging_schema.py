"""Unified rich logging schema for full experiment runs.

The schema is intentionally plain Python so training/eval scripts can import it
without additional dependencies. Missing metrics should be written as empty
strings or NaN, but columns should remain stable.
"""
from __future__ import annotations

from pathlib import Path
import csv


TRAIN_METRICS_COLUMNS = [
    "run_id", "method_name", "scenario_name", "train_steps",
    "total_env_steps_actual", "wall_time_sec", "steps_per_second",
    "avg_episode_return", "avg_team_reward", "avg_mav_reward", "avg_uav_reward",
    "red_win_rate", "blue_win_rate", "draw_rate", "timeout_rate",
    "red_elimination_win_rate", "red_timeout_alive_advantage_rate",
    "mav_survival_rate", "red_alive_final_mean", "blue_alive_final_mean",
    "red_missiles_fired_mean", "blue_missiles_fired_mean",
    "red_missile_hits_mean", "blue_missile_hits_mean",
    "red_dead_mean", "blue_dead_mean", "kill_death_ratio",
    "relative_win_ratio", "actor_loss", "critic_loss", "entropy",
    "policy_gradient_norm", "value_gradient_norm", "action_saturation_rate",
    "mav_action_saturation_rate", "uav_action_saturation_rate",
    "approx_kl_mav", "approx_kl_uav",
    "action_override_rate", "missile_evasion_override_rate", "gcas_override_rate",
    "actor_effective_sample_fraction",
    "engagement_target_valid_rate", "target_switch_count",
    "target_switches_per_episode", "target_hold_steps_mean",
    "reward_target_matches_fire_candidate", "reward_target_matches_lock_given_lock",
    "reward_target_matches_launch_given_launch", "reward_fire_candidate_comparable_count",
    "reward_lock_comparable_count", "reward_launch_comparable_count",
    "mav_reward_total", "uav_reward_total_mean", "uav_reward_total_sum",
    "mav_safety_sum", "mav_support_sum", "mav_event_sum",
    "uav_height_sum", "uav_speed_sum", "uav_angle_sum", "uav_distance_sum",
    "uav_dodge_sum", "uav_event_sum", "dense_reward_signed_sum",
    "dense_reward_abs_sum", "event_reward_signed_sum", "event_reward_abs_sum",
    "dense_event_abs_ratio", "mav_uav_reward_correlation",
    "mav_uav_correlation_sample_count", "mav_uav_sign_conflict_rate",
    "mav_uav_sign_comparable_count", "mav_team_reward_contribution_fraction",
    "uav_team_reward_contribution_fraction", "alive_reward_denominator_mean",
    "mask_keep_ratio", "mask_entropy", "masked_entity_count",
    "nan_detected",
]

SCALE_V1_TRAIN_METRIC_COLUMNS = [
    "effective_scale_v1_total", "effective_scale_v1_identity_error",
    "scale_v1_progress_positive_ratio", "scale_v1_progress_clip_ratio",
]
SCALE_V2_TRAIN_METRIC_COLUMNS = [
    "effective_scale_v2_uav_flight_raw", "effective_scale_v2_uav_flight_scaled",
    "effective_scale_v2_uav_progress", "effective_scale_v2_uav_event",
    "effective_scale_v2_mav_flight_raw", "effective_scale_v2_mav_flight_scaled",
    "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
    "effective_scale_v2_terminal", "effective_scale_v2_total",
    "effective_scale_v2_component_sum", "effective_scale_v2_identity_error",
    "scale_v2_progress_positive_ratio", "scale_v2_progress_negative_ratio",
    "scale_v2_progress_zero_ratio", "scale_v2_progress_clip_ratio",
    "scale_v2_target_switch_count", "scale_v2_terminal_mean",
    "scale_v2_kill_count", "scale_v2_death_count", "scale_v2_oob_count",
    "scale_v2_flight_raw_mean", "scale_v2_flight_scaled_mean",
]
FINAL_PURE_HAPPO_COLUMNS = [
    "final_approx_kl_mav", "final_approx_kl_uav",
    "final_approx_kl_abs_mav", "final_approx_kl_abs_uav",
    "final_clip_fraction_mav", "final_clip_fraction_uav",
    "final_ratio_mean_mav", "final_ratio_mean_uav",
    "final_ratio_std_mav", "final_ratio_std_uav",
    "final_ratio_p95_mav", "final_ratio_p95_uav",
    "final_ratio_p99_mav", "final_ratio_p99_uav",
    "final_actor_parameter_delta_mav", "final_actor_parameter_delta_uav",
]
PURE_HAPPO_LEARNABILITY_COLUMNS = [
    "actor_loss_mav", "actor_loss_uav",
    "critic_loss_scaled", "critic_loss_unscaled",
    "value_explained_variance_old", "value_explained_variance_new",
    "return_mean", "return_std",
    "value_pred_old_mean", "value_pred_old_std",
    "value_pred_new_mean", "value_pred_new_std",
    "advantage_raw_mean", "advantage_raw_std", "advantage_raw_abs_max",
    "actor_grad_norm_mav", "actor_grad_norm_uav", "critic_grad_norm",
    "policy_update_norm_mav", "policy_update_norm_uav", "critic_update_norm",
    "entropy_mav", "entropy_uav",
    "action_log_std_mav_mean", "action_log_std_uav_mean",
    "correction_M_mean", "correction_M_std", "correction_M_mean_abs",
    "correction_M_max_abs", "reward_nan_count", "action_nan_count",
    "value_nan_count", "log_prob_nan_count", "gradient_nonfinite_count",
]
for _col in SCALE_V1_TRAIN_METRIC_COLUMNS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)
for _col in SCALE_V2_TRAIN_METRIC_COLUMNS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)
for _col in FINAL_PURE_HAPPO_COLUMNS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)
for _col in PURE_HAPPO_LEARNABILITY_COLUMNS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)
# V3 role-situation effective fields (TRAIN_METRICS only -- episode fields appended later)
from uav_env.JSBSim.envs.role_situation_v3 import V3_EFFECTIVE_FIELDS
for _col in V3_EFFECTIVE_FIELDS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)

EVAL_EPISODE_COLUMNS = [
    "run_id", "checkpoint_name", "eval_scenario", "episode_id", "seed",
    "outcome", "episode_return", "team_reward", "mav_reward",
    "uav_reward_mean", "episode_length", "red_win", "blue_win", "draw",
    "timeout", "red_elimination_win", "red_timeout_alive_advantage",
    "mav_alive", "red_alive_final", "blue_alive_final", "red_missiles_fired",
    "blue_missiles_fired", "red_missile_hits", "blue_missile_hits",
    "red_dead", "blue_dead", "kill_death_ratio", "relative_win_ratio",
    "first_red_fire_time", "first_blue_fire_time", "first_hit_time",
    "first_death_time",
]

EVAL_SUMMARY_COLUMNS = [
    "checkpoint_name", "eval_scenario", "episodes", "avg_episode_return_mean",
    "avg_episode_return_std", "red_win_rate", "blue_win_rate", "draw_rate",
    "timeout_rate", "red_elimination_win_rate",
    "red_timeout_alive_advantage_rate", "mav_survival_rate",
    "red_alive_final_mean", "blue_alive_final_mean",
    "red_missiles_fired_mean", "blue_missiles_fired_mean",
    "red_missile_hits_mean", "blue_missile_hits_mean",
    "red_dead_mean", "blue_dead_mean", "kill_death_ratio",
    "relative_win_ratio", "red_win_rate_ci95",
]

AIRCRAFT_TIMESERIES_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "agent_id",
    "role", "team", "alive", "lon", "lat", "altitude", "roll", "pitch",
    "yaw", "heading", "velocity", "mach", "speed", "alpha", "beta",
    "action_pitch", "action_heading", "action_speed", "action_raw_0",
    "action_raw_1", "action_raw_2", "nearest_enemy_id",
    "nearest_enemy_distance", "target_id", "missile_warning", "is_mav",
    "is_uav",
]

MISSILE_EVENTS_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "event_type",
    "missile_id", "owner_id", "owner_team", "target_id", "target_team",
    "team", "shooter_id", "shooter_role", "target_role", "hit",
    "lon", "lat", "altitude", "distance_to_target", "hit_success",
    "death_caused",
    "raw_termination_reason", "termination_reason",
    "AO_rad", "AO_deg",
    "TA_rad", "TA_deg",
    "flight_time_sec",
    "launch_step", "termination_step", "step_delta",
    "target_alive_at_launch", "target_alive_at_termination",
    "shooter_speed_mps", "target_speed_mps", "closing_speed_mps",
    "shooter_alt_m", "target_alt_m",
    "launch_track_source", "launch_track_ok", "launch_track_block_reason",
    "range_3d_m", "range_2d_m",
    "ATA_3d_rad", "TA_3d_rad", "boresight_3d_rad",
    "los_elevation_body_rad", "los_azimuth_body_rad",
    "target_relative_altitude_m",
    "launch_geometry_ok_3d", "range_ok_3d", "ata_ok_3d",
    "ta_ok_3d", "boresight_ok_3d",
    "legacy_AO_2d_rad", "legacy_TA_2d_rad", "legacy_range_2d_m",
    "target_selection_mode", "selected_target_score",
    "selected_target_threat_score", "selected_target_mav_support_score",
    "selected_target_shot_quality_score", "selected_target_range_m",
    "selected_target_AO_rad", "selected_target_TA_rad",
    "selected_target_is_mav_observed", "candidate_count",
    "lock_target_id_at_launch", "lock_timer_frames_at_launch",
    "min_range_m", "directional_match_at_hit_check", "P_hit_at_hit_check",
    "speed_at_termination_mps", "closing_speed_at_termination_mps",
    "evasion_triggered", "evasion_team", "evasion_agent_id",
    "incoming_missile_id", "incoming_range_m",
    "incoming_closing_speed_mps", "incoming_t_go_sec", "evasion_mode",
]

MISSILE_TIMESERIES_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "missile_id",
    "owner_id", "target_id", "alive", "lon", "lat", "altitude", "speed",
]

REWARD_COMPONENT_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time", "agent_id",
    "role", "total_reward", "mav_survival_reward", "mav_support_reward",
    "uav_attack_reward", "uav_fire_reward", "uav_hit_reward", "event_reward",
    "tam_v7_total", "tam_v7_flight", "tam_v7_event", "tam_v7_terminal",
    "tam_v7_uav_flight", "tam_v7_uav_situation", "tam_v7_uav_event",
    "tam_v7_uav_terminal", "tam_v7_uav_total", "tam_v7_uav_pitch",
    "tam_v7_uav_roll", "tam_v7_uav_altitude", "tam_v7_uav_speed",
    "tam_v7_uav_boundary", "tam_v7_uav_own_adv_mean",
    "tam_v7_uav_enemy_threat_mean", "tam_v7_uav_distance_ref_m",
    "tam_v7_uav_situation_raw", "tam_v7_uav_kill", "tam_v7_uav_death",
    "tam_v7_uav_first_out_of_zone",
    "tam_v7_mav_flight", "tam_v7_mav_safety", "tam_v7_mav_support",
    "tam_v7_mav_event", "tam_v7_mav_terminal", "tam_v7_mav_total",
    "tam_v7_mav_pitch", "tam_v7_mav_roll", "tam_v7_mav_altitude",
    "tam_v7_mav_speed", "tam_v7_mav_boundary", "tam_v7_mav_safety_raw",
    "tam_v7_mav_safety_dist", "tam_v7_mav_safety_threat",
    "tam_v7_mav_safety_aspect", "tam_v7_mav_support_raw",
    "tam_v7_mav_support_pos", "tam_v7_mav_support_aware",
    "tam_v7_mav_death", "tam_v7_mav_team_credit_delta",
    "tam_v7_mav_team_credit_used", "tam_v7_terminal_per_agent",
    "tam_v7_blue_loss_frac", "tam_v7_red_loss_weighted",
    "tam_v7_shared_track_usage_log", "tam_v7_red_fire_with_mav_track_log",
    "tam_v7_red_hit_with_mav_track_log",
    "brma_role_no_missile_total", "brma_role_no_missile_active",
    "brma_role_active_brma_flight", "brma_role_active_brma_situation",
    "brma_role_active_brma_terminal", "brma_role_removed_situation",
    "brma_role_situation_active", "brma_role_removed_situation_is_weighted",
    "brma_role_is_mav",
    "paper_v1_uav_flight", "paper_v1_uav_adv", "paper_v1_uav_end",
    "paper_v1_uav_r_death_log", "paper_v1_uav_total",
    "paper_v1_mav_flight", "paper_v1_mav_removed_r_adv",
    "paper_v1_mav_removed_r_end", "paper_v1_mav_r_death_log",
    "paper_v1_mav_safety", "paper_v1_mav_dist", "paper_v1_mav_threat",
    "paper_v1_mav_aspect", "paper_v1_mav_nearest_blue_distance_m",
    "paper_v1_mav_safety_danger_m", "paper_v1_mav_safety_safe_m",
    "paper_v1_mav_support", "paper_v1_mav_pos", "paper_v1_mav_aware",
    "paper_v1_mav_aware_observed_count",
    "paper_v1_mav_battlefield_center_x", "paper_v1_mav_battlefield_center_y",
    "paper_v1_mav_pos_distance_m",
    "paper_v1_mav_event_raw", "paper_v1_mav_event_death_raw",
    "paper_v1_mav_event_team_credit_delta_raw",
    "paper_v1_mav_event_team_credit_used_raw",
    "paper_v1_mav_event_team_credit_cap_raw",
    "paper_v1_mav_scaled_tam", "paper_v1_mav_total",
    "paper_v1_mav_shared_track_log",
    "paper_v1_red_launch_with_mav_shared_track_log",
    "paper_v1_red_hit_with_mav_shared_track_log",
    "tam_table1_uav_height", "tam_table1_uav_height_pv",
    "tam_table1_uav_height_ph", "tam_table1_uav_speed",
    "tam_table1_uav_angle", "tam_table1_uav_distance",
    "tam_table1_uav_dodge", "tam_table1_uav_dodge_angle",
    "tam_table1_uav_dodge_speed", "tam_table1_uav_event",
    "tam_table1_uav_kill", "tam_table1_uav_death",
    "tam_table1_uav_out_of_zone", "tam_table1_uav_total",
    "tam_table1_uav_target_id_log", "tam_table1_uav_target_distance_km",
    "tam_table1_uav_target_ata_rad", "tam_table1_uav_target_aa_rad",
    "tam_table1_uav_missing_dodge_geometry",
    "tam_table1_uav_brma_adv_log", "tam_table1_uav_brma_end_log",
    "tam_table1_uav_launch_with_mav_shared_track_log",
    "tam_table1_uav_hit_with_mav_shared_track_log",
    "tam_table1_mav_safety", "tam_table1_mav_dist",
    "tam_table1_mav_threat", "tam_table1_mav_aspect",
    "tam_table1_mav_support", "tam_table1_mav_pos",
    "tam_table1_mav_aware", "tam_table1_mav_event",
    "tam_table1_mav_death", "tam_table1_mav_team_credit_delta",
    "tam_table1_mav_team_credit_used", "tam_table1_mav_team_credit_cap",
    "tam_table1_mav_total", "tam_table1_mav_support_anchor_x",
    "tam_table1_mav_support_anchor_y",
    "tam_table1_mav_support_distance_m",
    "tam_table1_mav_observed_count",
    "tam_table1_mav_removed_brma_adv_log",
    "tam_table1_mav_removed_brma_end_log",
    "tam_table1_mav_shared_track_slots_log",
    "tam_table1_red_launch_with_mav_shared_track_log",
    "tam_table1_red_hit_with_mav_shared_track_log",
    "tam_table1_total",
    "v1_mav_safety", "v1_mav_safety_dist", "v1_mav_safety_threat",
    "v1_mav_safety_aspect", "v1_mav_safety_danger_m",
    "v1_mav_safety_safe_m", "v1_mav_blue_launch_window_on_mav_log",
    "v1_mav_support", "v1_mav_support_pos", "v1_mav_support_pos_active",
    "v1_mav_support_aware", "v1_mav_support_observed_count",
    "v1_mav_support_aware_raw", "v1_mav_event", "v1_mav_event_death",
    "v1_mav_event_team_credit_delta", "v1_mav_event_team_credit_used",
    "v1_mav_event_team_credit_cap", "v1_mav_removed_r_adv",
    "v1_mav_removed_r_end", "v1_mav_removed_v0_overlay",
    "v1_mav_flight_base", "v1_mav_total_pre_clip", "v1_mav_total",
    "mav_observed_ratio", "mav_shared_track_ratio",
    "red_launch_with_mav_shared_track", "red_hit_with_mav_shared_track",
    "team_kill_while_mav_alive", "team_kill_after_mav_death",
    "red_launch_before_mav_death", "red_launch_after_mav_death",
    "red_uav_alive_steps_before_mav_death",
    "red_uav_alive_steps_after_mav_death",
    "red_launch_rate_before_mav_death", "red_launch_rate_after_mav_death",
    "mav_reward_safety_sum", "mav_reward_support_sum",
    "mav_reward_event_sum", "mav_reward_total_sum",
    "mav_removed_r_adv_sum", "mav_removed_r_end_sum",
]

BRMA_TAM_SCRIPTED_COMPONENT_COLUMNS = [
    "reward_contract_revision", "brma_pitch", "brma_roll", "brma_vel", "brma_alt_log_only",
    "brma_bound_log_only", "brma_adv_log_only", "brma_end_log_only",
    "brma_death_log_only", "tam_speed_raw", "tam_speed_weighted",
    "own_speed_mps", "target_speed_mps", "speed_ratio", "speed_ratio_valid",
    "tam_angle_raw", "tam_angle_weighted", "tam_ata_rad", "tam_aa_rad",
    "tam_geometry_valid", "tam_distance_raw", "tam_distance_weighted",
    "target_distance_m", "reward_target_distance_m", "reward_distance_zone_code",
    "launch_range_ok", "below_min_launch_range", "tam_dodge_raw_log",
    "tam_dodge_angle_log", "tam_dodge_speed_log", "tam_dodge_geometry_valid",
    "tam_dodge_missing_reason", "evasion_override_active",
    "script_selected_missile_numeric", "incoming_range_m",
    "incoming_closing_speed_mps", "incoming_t_go_sec",
    "reward_target_observed", "reward_target_direct_visible",
    "reward_target_mav_shared_visible", "reward_target_unavailable",
    "reward_target_matches_lock", "reward_target_matches_launch",
    "reward_target_switch_count", "reward_target_track_source_direct",
    "reward_target_track_source_mav_shared",
    "reward_target_track_source_direct_and_mav_shared",
    "reward_target_track_source_unknown",
    "uav_event_kill", "uav_event_loss", "uav_event_first_horizontal_out_of_zone",
    "uav_event_total", "uav_total", "above_altitude_max_steps",
    "max_altitude_m", "above_altitude_max_episode_flag",
    "mav_dist_raw", "mav_dist_weighted", "mav_nearest_blue_distance_m",
    "mav_threat_raw", "mav_threat_weighted", "mav_actual_incoming_missile_count",
    "mav_prelaunch_geometry_threat_log", "mav_prelaunch_geometry_threat_count_log",
    "mav_aspect_raw_sum",
    "mav_aspect_weighted", "mav_aspect_per_blue_mean", "alive_blue_count",
    "mav_pos_raw", "mav_pos_weighted", "battlefield_center_x",
    "battlefield_center_y", "battlefield_center_valid", "attack_uav_alive_count",
    "all_attack_uav_dead", "steps_after_all_attack_uav_dead",
    "mav_reward_after_all_attack_uav_dead", "mav_center_distance_m",
    "mav_aware_raw", "mav_aware_raw_sum", "mav_aware_per_blue_mean",
    "mav_aware_weighted", "mav_observed_blue_count",
    "mav_alive_blue_count", "mav_observation_coverage_log",
    "mav_shared_track_slot_count_log", "mav_shared_track_unique_blue_count_log",
    "mav_shared_track_count_log", "mav_event_death", "mav_team_credit_delta",
    "mav_team_credit_used", "mav_event_total", "mav_total",
    "brma_tam_scripted_composite_total",
    # -- diagnostic additions (revision 2 log-only) --
    "reward_target_valid", "effective_launch_min_range_m", "effective_launch_max_range_m",
    "mav_support_after_all_attack_uav_dead", "mav_safety_after_all_attack_uav_dead",
    "mav_flight_after_all_attack_uav_dead", "mav_event_after_all_attack_uav_dead",
    "mav_total_after_all_attack_uav_dead",
    "evasion_override_agent_steps", "evasion_override_env_steps",
    "above_altitude_max_agent_steps", "above_altitude_max_env_steps",
    "env_idx", "episode_uid",
]

for _col in BRMA_TAM_SCRIPTED_COMPONENT_COLUMNS:
    if _col not in REWARD_COMPONENT_COLUMNS:
        REWARD_COMPONENT_COLUMNS.append(_col)

BRMA_TAM_SCALE_V1_COMPONENT_COLUMNS = [
    "scale_v1_flight_pitch", "scale_v1_flight_roll", "scale_v1_flight_altitude",
    "scale_v1_flight_boundary", "scale_v1_flight_velocity", "scale_v1_flight_total",
    "scale_v1_brma_adv_log_only", "scale_v1_brma_end_log_only",
    "scale_v1_brma_death_log_only", "scale_v1_phi_distance", "scale_v1_phi_angle",
    "scale_v1_phi_speed", "scale_v1_delta_distance", "scale_v1_delta_angle",
    "scale_v1_delta_speed", "scale_v1_progress_raw", "scale_v1_progress_clipped",
    "scale_v1_progress_reset_flag", "scale_v1_progress_reset_reason",
    "scale_v1_reward_target_id", "scale_v1_reward_target_distance_m",
    "scale_v1_reward_target_valid", "scale_v1_geometry_valid",
    "scale_v1_reward_target_switch_count", "scale_v1_uav_event_kill",
    "scale_v1_uav_event_death", "scale_v1_uav_event_oob",
    "scale_v1_uav_event_total", "scale_v1_uav_total", "scale_v1_mav_dist_raw",
    "scale_v1_mav_threat_raw", "scale_v1_mav_aspect_raw_sum",
    "scale_v1_mav_aspect_mean", "scale_v1_mav_pos_raw",
    "scale_v1_mav_aware_raw_sum", "scale_v1_mav_aware_mean",
    "scale_v1_mav_center_distance_m", "scale_v1_mav_alive_blue_count",
    "scale_v1_mav_role_raw", "scale_v1_mav_role", "scale_v1_mav_event_death",
    "scale_v1_mav_team_credit_delta", "scale_v1_mav_team_credit_used",
    "scale_v1_mav_event_total", "scale_v1_mav_total", "scale_v1_blue_loss_fraction",
    "scale_v1_red_loss_fraction", "scale_v1_terminal", "scale_v1_terminal_applied",
    "scale_v1_total", "scale_v1_identity_error",
]
for _col in BRMA_TAM_SCALE_V1_COMPONENT_COLUMNS:
    if _col not in REWARD_COMPONENT_COLUMNS:
        REWARD_COMPONENT_COLUMNS.append(_col)
from uav_env.JSBSim.envs.role_situation_v3 import V3_REWARD_COMPONENT_FIELDS
for _col in V3_REWARD_COMPONENT_FIELDS:
    if _col not in REWARD_COMPONENT_COLUMNS:
        REWARD_COMPONENT_COLUMNS.append(_col)
from uav_env.JSBSim.envs.paper_calibrated_v4 import V4_COMPONENT_FIELDS
for _col in V4_COMPONENT_FIELDS:
    if _col not in REWARD_COMPONENT_COLUMNS:
        REWARD_COMPONENT_COLUMNS.append(_col)
from uav_env.JSBSim.envs.paper_formula_v5 import (
    V5_COMPONENT_FIELDS,
    V5_EPISODE_LAST_FIELDS,
    V5_EPISODE_STRING_FIELDS,
    V5_TRAIN_FIELDS,
)
for _col in V5_TRAIN_FIELDS:
    if _col not in TRAIN_METRICS_COLUMNS:
        TRAIN_METRICS_COLUMNS.append(_col)
for _col in V5_COMPONENT_FIELDS:
    if _col not in REWARD_COMPONENT_COLUMNS:
        REWARD_COMPONENT_COLUMNS.append(_col)

REWARD_TARGET_DIAGNOSTICS_COLUMNS = [
    "run_id", "scenario", "episode_id", "step", "sim_time",
    "agent_id", "reward_target_id", "reward_target_distance_m",
    "reward_target_observed", "reward_target_direct_visible",
    "reward_target_mav_shared_visible", "reward_target_unavailable",
    "lock_target_id", "lock_timer_frames", "launch_target_id",
    "launch_target_ids", "launch_count_this_step",
    "reward_target_matches_lock", "reward_target_matches_launch",
    "reward_target_switch_count",
    "reward_target_track_source", "script_selected_missile_id",
    "tam_dodge_geometry_valid", "tam_dodge_missing_reason",
    "evasion_override_active", "death_reason", "action_source",
]

PERTURBATION_EVAL_COLUMNS = [
    "perturbation_level", "altitude_delta", "lon_delta", "lat_delta",
    "heading_delta", "velocity_delta", "episodes", "win_rate",
    "avg_cumulative_team_reward", "std_cumulative_team_reward",
    "mav_survival_rate", "red_missile_hits_mean", "blue_dead_mean",
    "availability",
]

ATTENTION_METRICS_COLUMNS = [
    "method_name", "scenario", "episode_id", "agent_id", "attention_entropy",
    "attention_top1_entity", "attention_top1_weight", "masked_enemy_count",
    "masked_ally_count", "availability",
]

EPISODE_REWARD_COMPONENTS_COLUMNS = [
    "run_id", "scenario", "episode_id", "agent_id", "role", "team",
    "episode_length", "episode_return",
    "tam_v7_total_sum", "tam_v7_flight_sum", "tam_v7_event_sum",
    "tam_v7_terminal_sum", "tam_v7_uav_flight_sum",
    "tam_v7_uav_situation_sum", "tam_v7_uav_event_sum",
    "tam_v7_uav_terminal_sum", "tam_v7_uav_total_sum",
    "tam_v7_uav_altitude_sum", "tam_v7_uav_speed_sum",
    "tam_v7_uav_boundary_sum", "tam_v7_uav_first_out_of_zone_sum",
    "tam_v7_uav_kill_sum", "tam_v7_uav_death_sum",
    "tam_v7_mav_flight_sum", "tam_v7_mav_safety_sum",
    "tam_v7_mav_support_sum", "tam_v7_mav_event_sum",
    "tam_v7_mav_terminal_sum", "tam_v7_mav_total_sum",
    "tam_v7_mav_altitude_sum", "tam_v7_mav_speed_sum",
    "tam_v7_mav_boundary_sum", "tam_v7_mav_death_sum",
    "tam_v7_mav_team_credit_delta_sum",
    "tam_v7_mav_team_credit_used_max",
    "tam_v7_shared_track_usage_log_sum",
    "tam_v7_red_fire_with_mav_track_log_sum",
    "tam_v7_red_hit_with_mav_track_log_sum",
    "tam_v7_mav_pitch_sum", "tam_v7_mav_roll_sum",
    "tam_v7_mav_safety_raw_sum", "tam_v7_mav_safety_dist_sum",
    "tam_v7_mav_safety_threat_sum", "tam_v7_mav_safety_aspect_sum",
    "tam_v7_mav_support_raw_sum", "tam_v7_mav_support_pos_sum",
    "tam_v7_mav_support_aware_sum",
    "tam_v7_terminal_per_agent_sum",
    "tam_v7_uav_pitch_sum", "tam_v7_uav_roll_sum",
    "tam_v7_uav_own_adv_mean_sum", "tam_v7_uav_enemy_threat_mean_sum",
    "tam_v7_uav_distance_ref_m_sum", "tam_v7_uav_situation_raw_sum",
    "tam_v7_blue_loss_frac_last", "tam_v7_red_loss_weighted_last",
    "brma_role_no_missile_total_sum", "brma_role_removed_situation_sum",
    "brma_role_situation_active_sum", "brma_role_is_mav_last",
    "paper_v1_uav_flight_sum", "paper_v1_uav_adv_sum",
    "paper_v1_uav_end_sum", "paper_v1_uav_total_sum",
    "paper_v1_mav_flight_sum", "paper_v1_mav_safety_sum",
    "paper_v1_mav_support_sum", "paper_v1_mav_event_raw_sum",
    "paper_v1_mav_scaled_tam_sum", "paper_v1_mav_total_sum",
    "paper_v1_mav_removed_r_adv_sum", "paper_v1_mav_removed_r_end_sum",
    "paper_v1_mav_shared_track_log_sum",
    "paper_v1_red_launch_with_mav_shared_track_log_sum",
    "paper_v1_red_hit_with_mav_shared_track_log_sum",
    "tam_table1_uav_height_sum", "tam_table1_uav_height_pv_sum",
    "tam_table1_uav_height_ph_sum", "tam_table1_uav_speed_sum",
    "tam_table1_uav_angle_sum", "tam_table1_uav_distance_sum",
    "tam_table1_uav_dodge_sum", "tam_table1_uav_dodge_angle_sum",
    "tam_table1_uav_dodge_speed_sum", "tam_table1_uav_event_sum",
    "tam_table1_uav_kill_sum", "tam_table1_uav_death_sum",
    "tam_table1_uav_out_of_zone_sum", "tam_table1_uav_total_sum",
    "tam_table1_uav_target_id_log_last",
    "tam_table1_uav_target_distance_km_sum",
    "tam_table1_uav_target_ata_rad_sum",
    "tam_table1_uav_target_aa_rad_sum",
    "tam_table1_uav_missing_dodge_geometry_sum",
    "tam_table1_uav_brma_adv_log_sum",
    "tam_table1_uav_brma_end_log_sum",
    "tam_table1_uav_launch_with_mav_shared_track_log_sum",
    "tam_table1_uav_hit_with_mav_shared_track_log_sum",
    "tam_table1_mav_safety_sum", "tam_table1_mav_dist_sum",
    "tam_table1_mav_threat_sum", "tam_table1_mav_aspect_sum",
    "tam_table1_mav_support_sum", "tam_table1_mav_pos_sum",
    "tam_table1_mav_aware_sum", "tam_table1_mav_event_sum",
    "tam_table1_mav_death_sum",
    "tam_table1_mav_team_credit_delta_sum",
    "tam_table1_mav_team_credit_used_sum",
    "tam_table1_mav_team_credit_cap_sum",
    "tam_table1_mav_total_sum",
    "tam_table1_mav_support_anchor_x_sum",
    "tam_table1_mav_support_anchor_y_sum",
    "tam_table1_mav_support_distance_m_sum",
    "tam_table1_mav_observed_count_sum",
    "tam_table1_mav_removed_brma_adv_log_sum",
    "tam_table1_mav_removed_brma_end_log_sum",
    "tam_table1_mav_shared_track_slots_log_sum",
    "tam_table1_red_launch_with_mav_shared_track_log_sum",
    "tam_table1_red_hit_with_mav_shared_track_log_sum",
    "tam_table1_total_sum",
    "v1_mav_safety_sum", "v1_mav_support_sum", "v1_mav_event_sum",
    "v1_mav_total_sum", "v1_mav_flight_base_sum",
    "v1_mav_removed_r_adv_sum", "v1_mav_removed_r_end_sum",
    "v1_mav_safety_dist_sum", "v1_mav_safety_threat_sum",
    "v1_mav_safety_aspect_sum", "v1_mav_safety_danger_m_sum",
    "v1_mav_safety_safe_m_sum",
    "v1_mav_blue_launch_window_on_mav_log_sum",
    "v1_mav_support_pos_sum", "v1_mav_support_pos_active_sum",
    "v1_mav_support_aware_sum", "v1_mav_support_observed_count_sum",
    "v1_mav_support_aware_raw_sum",
    "v1_mav_event_death_sum", "v1_mav_event_team_credit_delta_sum",
    "v1_mav_event_team_credit_used_sum",
    "v1_mav_event_team_credit_cap_sum",
    "v1_mav_removed_v0_overlay_sum", "v1_mav_total_pre_clip_sum",
    "mav_observed_ratio", "mav_shared_track_ratio",
    "mav_observed_ratio_sum", "mav_shared_track_ratio_sum",
    "red_launch_with_mav_shared_track_sum",
    "red_hit_with_mav_shared_track_sum",
    "team_kill_while_mav_alive_sum", "team_kill_after_mav_death_sum",
    "red_launch_before_mav_death_sum",
    "red_launch_after_mav_death_sum",
    "red_uav_alive_steps_before_mav_death_sum",
    "red_uav_alive_steps_after_mav_death_sum",
    "red_launch_rate_before_mav_death",
    "red_launch_rate_after_mav_death",
    "red_launch_rate_before_mav_death_sum",
    "red_launch_rate_after_mav_death_sum",
    "mav_reward_safety_sum", "mav_reward_support_sum",
    "mav_reward_event_sum", "mav_reward_total_sum",
    "mav_removed_r_adv_sum", "mav_removed_r_end_sum",
    "red_launch_count", "red_hit_count",
    "blue_launch_count", "blue_hit_count",
    "mav_alive_final", "red_alive_final", "blue_alive_final",
    "outcome", "end_reason",
    "env_idx", "episode_uid",
]

LAUNCH_GATE_DIAGNOSTICS_COLUMNS = [
    "run_id", "scenario", "env_idx", "episode_id", "episode_uid",
    "step", "sim_time", "agent_id", "alive_before", "alive_after",
    "ammo_remaining", "cooldown_remaining", "cooldown_ok",
    "alive_target_count",
    "track_pass_count", "range_pass_count", "ata_pass_count",
    "ta_pass_count", "line_pass_count", "geometry_pass_count",
    "deconfliction_pass_count",
    "lock_target_id", "lock_candidate_target_id",
    "lock_timer_frames", "lock_required_frames", "lock_mature",
    "launch_executed", "launch_target_id",
    "nearest_alive_target_distance_m", "nearest_track_target_distance_m",
    "best_geometry_target_id", "best_geometry_range_m",
    "best_geometry_ata_rad", "best_geometry_ta_rad",
    "primary_block_reason",
]

_BRMA_TAM_EPISODE_NON_SUM_FIELDS = {
    "tam_dodge_missing_reason",
    "max_altitude_m",
    "above_altitude_max_episode_flag",
    "reward_target_valid",
    "effective_launch_min_range_m",
    "effective_launch_max_range_m",
    "evasion_override_env_steps",
    "above_altitude_max_env_steps",
    "env_idx",
    "episode_uid",
}
for _col in BRMA_TAM_SCRIPTED_COMPONENT_COLUMNS:
    if _col in _BRMA_TAM_EPISODE_NON_SUM_FIELDS:
        continue
    _sum_col = f"{_col}_sum"
    if _sum_col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_sum_col)
for _col in ("max_altitude_m", "above_altitude_max_episode_flag"):
    if _col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_col)

_SCALE_V1_EPISODE_LAST_FIELDS = {
    "scale_v1_progress_reset_reason", "scale_v1_reward_target_id",
    "scale_v1_reward_target_distance_m", "scale_v1_reward_target_valid",
    "scale_v1_geometry_valid", "scale_v1_reward_target_switch_count",
    "scale_v1_mav_team_credit_used", "scale_v1_blue_loss_fraction",
    "scale_v1_red_loss_fraction", "scale_v1_terminal_applied",
}
for _col in BRMA_TAM_SCALE_V1_COMPONENT_COLUMNS:
    _episode_col = f"{_col}_last" if _col in _SCALE_V1_EPISODE_LAST_FIELDS else f"{_col}_sum"
    if _episode_col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_episode_col)
for _col in ("v4_final_j_combat", "v4_max_abs_identity_error"):
    if _col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_col)
from uav_env.JSBSim.envs.role_situation_v3 import V3_EPISODE_FIELDS
for _col in V3_EPISODE_FIELDS:
    if _col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_col)
for _col in V4_COMPONENT_FIELDS:
    _episode_col = f"{_col}_sum"
    if _episode_col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_episode_col)
for _col in V5_COMPONENT_FIELDS:
    if _col == "identity_error":
        _episode_col = "identity_error_max_abs"
    elif _col in V5_EPISODE_LAST_FIELDS or _col in V5_EPISODE_STRING_FIELDS:
        _episode_col = f"{_col}_last"
    else:
        _episode_col = f"{_col}_sum"
    if _episode_col not in EPISODE_REWARD_COMPONENTS_COLUMNS:
        EPISODE_REWARD_COMPONENTS_COLUMNS.append(_episode_col)

FILE_SCHEMAS = {
    "train_metrics.csv": TRAIN_METRICS_COLUMNS,
    "eval_episode_metrics.csv": EVAL_EPISODE_COLUMNS,
    "eval_summary_metrics.csv": EVAL_SUMMARY_COLUMNS,
    "aircraft_timeseries.csv": AIRCRAFT_TIMESERIES_COLUMNS,
    "missile_events.csv": MISSILE_EVENTS_COLUMNS,
    "missile_timeseries.csv": MISSILE_TIMESERIES_COLUMNS,
    "reward_components.csv": REWARD_COMPONENT_COLUMNS,
    "episode_reward_components.csv": EPISODE_REWARD_COMPONENTS_COLUMNS,
    "reward_target_diagnostics.csv": REWARD_TARGET_DIAGNOSTICS_COLUMNS,
    "perturbation_eval_summary.csv": PERTURBATION_EVAL_COLUMNS,
    "attention_metrics.csv": ATTENTION_METRICS_COLUMNS,
    "launch_gate_diagnostics.csv": LAUNCH_GATE_DIAGNOSTICS_COLUMNS,
}

FIELD_DESCRIPTIONS = {
    "relative_win_ratio": "red_win_rate / max(blue_win_rate, epsilon)",
    "kill_death_ratio": "blue_dead_mean / max(red_dead_mean, epsilon)",
    "attention_metrics.csv": "not_available rows are valid when no attention module is implemented",
}


def ensure_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)


def ensure_schema_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, columns in FILE_SCHEMAS.items():
        ensure_csv(directory / filename, columns)
