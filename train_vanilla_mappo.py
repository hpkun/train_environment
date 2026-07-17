"""
train_vanilla_mappo.py —— 纯 MLP MAPPO 基线训练脚本

剥离了论文的 EntityObservationEncoder 和 MaskVectorGenerator，
使用展平观测 → GRU → MLP 的最简架构，仅保留 PPO Clip + MSE + Entropy。

用途：验证 my_uav_env 环境连通性 (物理 / 奖励 / 开火 / 终止)。
用法：python train_vanilla_mappo.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import sys
from dataclasses import asdict

# ---- 多进程性能：禁止底层库的线程池竞争 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Allow Intel + LLVM OpenMP runtimes to coexist (JSBSim vs numpy/torch).
# Without this, ``FGFDMExec()`` aborts the process with OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter, deque
import multiprocessing as mp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    sys.modules.setdefault("train_vanilla_mappo", sys.modules[__name__])

from my_uav_env.alignment.reward_utils import (
    AltitudeRewardConfig,
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
)
from configs.paper_3v3_spec import (
    PAPER_BLUE_POLICY_PROFILE,
    PAPER_CHECKPOINT_SCHEMA,
    PAPER_ENVIRONMENT_PROFILE,
    PAPER_MISSILE_GUIDANCE_MODE,
    PAPER_PID_PROFILE,
    PAPER_REWARD_MODE,
    PID_THROTTLE_BASE,
    paper_environment_snapshot,
)
from my_uav_env.pid_controller import PAPER_PID_ERROR_DEFINITION
from my_uav_env.pid_controller import PAPER_PID_DERIVATIVE_SEMANTICS

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass

# NB: UavCombatEnv is imported inside _worker() lazily.
# JSBSim C++ banners are *not* suppressed — suppress_jsbsim_output defaults to False.
# Tracer files (_jsbsim_tracer_*.txt, _worker_reset_tracer_*.txt) are left behind
# if a worker crashes during JSBSim construction, aiding diagnosis.

# ==============================================================================
#  配置 (2v2 快速验证)
# ==============================================================================
# class Config:
#     # ---- 环境 (对标论文 6v6) ----
#     num_envs: int = 8           # Reduced from 32 to avoid JSBSim C++ resource exhaustion
#     num_red: int = 6            # 6v6 训练场景
#     num_blue: int = 6
#     max_episode_length: int = 1400  # 论文一致
#     action_dim: int = 3

#     # ---- PPO (适配大规模数据) ----
#     replay_buffer_size: int = 500   # 500 steps / 8 envs = 62.5 rollout steps
#     n_update_epochs: int = 10       # 保持不变
#     n_minibatches: int = 4          # 保持不变
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     clip_epsilon: float = 0.2
#     max_grad_norm: float = 5.0      # 硬件防爆器，保持不变

#     # ---- 学习率 (对标论文求稳) ----
#     actor_lr: float = 0.0002        # 论文原版 Actor learning rate
#     critic_lr: float = 0.0005       # 论文原版 Critic learning rate

#     # ---- 损失系数 (高探索度) ----
#     entropy_coef: float = 0.05      # 论文原版 Entropy loss coefficient

#     # ---- 网络 ----
#     mlp_hidden: int = 128           # 论文一致
#     rnn_hidden_size: int = 128      # 论文一致

#     # ---- 训练总量 ----
#     total_env_steps: int = 10_000_000  # 1000 万步 (1e7)
class Config:
    # ---- 环境 (从局部冲突开始，降低协同难度) ----
    num_envs: int = 8           
    num_red: int = 3
    num_blue: int = 3
    max_episode_length: int = 1400
    enable_blue_gcas: bool = False
    obs_mode: str = "paper_strict"
    obs_normalization: str = "paper_fixed_v1"
    pid_profile: str = PAPER_PID_PROFILE
    pid_throttle_base: float = PID_THROTTLE_BASE
    reward_mode: str = PAPER_REWARD_MODE
    missile_guidance_mode: str = PAPER_MISSILE_GUIDANCE_MODE
    altitude_reward_config = DEFAULT_ALTITUDE_REWARD_CONFIG
    resume_from_best: bool = False
    action_dim: int = 3
    algorithm_type: str = "mappo_mlp"
    environment_version: str = PAPER_ENVIRONMENT_PROFILE
    environment_profile: str = PAPER_ENVIRONMENT_PROFILE
    blue_policy_profile: str = PAPER_BLUE_POLICY_PROFILE
    initial_condition_randomization_mode: str = "deterministic_v1"

    # ---- PPO (论文 Table 3) ----
    replay_buffer_size: int = 2000  
    n_update_epochs: int = 10       
    n_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 5.0      

    # ---- 学习率 (论文 Table 3) ----
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4

    # ---- 损失系数 (论文 Table 3: constant 0.05, no decay) ----
    entropy_coef: float = 0.05      

    # ---- 网络 (论文 Table 3: [128, 128]) ----
    mlp_hidden: int = 128
    rnn_hidden_size: int = 128      

    # ---- 训练总量 ----
    total_env_steps: int = 10_000_000

    # ---- Runtime / persistence ----
    log_file: str = "vanilla_training_log.csv"
    results_file: str = "results/vanilla_mappo_results.csv"
    launch_quality_file: str | None = None
    extreme_load_trace_file: str | None = None
    checkpoint_dir: str = "checkpoints"
    resume_latest: bool = False
    resume_state: str | None = None
    overwrite_existing: bool = False
    eval_during_training: bool = False
    eval_interval_steps: int = 50_000
    eval_episodes: int = 20
    eval_log_file: str | None = None
    seed = None
    device: str = "auto"


# ==============================================================================
#  纯 MLP Actor / Critic (展平观测 → GRU → MLP)
# ==============================================================================

def _include_aux_obs_default(obs_mode: str, include_aux_obs: bool | None = None) -> bool:
    if include_aux_obs is not None:
        return bool(include_aux_obs)
    return obs_mode != "paper_strict"


def _compute_obs_dim(num_red: int, num_blue: int, is_red: bool,
                     obs_mode: str = "paper_strict",
                     include_aux_obs: bool | None = None) -> int:
    """Return the fixed Table 1/Table 2 actor input dimension."""
    if obs_mode != "paper_strict" or include_aux_obs:
        raise ValueError("paper_3v3_v1 only supports strict entity observations")
    if is_red:
        n_ally = num_red - 1
        n_enemy = num_blue
    else:
        n_ally = num_blue - 1
        n_enemy = num_red
    total_entities = 1 + max(n_ally, 0) + n_enemy
    return 10 * total_entities


def _compute_global_state_dim(num_red: int, obs_mode: str = "paper_strict") -> int:
    """Paper CTDE state: native ego state of every red UAV."""
    if obs_mode != "paper_strict":
        raise ValueError("paper_3v3_v1 only supports paper_strict")
    return num_red * 10


def _global_state_from_local_obs_flats(
    local_obs_flats: list[np.ndarray],
    obs_mode: str = "paper_strict",
) -> np.ndarray:
    """Extract and concatenate each red agent's leading ego-state entity."""
    if obs_mode != "paper_strict":
        raise ValueError("paper_3v3_v1 only supports paper_strict")
    entity_dim = 10
    return np.concatenate([
        np.asarray(obs, dtype=np.float32).reshape(-1)[:entity_dim]
        for obs in local_obs_flats
    ]).astype(np.float32)


ACTION_STD_MIN = 0.05
ACTION_STD_MAX = 0.6
ACTION_STD_INIT = 0.3
ACTION_LOG_STD_INIT = float(np.log(ACTION_STD_INIT))


def _initial_log_std_head_bias() -> float:
    lower = np.log(ACTION_STD_MIN)
    upper = np.log(ACTION_STD_MAX)
    normalized = 2.0 * (ACTION_LOG_STD_INIT - lower) / (upper - lower) - 1.0
    return float(np.arctanh(np.clip(normalized, -0.999999, 0.999999)))


class SquashedNormal:
    """Tanh-squashed diagonal Gaussian with stable inverse likelihood."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        self.loc = loc
        self.scale = scale
        self.eps = float(eps)
        self.base_dist = torch.distributions.Normal(loc, scale)

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.rsample(sample_shape))

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        bounded = action.clamp(-1.0 + self.eps, 1.0 - self.eps)
        latent = torch.atanh(bounded)
        log_det = 2.0 * (
            np.log(2.0) - latent - F.softplus(-2.0 * latent))
        return self.base_dist.log_prob(latent) - log_det

    def base_entropy(self) -> torch.Tensor:
        return self.base_dist.entropy()


class VanillaActor(nn.Module):
    """Two-layer MLP -> GRU -> tanh-squashed diagonal Gaussian."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden: int = 128, rnn_hidden: int = 128):
        super().__init__()
        self.fc_in = nn.Linear(obs_dim, hidden)
        self.fc_hidden = nn.Linear(hidden, hidden)
        self.rnn = nn.GRUCell(hidden, rnn_hidden)
        self.action_head = nn.Linear(rnn_hidden, action_dim)
        self.action_log_std_head = nn.Linear(rnn_hidden, action_dim)
        nn.init.zeros_(self.action_log_std_head.weight)
        nn.init.constant_(
            self.action_log_std_head.bias, _initial_log_std_head_bias())

    def forward(self, obs_flat: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        Args:
            obs_flat:   (B, obs_dim)  展平观测
            rnn_hidden: (B, rnn_hidden)  GRU 隐藏状态
        Returns:
            action_dist: SquashedNormal
            rnn_hidden:  (B, rnn_hidden)
        """
        x = F.relu(self.fc_in(obs_flat))
        x = F.relu(self.fc_hidden(x))
        rnn_hidden_new = self.rnn(x, rnn_hidden)  # (B, rnn_hidden)
        action_mean = self.action_head(rnn_hidden_new)
        raw_log_std = self.action_log_std_head(rnn_hidden_new)
        log_std_min = float(np.log(ACTION_STD_MIN))
        log_std_max = float(np.log(ACTION_STD_MAX))
        log_std = log_std_min + 0.5 * (torch.tanh(raw_log_std) + 1.0) * (
            log_std_max - log_std_min)
        sigma = torch.exp(log_std)
        return SquashedNormal(action_mean, sigma), rnn_hidden_new


class CentralizedCritic(nn.Module):
    """MAPPO centralized critic: all red native ego states → V(s).

    Paper §3.4 describes the global state as the native states of all friendly
    UAVs. It therefore excludes repeated enemy portions of local observations.
    """

    def __init__(self, global_obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor):
        """global_obs: (B, global_obs_dim) → value: (B, 1)"""
        return self.net(global_obs)


# ==============================================================================
#  观测展平工具
# ==============================================================================
PAPER_EGO_SCALE_V1 = np.array([
    40000.0, 40000.0, 10000.0, 600.0,
    np.pi, np.pi, np.pi, np.pi, np.pi, 600.0,
], dtype=np.float32)
PAPER_RELATIVE_SCALE_V1 = np.array([
    40000.0, 40000.0, 10000.0, np.pi, np.pi,
    600.0, np.pi, np.pi, np.pi, 100000.0,
], dtype=np.float32)


def _normalize_paper_fixed_v1(obs_np: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego = np.asarray(obs_np["ego_state"], dtype=np.float32) / PAPER_EGO_SCALE_V1
    allies = (np.asarray(obs_np["ally_states"], dtype=np.float32)
              / PAPER_RELATIVE_SCALE_V1)
    enemies = (np.asarray(obs_np["enemy_states"], dtype=np.float32)
               / PAPER_RELATIVE_SCALE_V1)
    return ego, allies, enemies


def _flatten_obs(obs_np: dict, obs_mode: str = "paper_strict",
                 include_aux_obs: bool | None = None,
                 obs_normalization: str = "paper_fixed_v1") -> np.ndarray:
    """Flatten only the six Table 1/Table 2 entities (6 x 10)."""
    if obs_mode != "paper_strict" or include_aux_obs:
        raise ValueError("paper_3v3_v1 excludes auxiliary and legacy observations")
    if obs_normalization not in ("paper_fixed_v1", "none"):
        raise ValueError("obs_normalization must be 'paper_fixed_v1' or 'none'")
    if obs_normalization == "paper_fixed_v1":
        ego, allies, enemies = _normalize_paper_fixed_v1(obs_np)
    else:
        ego = obs_np["ego_state"]
        allies = obs_np["ally_states"]
        enemies = obs_np["enemy_states"]
    parts = [
        np.asarray(ego, dtype=np.float32).ravel(),
        np.asarray(allies, dtype=np.float32).ravel(),
        np.asarray(enemies, dtype=np.float32).ravel(),
    ]
    return np.concatenate(parts)


LAUNCH_DIAG_BASE_KEYS = (
    "range_ok_pairs",
    "ao_ok_pairs",
    "ta_ok_pairs",
    "geometry_ok_pairs",
    "lock_mature_pairs",
    "cooldown_blocked",
    "engaged_blocked",
    "range_low_blocked",
    "range_high_blocked",
    "launches",
)

LAUNCH_DIAG_CSV_FIELDS = (
    "LaunchDiagRedGeometryOk",
    "LaunchDiagBlueGeometryOk",
    "LaunchDiagRedLaunches",
    "LaunchDiagBlueLaunches",
    "LaunchDiagRedRangeOk",
    "LaunchDiagRedAoOk",
    "LaunchDiagRedTaOk",
    "LaunchDiagBlueRangeOk",
    "LaunchDiagBlueAoOk",
    "LaunchDiagBlueTaOk",
    "LaunchDiagRedEngagedBlocked",
    "LaunchDiagBlueEngagedBlocked",
    "LaunchDiagRedCooldownBlocked",
    "LaunchDiagBlueCooldownBlocked",
    "LaunchDiagRedLockMature",
    "LaunchDiagBlueLockMature",
    "RedGeometryToLaunchRate",
    "BlueGeometryToLaunchRate",
    "RedRangeToGeometryRate",
    "BlueRangeToGeometryRate",
    "LaunchDiagRedRangeLowBlocked",
    "LaunchDiagBlueRangeLowBlocked",
    "LaunchDiagRedRangeHighBlocked",
    "LaunchDiagBlueRangeHighBlocked",
)

LOCAL_REWARD_COMPONENT_KEYS = (
    "r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv",
)
EPISODE_REWARD_COMPONENT_KEYS = (*LOCAL_REWARD_COMPONENT_KEYS, "r_end", "r_death")
REWARD_COMPONENT_LOG_FIELDS = tuple(
    field
    for key in LOCAL_REWARD_COMPONENT_KEYS
    for field in (f"{key}_team_sum", f"{key}_per_agent_mean")
) + ("r_end_team",)

LAUNCH_QUALITY_DETAIL_FIELDS = (
    "team",
    "shooter_id",
    "target_id",
    "missile_id",
    "current_step",
    "physics_frame",
    "range_m",
    "AO_rad",
    "AO_deg",
    "TA_rad",
    "TA_deg",
    "relative_distance_3d_m",
    "horizontal_range_m",
    "altitude_diff_m",
    "shooter_speed_mps",
    "target_speed_mps",
    "closing_speed_mps",
    "shooter_alt_m",
    "target_alt_m",
    "target_alive_at_launch",
    "raw_termination_reason",
    "termination_reason",
    "is_success",
    "flight_time_sec",
    "launch_step",
    "termination_step",
    "step_delta",
    "target_alive_at_termination",
    "pn_guidance_frames",
    "pn_nonzero_command_frames",
    "maximum_command_g",
)

LAUNCH_QUALITY_AGG_CSV_FIELDS = (
    "RedLaunchRangeMean",
    "RedLaunchRangeP25",
    "RedLaunchRangeP50",
    "RedLaunchRangeP75",
    "RedLaunchAoDegMean",
    "RedLaunchAoDegP50",
    "RedLaunchTaDegMean",
    "RedLaunchTaDegP50",
    "RedLaunchClosingSpeedMean",
    "RedLaunchAltitudeDiffAbsMean",
    "RedLaunchHitRateFromQualityRecords",
    "BlueLaunchRangeMean",
    "BlueLaunchRangeP25",
    "BlueLaunchRangeP50",
    "BlueLaunchRangeP75",
    "BlueLaunchAoDegMean",
    "BlueLaunchAoDegP50",
    "BlueLaunchTaDegMean",
    "BlueLaunchTaDegP50",
    "BlueLaunchClosingSpeedMean",
    "BlueLaunchAltitudeDiffAbsMean",
    "BlueLaunchHitRateFromQualityRecords",
)

ACTION_BOUND_CSV_FIELDS = (
    "ExecutedActionAbsMean",
    "ExecutedActionNearBoundFrac",
    "ExecutedActionNearBoundFracPitch",
    "ExecutedActionNearBoundFracHeading",
    "ExecutedActionNearBoundFracVelocity",
    "PolicyMeanNearBoundFrac",
)

AIRCRAFT_ENVELOPE_CSV_FIELDS = (
    "MaximumSpeedBeforeLimiterMps",
    "MaximumSpeedAfterLimiterMps",
    "SpeedLimiterActivations",
    "SpeedLimiterActivationRatePer1000PhysicsSteps",
    "MaximumLoadG",
    "LoadLimiterActivations",
    "EnvironmentDynamicsWarning",
)

ROLLOUT_LAYOUT_CSV_FIELDS = (
    "requested_replay_buffer_size",
    "rollout_horizon_per_env",
    "transitions_per_update",
    "unused_replay_slots",
)

PPO_DIAG_CSV_FIELDS = (
    "PolicyLoss",
    "EntropyBonus",
    "ActorUpdateAttempts",
    "ActorUpdatesApplied",
    "ActorUpdatesSkipped",
    "CriticUpdateAttempts",
    "CriticUpdatesApplied",
    "CriticUpdatesSkipped",
    "ActionStdDeltaFromInit",
    "ActionStdGrowthRatio",
)

BLUE_POLICY_DIAG_CSV_FIELDS = (
    "blue_target_switches_total",
    "blue_target_dead_switches",
    "blue_distance_triggered_switches",
    "blue_engaged_triggered_switches",
    "blue_mws_detected_agent_decisions",
    "blue_mws_override_agent_decisions",
    "blue_route_phase_changes",
    "blue_base_heading_command_discontinuities",
    "blue_executed_heading_command_discontinuities",
    "blue_altitude_recovery_frames",
    "blue_target_reallocations",
    "blue_target_reallocations_after_death",
    "blue_target_switches_while_alive",
    "blue_engaged_wait_agent_decisions",
    "blue_no_alive_target_agent_decisions",
)

LEARNABILITY_DIAG_CSV_FIELDS = (
    "RedMissileTermHit", "RedMissileTermPHitFail",
    "RedMissileTermOvershoot", "RedMissileTermTimeout",
    "RedMissileTermTargetDead", "BlueMissileTermHit",
    "BlueMissileTermPHitFail", "BlueMissileTermOvershoot",
    "BlueMissileTermTimeout", "BlueMissileTermTargetDead",
    "MissileLifetimeMeanS", "MissileLifetimeP50S",
    "MissileOneFrameTerminations", "MissileLifetimeOverPoint2S",
    "MissilePNGuidanceFrames", "MissilePNNonzeroCommandFrames",
    "MissileMaximumCommandG", "TargetReallocations",
    "TargetReallocationsAfterDeath", "TargetSwitchesWhileAlive",
    "TargetEngagedWaitFrames", "TargetNoAliveFrames",
    "RedMWSDetectedAgentDecisions", "RedMWSOverrideAgentDecisions",
    "BlueMWSDetectedAgentDecisions", "BlueMWSOverrideAgentDecisions",
    "RedWarningToTerminalMeanS", "RedWarningToTerminalP50S",
    "RedWarningToHitMeanS", "BlueWarningToTerminalMeanS",
    "BlueWarningToTerminalP50S", "BlueWarningToHitMeanS",
    "CompletedEpisodesThisIteration", "NoCompletedEpisodeThisIteration",
    "IterationWallTimeS", "EnvironmentStepsPerSecond",
    "RedMissileLaunches", "RedMissilePHitFail",
    "RedMissileOvershoot", "RedMissileTimeout", "RedMissileTargetDead",
    "RedMissileUnknownTermination", "BlueMissileLaunches",
    "BlueMissilePHitFail", "BlueMissileOvershoot",
    "BlueMissileTimeout", "BlueMissileTargetDead",
    "BlueMissileUnknownTermination", "RedMissileLifetimeMeanSec",
    "RedMissileLifetimeMedianSec", "RedMissileLifetimeP90Sec",
    "RedMissileLifetimeMinSec", "RedMissileLifetimeMaxSec",
    "BlueMissileLifetimeMeanSec", "BlueMissileLifetimeMedianSec",
    "BlueMissileLifetimeP90Sec", "BlueMissileLifetimeMinSec",
    "BlueMissileLifetimeMaxSec", "RedLaunchInsideHitRadiusCount",
    "RedLaunchInsideHitRadiusFrac", "BlueLaunchInsideHitRadiusCount",
    "BlueLaunchInsideHitRadiusFrac", "RedOnePhysicsFrameHitCount",
    "RedOnePhysicsFrameHitFrac", "BlueOnePhysicsFrameHitCount",
    "BlueOnePhysicsFrameHitFrac", "RedMissileSurvivedOneDecisionCount",
    "RedMissileSurvivedOneDecisionFrac",
    "BlueMissileSurvivedOneDecisionCount",
    "BlueMissileSurvivedOneDecisionFrac", "RedPNFramesMean",
    "RedPNNonZeroCommandFrac", "BluePNFramesMean",
    "BluePNNonZeroCommandFrac", "RedFirstLaunchStepMean",
    "BlueFirstLaunchStepMean", "RedFirstHitStepMean",
    "BlueFirstHitStepMean", "EpisodeLengthMean",
    "EpisodeLengthP50", "EpisodeLengthP90", "EstimatedRemainingTimeSec",
    "WorkerRestartCount", "ResumeCount",
    "RedMaximumGSeen", "BlueMaximumGSeen",
    "RedFramesAbove9G", "BlueFramesAbove9G",
    "RedMaximumConsecutiveAbove9GFrames",
    "BlueMaximumConsecutiveAbove9GFrames",
    "RedEpisodeEverExceeded9G", "BlueEpisodeEverExceeded9G",
    "RedTransientAbove30GEvents", "BlueTransientAbove30GEvents",
    "RedMaximumConsecutiveAbove30GFrames",
    "BlueMaximumConsecutiveAbove30GFrames",
    "RedLoadProtectionActiveFrames", "BlueLoadProtectionActiveFrames",
    "RedMWSWarningGenerations", "BlueMWSWarningGenerations",
    "RedMWSDirectionChangesWithinSameMissile",
    "RedMWSSuppressedDirectionFlipAttempts",
    "BlueMWSDirectionChangesWithinSameMissile",
    "RedSetpointRateLimitActivations", "BlueSetpointRateLimitActivations",
    "RedRequestedHeadingJumpMaxDeg", "BlueRequestedHeadingJumpMaxDeg",
    "RedAppliedHeadingJumpMaxDeg", "BlueAppliedHeadingJumpMaxDeg",
    "RedRequestedPitchJumpMaxDeg", "BlueRequestedPitchJumpMaxDeg",
    "RedAppliedPitchJumpMaxDeg", "BlueAppliedPitchJumpMaxDeg",
    "RedMaximumAbsoluteEPhi", "BlueMaximumAbsoluteEPhi",
    "RedMaximumAbsoluteETheta", "BlueMaximumAbsoluteETheta",
    "RedMaximumAbsoluteDerivativeTerm",
    "BlueMaximumAbsoluteDerivativeTerm",
    "RedPIDOutputSaturationFrames", "BluePIDOutputSaturationFrames",
    "RedDegenerateArctanRatioCount", "BlueDegenerateArctanRatioCount",
    "RedMWSMaximumContinuousDecisions", "RedMWSTargetHeadingDeltaMaxDeg",
    "NonFiniteLoadInvalidEpisodes", "CatastrophicFiniteLoadInvalidEpisodes",
    "PersistentExtremeFiniteLoadInvalidEpisodes",
)

CHECKPOINT_SCHEMA_VERSION = PAPER_CHECKPOINT_SCHEMA
TRAINING_STATE_SCHEMA_VERSION = "vanilla_mappo_training_state_v1"
ACTION_DISTRIBUTION_VERSION = "tanh_squashed_diag_gaussian_v1"
ENTROPY_ESTIMATOR_VERSION = "pre_tanh_base_normal_entropy_v1"


def _training_log_fields() -> list[str]:
    return [
        "Iteration", "Step", "ActorLoss", "CriticLoss", "BaseNormalEntropy",
        *PPO_DIAG_CSV_FIELDS, "RedMeanReward", "RedWinRate", "RedRewardStd",
        "WinRateRecent", "RedMissiles", "BlueMissiles",
        *ROLLOUT_LAYOUT_CSV_FIELDS, "Episodes", "InvalidNumericalEpisodes",
        "InvalidTransitionsDropped", "InvalidEpisodesDropped", "UpdateSkipReason",
        "RedWins", "BlueWins", "Draws", "RedAliveMean", "BlueAliveMean",
        "RedDeathsMissile", "RedDeathsCrash", "BlueDeathsMissile",
        "BlueDeathsCrash", "RedMissileHits", "BlueMissileHits",
        "RedMissileHitRate", "BlueMissileHitRate", "KD_Red_AllDeaths",
        "KD_Red_MissileOnly", "RWR", "RWRDenominatorZero", "RewardVersion",
        "RewardMode", "EnvironmentProfile", "ObsNormalization", "PIDProfile",
        "PIDThrottleBase", "MissileGuidanceMode", "CheckpointSchema",
        "ActionDistribution", "EntropyEstimator",
        "EnvironmentConfigFingerprint", "BluePolicyProfile",
        "RedMWSMode", "BlueMWSMode", "NumRed", "NumBlue", "MaxSteps",
        "AltitudeRewardConfigVersion", "AltitudeRewardConfig",
        *REWARD_COMPONENT_LOG_FIELDS, "ActionStdMean", "ActionStdMin",
        "ActionStdMax", "ActionLogStdMean", "StateDependentStdMean",
        "StateDependentStdMin", "StateDependentStdMax",
        "StateDependentStdLowerBoundFrac", "StateDependentStdUpperBoundFrac",
        *LAUNCH_DIAG_CSV_FIELDS, *LAUNCH_QUALITY_AGG_CSV_FIELDS,
        *ACTION_BOUND_CSV_FIELDS, *AIRCRAFT_ENVELOPE_CSV_FIELDS,
        *BLUE_POLICY_DIAG_CSV_FIELDS, *LEARNABILITY_DIAG_CSV_FIELDS,
    ]


def _minimal_altitude_reward_config() -> AltitudeRewardConfig:
    return AltitudeRewardConfig(
        version="eq17_minimal_finite_tail_v1",
        h_min_m=0.0, h_att_m=2000.0, h_adv_m=5000.0,
        h_max_m=10000.0, d_att_max_m=10000.000001,
        high_altitude_tail=0.0)


def _rollout_layout(replay_buffer_size: int, num_envs: int) -> dict:
    requested = int(replay_buffer_size)
    envs = int(num_envs)
    if requested < envs:
        raise ValueError(
            "replay_buffer_size must be >= num_envs for at least one "
            f"rollout step per environment: replay_buffer_size={requested}, "
            f"num_envs={envs}")
    horizon = requested // envs
    transitions = horizon * envs
    return {
        "requested_replay_buffer_size": requested,
        "rollout_horizon_per_env": horizon,
        "transitions_per_update": transitions,
        "unused_replay_slots": requested - transitions,
    }


def _checkpoint_metadata(config, obs_dim: int, global_state_dim: int) -> dict:
    if (config.environment_profile != PAPER_ENVIRONMENT_PROFILE
            or config.num_red != 3 or config.num_blue != 3
            or config.max_episode_length != 1400):
        raise ValueError("formal MAPPO checkpoints require paper_3v3_v1")
    environment_snapshot = paper_environment_snapshot(seed=config.seed)
    rollout_layout = _rollout_layout(config.replay_buffer_size, config.num_envs)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "obs_mode": config.obs_mode,
        "obs_normalization": config.obs_normalization,
        "reward_version": "paper_3v3_joint_eq15_23_v1",
        "reward_mode": config.reward_mode,
        "pid_profile": config.pid_profile,
        "pid_throttle_base": float(config.pid_throttle_base),
        "pid_error_definition": environment_snapshot.get(
            "pid_error_definition", {}).get(
                "value", PAPER_PID_ERROR_DEFINITION),
        "derivative_semantics": environment_snapshot.get(
            "derivative_semantics", {}).get(
                "value", PAPER_PID_DERIVATIVE_SEMANTICS),
        "missile_guidance_mode": config.missile_guidance_mode,
        "missile_hit_radius_m": float(
            environment_snapshot["missile_hit_radius_m"]["value"]),
        "altitude_reward_config": asdict(config.altitude_reward_config),
        "action_distribution": ACTION_DISTRIBUTION_VERSION,
        "entropy_estimator": ENTROPY_ESTIMATOR_VERSION,
        "algorithm_type": str(config.algorithm_type),
        "environment_version": str(config.environment_version),
        "environment_profile": str(config.environment_profile),
        "blue_policy_profile": str(config.blue_policy_profile),
        "initial_condition_randomization_mode": str(
            config.initial_condition_randomization_mode),
        "q_los_version": "observer_velocity_to_target_los_3d_v1",
        "altitude_reward_interpretation": (
            "paper_unspecified_engineering_mean_over_alive_enemies"),
        "num_red": int(config.num_red),
        "num_blue": int(config.num_blue),
        "max_episode_length": int(config.max_episode_length),
        "total_env_steps": int(config.total_env_steps),
        "entropy_coef": float(config.entropy_coef),
        "actor_lr": float(config.actor_lr),
        "critic_lr": float(config.critic_lr),
        "n_update_epochs": int(config.n_update_epochs),
        "n_minibatches": int(config.n_minibatches),
        "gamma": float(config.gamma),
        "gae_lambda": float(config.gae_lambda),
        "clip_epsilon": float(config.clip_epsilon),
        **rollout_layout,
        "global_state_dim": int(global_state_dim),
        "actor_obs_dim": int(obs_dim),
        "actor_hidden_sizes": [int(config.mlp_hidden), int(config.mlp_hidden)],
        "actor_rnn_hidden_size": int(config.rnn_hidden_size),
        "recurrent_n": 1,
        "action_log_std_init": ACTION_LOG_STD_INIT,
        "action_std_init": ACTION_STD_INIT,
        "action_std_bounds": [ACTION_STD_MIN, ACTION_STD_MAX],
        "action_std_parameterization": "gru_state_tanh_bounded_log_std_head_v1",
        "environment_config": environment_snapshot,
        "environment_config_fingerprint": environment_snapshot[
            "environment_config_fingerprint"],
    }
    metadata["profile_provenance_fields"] = {
        key: value["source"]
        for key, value in environment_snapshot.items()
        if isinstance(value, dict) and "source" in value
    }
    return metadata


def _save_model_checkpoint(path: str, model: nn.Module, metadata: dict,
                           model_kind: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
        "model_kind": model_kind,
    }, path)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: dict, path: str) -> None:
    _ensure_parent_dir(path)
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _training_core_config(config, checkpoint_meta: dict) -> dict:
    fields = (
        "num_red", "num_blue", "num_envs", "max_episode_length",
        "replay_buffer_size", "n_update_epochs", "n_minibatches", "gamma",
        "gae_lambda", "clip_epsilon", "max_grad_norm", "actor_lr",
        "critic_lr", "entropy_coef", "mlp_hidden", "rnn_hidden_size",
        "action_dim", "obs_mode", "obs_normalization", "pid_profile",
        "pid_throttle_base", "reward_mode", "missile_guidance_mode",
        "environment_profile", "blue_policy_profile",
        "initial_condition_randomization_mode")
    return {
        **{field: getattr(config, field) for field in fields},
        "environment_config_fingerprint": checkpoint_meta[
            "environment_config_fingerprint"],
        "training_log_schema": _training_log_fields(),
    }


def _build_training_state(
    actor, critic, actor_opt, critic_opt, config, checkpoint_meta: dict,
    runtime: dict,
) -> dict:
    return {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
        "runtime": runtime,
        "rng_state": _capture_rng_state(),
        "checkpoint_metadata": dict(checkpoint_meta),
        "core_config": _training_core_config(config, checkpoint_meta),
        "run_id": runtime.get("run_id"),
        "log_schema_version": CHECKPOINT_SCHEMA_VERSION,
    }


def _validate_training_state(payload: dict, config, checkpoint_meta: dict) -> None:
    if payload.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("training-state schema mismatch")
    saved_metadata = payload.get("checkpoint_metadata", {})
    if saved_metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if saved_metadata.get("action_distribution") != ACTION_DISTRIBUTION_VERSION:
        raise ValueError("checkpoint action distribution mismatch")
    expected = _training_core_config(config, checkpoint_meta)
    if payload.get("core_config") != expected:
        raise ValueError("training-state core configuration mismatch")
    completed_steps = int(payload.get("runtime", {}).get("total_steps", -1))
    if int(config.total_env_steps) < completed_steps:
        raise ValueError(
            "total_env_steps may only stay equal or increase when resuming")


def _validate_resume_csv(path: str, expected_header: list[str], step: int) -> None:
    if not os.path.exists(path):
        raise ValueError(f"resume log does not exist: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != expected_header:
        raise ValueError(f"resume log header mismatch: {path}")
    if len(rows) < 2 or int(rows[-1][expected_header.index("Step")]) != int(step):
        raise ValueError(
            f"resume log last Step does not match checkpoint total_steps: {path}")


def _validate_resume_csv_header(path: str, expected_header: list[str]) -> None:
    if not os.path.exists(path):
        raise ValueError(f"resume log does not exist: {path}")
    with open(path, newline="") as handle:
        header = next(csv.reader(handle), None)
    if header != expected_header:
        raise ValueError(f"resume log header mismatch: {path}")


def _flush_and_periodic_fsync(handle, iteration: int) -> None:
    handle.flush()
    if iteration % 10 == 0:
        os.fsync(handle.fileno())


def _atomic_json_save(payload: dict, path: str) -> None:
    _ensure_parent_dir(path)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_finite(value):
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _append_invalid_trace_jsonl(
        path: str, *, run_id: str, seed, total_step: int, env_index: int,
        episode_info: dict, traces: list[dict]) -> int:
    if not traces:
        return 0
    reasons = list(episode_info.get("invalid_numerical_reasons", []))
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for trace in traces:
            agent_id = str(trace.get("trigger_agent_id", ""))
            matching = [reason for reason in reasons
                        if reason.startswith(f"{agent_id}:")]
            fallback_reason = (
                reasons[0].split(":", 1)[-1] if reasons
                else trace.get("invalid_reason", "numerical_invalid"))
            record = {
                "run_id": run_id,
                "seed": seed,
                "total_step": int(total_step),
                "env_index": int(env_index),
                "episode_step": int(episode_info.get("EpisodeLength", 0)),
                "agent_id": agent_id,
                "team": "blue" if agent_id.startswith("blue") else "red",
                "invalid_reason": (matching[0].split(":", 1)[1]
                                   if matching else fallback_reason),
                "trigger_g": trace.get("trigger_g"),
                "trigger_level": trace.get("trigger_level"),
                "physics_frames": trace.get("frames", []),
            }
            handle.write(json.dumps(
                _json_finite(record), sort_keys=True, separators=(",", ":"))
                + "\n")
            written += 1
        handle.flush()
    return written


_STOP_REQUESTED = False


def _request_safe_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _unpack_and_validate_checkpoint(payload, expected_metadata: dict,
                                    model_kind: str) -> dict:
    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        raise ValueError(
            "checkpoint lacks required environment metadata; legacy raw state_dict "
            "is incompatible with the paper baseline")
    if payload.get("model_kind") != model_kind:
        raise ValueError(
            f"checkpoint model_kind mismatch: expected {model_kind!r}, "
            f"got {payload.get('model_kind')!r}")
    metadata = dict(payload["metadata"])
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, current={expected!r}"
            for key, (actual, expected) in mismatches.items())
        raise ValueError(f"checkpoint environment metadata mismatch: {details}")
    return payload["state_dict"]


def _empty_launch_diag_totals() -> dict:
    return {team: {key: 0 for key in LAUNCH_DIAG_BASE_KEYS}
            for team in ("red", "blue")}


def _accumulate_launch_diag_totals(totals: dict, launch_diag: dict | None) -> None:
    if not isinstance(launch_diag, dict):
        return
    for team in ("red", "blue"):
        team_diag = launch_diag.get(team, {})
        if not isinstance(team_diag, dict):
            continue
        for key in LAUNCH_DIAG_BASE_KEYS:
            totals[team][key] += int(team_diag.get(key, 0))


def _launch_diag_metrics(totals: dict) -> dict:
    red = totals["red"]
    blue = totals["blue"]
    red_geometry = red["geometry_ok_pairs"]
    blue_geometry = blue["geometry_ok_pairs"]
    red_launches = red["launches"]
    blue_launches = blue["launches"]
    return {
        "LaunchDiagRedGeometryOk": red_geometry,
        "LaunchDiagBlueGeometryOk": blue_geometry,
        "LaunchDiagRedLaunches": red_launches,
        "LaunchDiagBlueLaunches": blue_launches,
        "LaunchDiagRedRangeOk": red["range_ok_pairs"],
        "LaunchDiagRedAoOk": red["ao_ok_pairs"],
        "LaunchDiagRedTaOk": red["ta_ok_pairs"],
        "LaunchDiagBlueRangeOk": blue["range_ok_pairs"],
        "LaunchDiagBlueAoOk": blue["ao_ok_pairs"],
        "LaunchDiagBlueTaOk": blue["ta_ok_pairs"],
        "LaunchDiagRedEngagedBlocked": red["engaged_blocked"],
        "LaunchDiagBlueEngagedBlocked": blue["engaged_blocked"],
        "LaunchDiagRedCooldownBlocked": red["cooldown_blocked"],
        "LaunchDiagBlueCooldownBlocked": blue["cooldown_blocked"],
        "LaunchDiagRedLockMature": red["lock_mature_pairs"],
        "LaunchDiagBlueLockMature": blue["lock_mature_pairs"],
        "RedGeometryToLaunchRate": _safe_div(red_launches, red_geometry),
        "BlueGeometryToLaunchRate": _safe_div(blue_launches, blue_geometry),
        "RedRangeToGeometryRate": _safe_div(red_geometry, red["range_ok_pairs"]),
        "BlueRangeToGeometryRate": _safe_div(blue_geometry, blue["range_ok_pairs"]),
        "LaunchDiagRedRangeLowBlocked": red["range_low_blocked"],
        "LaunchDiagBlueRangeLowBlocked": blue["range_low_blocked"],
        "LaunchDiagRedRangeHighBlocked": red["range_high_blocked"],
        "LaunchDiagBlueRangeHighBlocked": blue["range_high_blocked"],
    }


def _learnability_iteration_metrics(
    launch_records: list[dict], done_records: list[dict],
    environment_diag: Counter, episode_lengths: list[int],
    completed_episode_events: list[dict], missile_hit_radius_m: float,
    iter_episodes: int, wall_time_s: float, environment_steps: int,
    remaining_steps: int, worker_restart_count: int, resume_count: int,
) -> dict:
    term = Counter((str(row.get("team", "")),
                    str(row.get("raw_termination_reason", "unknown")))
                   for row in done_records)
    lifetimes = _numeric_values(done_records, "flight_time_sec")
    pn_frames = [int(row.get("pn_guidance_frames", 0)) for row in done_records]
    pn_nonzero = [int(row.get("pn_nonzero_command_frames", 0))
                  for row in done_records]
    max_commands = _numeric_values(done_records, "maximum_command_g")
    result = {}
    for team, prefix in (("red", "Red"), ("blue", "Blue")):
        for reason, suffix in (
                ("hit", "Hit"), ("p_hit_fail", "PHitFail"),
                ("overshoot", "Overshoot"), ("timeout", "Timeout"),
                ("target_dead", "TargetDead")):
            result[f"{prefix}MissileTerm{suffix}"] = int(term[(team, reason)])
        launches = [row for row in launch_records if row.get("team") == team]
        dones = [row for row in done_records if row.get("team") == team]
        lifetimes_team = _numeric_values(dones, "flight_time_sec")
        hits = [row for row in dones if row.get("raw_termination_reason") == "hit"]
        known = {"hit", "p_hit_fail", "overshoot", "timeout", "target_dead"}
        inside = sum(
            float(row.get("range_m", float("inf"))) < missile_hit_radius_m
                     for row in launches)
        one_frame_hits = sum(
            float(row.get("flight_time_sec", float("inf"))) <= 1.0 / 60.0 + 1e-9
            for row in hits)
        survived = sum(value > 0.2 for value in lifetimes_team)
        team_pn_frames = [int(row.get("pn_guidance_frames", 0)) for row in dones]
        team_pn_nonzero = [
            int(row.get("pn_nonzero_command_frames", 0)) for row in dones]
        result.update({
            f"{prefix}MissileLaunches": len(launches),
            f"{prefix}MissileHits": len(hits),
            f"{prefix}MissilePHitFail": int(term[(team, "p_hit_fail")]),
            f"{prefix}MissileOvershoot": int(term[(team, "overshoot")]),
            f"{prefix}MissileTimeout": int(term[(team, "timeout")]),
            f"{prefix}MissileTargetDead": int(term[(team, "target_dead")]),
            f"{prefix}MissileUnknownTermination": sum(
                1 for row in dones
                if row.get("raw_termination_reason") not in known),
            f"{prefix}MissileLifetimeMeanSec": _mean_or_zero(lifetimes_team),
            f"{prefix}MissileLifetimeMedianSec": _percentile_or_zero(
                lifetimes_team, 50),
            f"{prefix}MissileLifetimeP90Sec": _percentile_or_zero(
                lifetimes_team, 90),
            f"{prefix}MissileLifetimeMinSec": (
                min(lifetimes_team) if lifetimes_team else 0.0),
            f"{prefix}MissileLifetimeMaxSec": (
                max(lifetimes_team) if lifetimes_team else 0.0),
            f"{prefix}LaunchInsideHitRadiusCount": inside,
            f"{prefix}LaunchInsideHitRadiusFrac": _safe_div(inside, len(launches)),
            f"{prefix}OnePhysicsFrameHitCount": one_frame_hits,
            f"{prefix}OnePhysicsFrameHitFrac": _safe_div(
                one_frame_hits, len(hits)),
            f"{prefix}MissileSurvivedOneDecisionCount": survived,
            f"{prefix}MissileSurvivedOneDecisionFrac": _safe_div(
                survived, len(dones)),
            f"{prefix}PNFramesMean": _mean_or_zero(team_pn_frames),
            f"{prefix}PNNonZeroCommandFrac": _safe_div(
                sum(team_pn_nonzero), sum(team_pn_frames)),
        })
    first_events = {
        key: _numeric_values(completed_episode_events, key)
        for key in (
            "red_first_launch_step", "blue_first_launch_step",
            "red_first_hit_step", "blue_first_hit_step")
    }
    result.update({
        "MissileLifetimeMeanS": _mean_or_zero(lifetimes),
        "MissileLifetimeP50S": _percentile_or_zero(lifetimes, 50),
        "MissileOneFrameTerminations": sum(value <= 1.0 / 60.0 + 1e-9
                                             for value in lifetimes),
        "MissileLifetimeOverPoint2S": sum(value > 0.2 for value in lifetimes),
        "MissilePNGuidanceFrames": sum(pn_frames),
        "MissilePNNonzeroCommandFrames": sum(pn_nonzero),
        "MissileMaximumCommandG": max(max_commands) if max_commands else 0.0,
        "TargetReallocations": int(environment_diag["target_reallocations"]),
        "TargetReallocationsAfterDeath": int(
            environment_diag["target_reallocations_after_death"]),
        "TargetSwitchesWhileAlive": int(
            environment_diag["target_switches_while_alive"]),
        "TargetEngagedWaitFrames": int(environment_diag["engaged_wait_frames"]),
        "TargetNoAliveFrames": int(environment_diag["no_alive_target_frames"]),
        "RedMWSDetectedAgentDecisions": int(
            environment_diag["red_detected_agent_decisions"]),
        "RedMWSOverrideAgentDecisions": int(
            environment_diag["red_override_agent_decisions"]),
        "BlueMWSDetectedAgentDecisions": int(
            environment_diag["blue_detected_agent_decisions"]),
        "BlueMWSOverrideAgentDecisions": int(
            environment_diag["blue_override_agent_decisions"]),
        "RedMWSWarningGenerations": int(
            environment_diag["red_warning_generations"]),
        "BlueMWSWarningGenerations": 0,
        "RedMWSDirectionChangesWithinSameMissile": int(
            environment_diag["red_direction_changes_within_same_missile"]),
        "RedMWSSuppressedDirectionFlipAttempts": int(
            environment_diag["red_suppressed_direction_flip_attempts"]),
        "BlueMWSDirectionChangesWithinSameMissile": 0,
        "NonFiniteLoadInvalidEpisodes": int(
            environment_diag["invalid_nonfinite_load_count"]),
        "CatastrophicFiniteLoadInvalidEpisodes": int(
            environment_diag["invalid_catastrophic_finite_load_count"]),
        "PersistentExtremeFiniteLoadInvalidEpisodes": int(
            environment_diag["invalid_persistent_extreme_finite_load_count"]),
        "RedMWSMaximumContinuousDecisions": int(
            environment_diag["red_maximum_continuous_decisions"]),
        "RedMWSTargetHeadingDeltaMaxDeg": float(
            environment_diag["red_target_heading_delta_max_deg"]),
        **{
            f"{prefix}{suffix}": _safe_div(
                environment_diag[field], environment_diag[f"{field}_count"])
            for prefix, field, suffix in (
                ("Red", "red_warning_to_terminal_mean_s", "WarningToTerminalMeanS"),
                ("Red", "red_warning_to_terminal_p50_s", "WarningToTerminalP50S"),
                ("Red", "red_warning_to_hit_mean_s", "WarningToHitMeanS"),
                ("Blue", "blue_warning_to_terminal_mean_s", "WarningToTerminalMeanS"),
                ("Blue", "blue_warning_to_terminal_p50_s", "WarningToTerminalP50S"),
                ("Blue", "blue_warning_to_hit_mean_s", "WarningToHitMeanS"),
            )},
        "CompletedEpisodesThisIteration": int(iter_episodes),
        "NoCompletedEpisodeThisIteration": int(iter_episodes == 0),
        "IterationWallTimeS": float(wall_time_s),
        "EnvironmentStepsPerSecond": (
            float(environment_steps / wall_time_s) if wall_time_s > 0 else 0.0),
        "RedFirstLaunchStepMean": (
            float(np.mean(first_events["red_first_launch_step"]))
            if first_events["red_first_launch_step"] else float("nan")),
        "BlueFirstLaunchStepMean": (
            float(np.mean(first_events["blue_first_launch_step"]))
            if first_events["blue_first_launch_step"] else float("nan")),
        "RedFirstHitStepMean": (
            float(np.mean(first_events["red_first_hit_step"]))
            if first_events["red_first_hit_step"] else float("nan")),
        "BlueFirstHitStepMean": (
            float(np.mean(first_events["blue_first_hit_step"]))
            if first_events["blue_first_hit_step"] else float("nan")),
        "EpisodeLengthMean": (
            float(np.mean(episode_lengths)) if episode_lengths else float("nan")),
        "EpisodeLengthP50": (
            float(np.percentile(episode_lengths, 50))
            if episode_lengths else float("nan")),
        "EpisodeLengthP90": (
            float(np.percentile(episode_lengths, 90))
            if episode_lengths else float("nan")),
        "EstimatedRemainingTimeSec": (
            float(remaining_steps * wall_time_s / environment_steps)
            if environment_steps > 0 else 0.0),
        "WorkerRestartCount": int(worker_restart_count),
        "ResumeCount": int(resume_count),
    })
    return result


def _numeric_values(records: list[dict], key: str, abs_value: bool = False) -> list[float]:
    vals = []
    for record in records:
        try:
            value = float(record.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        vals.append(abs(value) if abs_value else value)
    return vals


def _mean_or_zero(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _percentile_or_zero(vals: list[float], pct: float) -> float:
    return float(np.percentile(vals, pct)) if vals else 0.0


def _launch_quality_metrics(
    launch_records: list[dict],
    done_records: list[dict],
) -> dict:
    metrics = {}
    for team, prefix in (("red", "Red"), ("blue", "Blue")):
        launches = [r for r in launch_records if r.get("team") == team]
        dones = [r for r in done_records if r.get("team") == team]
        ranges = _numeric_values(launches, "range_m")
        ao_deg = _numeric_values(launches, "AO_deg")
        ta_deg = _numeric_values(launches, "TA_deg")
        closing = _numeric_values(launches, "closing_speed_mps")
        alt_abs = _numeric_values(launches, "altitude_diff_m", abs_value=True)
        hit_count = sum(1 for r in dones if str(r.get("is_success")).lower() == "true"
                        or r.get("is_success") is True)
        metrics.update({
            f"{prefix}LaunchRangeMean": _mean_or_zero(ranges),
            f"{prefix}LaunchRangeP25": _percentile_or_zero(ranges, 25),
            f"{prefix}LaunchRangeP50": _percentile_or_zero(ranges, 50),
            f"{prefix}LaunchRangeP75": _percentile_or_zero(ranges, 75),
            f"{prefix}LaunchAoDegMean": _mean_or_zero(ao_deg),
            f"{prefix}LaunchAoDegP50": _percentile_or_zero(ao_deg, 50),
            f"{prefix}LaunchTaDegMean": _mean_or_zero(ta_deg),
            f"{prefix}LaunchTaDegP50": _percentile_or_zero(ta_deg, 50),
            f"{prefix}LaunchClosingSpeedMean": _mean_or_zero(closing),
            f"{prefix}LaunchAltitudeDiffAbsMean": _mean_or_zero(alt_abs),
            f"{prefix}LaunchHitRateFromQualityRecords": _safe_div(hit_count, len(dones)),
        })
    return metrics


def _empty_action_bound_totals() -> dict:
    return {
        "executed_abs_sum": 0.0,
        "element_count": 0,
        "executed_near_bound_count": 0,
        "executed_dim_near_bound_count": np.zeros(3, dtype=np.int64),
        "policy_mean_near_bound_count": 0,
        "policy_mean_element_count": 0,
        "dim_count": np.zeros(3, dtype=np.int64),
    }


def _accumulate_action_bound_totals(
    totals: dict,
    executed_actions,
    policy_mean_actions,
    threshold: float = 0.999,
) -> None:
    executed = np.asarray(executed_actions, dtype=np.float64)
    policy_mean = np.asarray(policy_mean_actions, dtype=np.float64)
    if executed.size == 0:
        return
    executed = executed.reshape(-1, executed.shape[-1])
    policy_mean = policy_mean.reshape(-1, policy_mean.shape[-1])
    executed_near_bound = np.abs(executed) >= threshold
    policy_mean_near_bound = np.abs(policy_mean) >= threshold
    totals["executed_abs_sum"] += float(np.abs(executed).sum())
    totals["element_count"] += int(executed.size)
    totals["executed_near_bound_count"] += int(executed_near_bound.sum())
    totals["policy_mean_near_bound_count"] += int(policy_mean_near_bound.sum())
    totals["policy_mean_element_count"] += int(policy_mean.size)
    dims = min(3, executed.shape[-1])
    totals["executed_dim_near_bound_count"][:dims] += executed_near_bound[:, :dims].sum(
        axis=0).astype(np.int64)
    totals["dim_count"][:dims] += executed.shape[0]


def _action_bound_metrics(totals: dict) -> dict:
    elem_count = int(totals["element_count"])
    dim_count = totals["dim_count"]
    executed_dim_near = totals["executed_dim_near_bound_count"]
    if elem_count == 0:
        return {field: float("nan") for field in ACTION_BOUND_CSV_FIELDS}
    return {
        "ExecutedActionAbsMean": _safe_div(
            totals["executed_abs_sum"], elem_count),
        "ExecutedActionNearBoundFrac": _safe_div(
            totals["executed_near_bound_count"], elem_count),
        "ExecutedActionNearBoundFracPitch": _safe_div(
            executed_dim_near[0], dim_count[0]),
        "ExecutedActionNearBoundFracHeading": _safe_div(
            executed_dim_near[1], dim_count[1]),
        "ExecutedActionNearBoundFracVelocity": _safe_div(
            executed_dim_near[2], dim_count[2]),
        "PolicyMeanNearBoundFrac": _safe_div(
            totals["policy_mean_near_bound_count"],
            totals["policy_mean_element_count"]),
    }


def _cleanup_rotating_checkpoints(directory: str, prefix: str, keep: int = 5):
    """删除超出保留数量的旧轮转 checkpoint 文件。

    Args:
        directory: checkpoint 所在目录
        prefix:    文件名前缀 (e.g. "vanilla_actor_latest")
        keep:      保留最新的文件数量 (默认 5)
    """
    import glob as _glob
    pattern = os.path.join(directory, f"{prefix}_*.pt")
    files = sorted(_glob.glob(pattern))
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass


# ==============================================================================
#  蓝方自动驾驶仪 —— 高度预算控制器 (Altitude Budget Controller)
#
#  核心思想: 高度是消耗性资源，机动侵略性必须随高度降低而收敛。
#  高空 = 富有 → 可以大坡度追击。  低空 = 贫穷 → 只求自保，禁止一切战斗。
#
#  动作通道映射 (env.py _parse_actions 中的定义):
#    action[0] = pitch_cmd    → target_pitch        (正=抬头, PID→elevator)
#    action[1] = heading_cmd  → target_heading_delta (正=右转, PID→aileron)
#    action[2] = vel_cmd      → target_velocity      (正=加速, PID→throttle)
# ==============================================================================

# ==============================================================================
#  SubprocVecEnv (从 train_ppo.py 精简，保留超时保护)
# ==============================================================================
def _worker(remote: mp.connection.Connection,
            parent_remote: mp.connection.Connection,
            env_kwargs: dict):
    """子进程入口：自行构造 UavCombatEnv 并循环响应指令。"""
    parent_remote.close()

    # Worker 进程也锁定单线程：NumPy 操作（obs 拼接等）不使用 OpenMP 线程池
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # Allow Intel + LLVM OpenMP runtimes to coexist (JSBSim vs numpy/torch)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Note: we do NOT apply permanent CRT/Win32 stdout suppression here.
    # The SuppressOutput context manager inside AircraftSimulator.reload()
    # handles JSBSim C++ banner suppression *temporarily* during FGFDMExec
    # construction, with proper save/restore of fds, handles, and Python
    # stdout objects.  Permanent OS-level suppression was causing worker
    # crashes on some Windows configurations.

    # ---- Step-by-step init with per-stage crash diagnostics ----
    env = None
    try:
        from my_uav_env import UavCombatEnv
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_import_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during import:\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    try:
        env = UavCombatEnv(**env_kwargs)
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_env_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during UavCombatEnv():\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    try:
        remote.send(("ready", os.getpid()))
    except Exception:
        import traceback as _tb
        with open(f"_worker_crash_ready_{os.getpid()}.txt", "w") as _f:
            _f.write(f"Worker PID {os.getpid()} crashed during ready signal:\n")
            _tb.print_exc(file=_f)
        remote.close()
        return

    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, BrokenPipeError, OSError):
            break
        try:
            if cmd == "step":
                obs, rewards, terminated, truncated, info = env.step(data)
                dones = {}
                for aid in env.agent_ids:
                    dones[aid] = bool(terminated.get(aid, False) or truncated.get(aid, False))
                if all(dones.values()):
                    obs, _ = env.reset()
                    import gc
                    gc.collect()
                remote.send((obs, rewards, dones, info))
            elif cmd == "reset":
                obs, info = env.reset(seed=data)
                import gc
                gc.collect()
                remote.send(obs)
            elif cmd == "call":
                method_name, args, kwargs = data
                result = getattr(env, method_name)(*args, **kwargs)
                remote.send(result)
            elif cmd == "close":
                remote.close()
                break
        except Exception:
            import traceback as _tb
            with open(f"_worker_crash_{os.getpid()}.txt", "w") as _f:
                _f.write(f"Worker PID {os.getpid()} crashed on cmd={cmd}:\n")
                _tb.print_exc(file=_f)
            try:
                remote.send(("error", _tb.format_exc()))
            except Exception:
                pass
            break


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, startup_delay: float = 0.5,
                 ready_timeout: float = 600.0, base_seed: int | None = None):
        self.n_envs = num_envs
        self._dead_workers: set[int] = set()
        self._env_kwargs = env_kwargs  # stored for worker restart
        self._base_seed = base_seed
        self._has_reset = False
        self.worker_restart_count = 0
        ctx = mp.get_context("spawn")
        remotes_tup, work_remotes_tup = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.remotes = list(remotes_tup)
        self.work_remotes = list(work_remotes_tup)
        self.processes = []
        for i in range(num_envs):
            p = ctx.Process(target=_worker,
                            args=(self.work_remotes[i], self.remotes[i], env_kwargs),
                            daemon=True)
            p.start()
            self.processes.append(p)
            self.work_remotes[i].close()
            if i < num_envs - 1:
                time.sleep(startup_delay)

        # Wait for every worker to signal readiness (env constructed, import done)
        ready_count = 0
        for i, remote in enumerate(self.remotes):
            try:
                if not remote.poll(ready_timeout):
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not become ready within {ready_timeout:.0f}s")
                msg = remote.recv()
            except (BrokenPipeError, OSError, EOFError):
                raise RuntimeError(
                    f"Worker {i} (PID {self.processes[i].pid}) "
                    f"pipe broken during startup — OS may have killed the process") from None
            if isinstance(msg, tuple) and msg[0] == "ready":
                ready_count += 1
            else:
                raise RuntimeError(
                    f"Worker {i} sent unexpected init message: {str(msg)[:200]}")
        print(f"  All {ready_count} workers ready", flush=True)

    def reset(self, timeout: float = 300.0,
              serial: bool = True) -> list[dict]:
        """Reset all workers.

        When ``serial=True`` (default), each worker is reset one at a time —
        send reset, wait for response, then move to the next worker.  This
        prevents concurrent JSBSim construction (96 FGFDMExec instances) that
        triggers OS-level process kills on Windows.  Subsequent resets that use
        ``reload()`` (fast) can pass ``serial=False``.
        """
        results = [None] * len(self.remotes)
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results[i] = {}
                continue
            # Send reset command
            try:
                seed = (None if self._has_reset or self._base_seed is None
                        else int(self._base_seed) + i)
                remote.send(("reset", seed))
            except (BrokenPipeError, OSError):
                self._dead_workers.add(i)
                results[i] = {}
                continue

            if serial:
                # Wait for this worker's response before talking to the next one
                try:
                    ready = remote.poll(timeout)
                except (BrokenPipeError, OSError, EOFError):
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died before reset response", flush=True)
                    self._dead_workers.add(i)
                    results[i] = {}
                    continue
                if ready:
                    try:
                        msg = remote.recv()
                    except EOFError:
                        print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                              f"died during reset", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                        continue
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error during reset: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                    else:
                        results[i] = msg
                else:
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not respond within {timeout:.0f}s")

        # If not serial, collect responses after all sends (original behaviour)
        if not serial:
            for i, remote in enumerate(self.remotes):
                if results[i] is not None:
                    continue  # already handled (dead worker)
                try:
                    ready = remote.poll(timeout)
                except (BrokenPipeError, OSError, EOFError):
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died before reset response", flush=True)
                    self._dead_workers.add(i)
                    results[i] = {}
                    continue
                if ready:
                    try:
                        msg = remote.recv()
                    except EOFError:
                        print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                              f"died during reset", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                        continue
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error during reset: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results[i] = {}
                    else:
                        results[i] = msg
                else:
                    raise TimeoutError(
                        f"Worker {i} (PID {self.processes[i].pid}) "
                        f"did not respond within {timeout:.0f}s")

        # Auto-restart dead workers after reset (retry up to 3 times)
        for i in list(self._dead_workers):
            # Clean up any crash tracer files from the dead worker
            _dead_pid = self.processes[i].pid
            for _pattern in ("_worker_crash_import_", "_worker_crash_env_",
                             "_worker_crash_ready_", "_worker_crash_",
                             "_worker_reset_tracer_", "_jsbsim_tracer_"):
                _tracer_path = f"{_pattern}{_dead_pid}.txt"
                if os.path.exists(_tracer_path):
                    try:
                        os.remove(_tracer_path)
                    except OSError:
                        pass
            restarted = False
            for attempt in range(3):
                try:
                    new_obs = self._restart_worker(i, self._env_kwargs)
                    self.worker_restart_count += 1
                    results[i] = new_obs
                    print(f"  [INFO] Worker {i} restarted after reset (attempt {attempt+1})", flush=True)
                    restarted = True
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  [WARN] Worker {i} restart attempt {attempt+1} failed: {e}", flush=True)
                        time.sleep(5.0)
                    else:
                        print(f"  [ERROR] Worker {i} restart failed after 3 attempts: {e}", flush=True)
            if not restarted:
                raise RuntimeError(
                    f"Worker {i} could not be restarted after 3 attempts. "
                    f"Training cannot continue with a dead environment.")
        self._has_reset = True
        return results

    def step(self, actions_list: list[dict], timeout: float = 60.0) -> tuple:
        for i, (remote, actions) in enumerate(zip(self.remotes, actions_list)):
            if i not in self._dead_workers:
                try:
                    remote.send(("step", actions))
                except (BrokenPipeError, OSError):
                    self._dead_workers.add(i)
        results = []
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results.append(({}, {}, {}, {}))
                continue
            try:
                ready = remote.poll(timeout)
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died before step response", flush=True)
                self._dead_workers.add(i)
                results.append(({}, {}, {}, {}))
                continue
            if ready:
                try:
                    msg = remote.recv()
                except EOFError:
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"died (EOFError)", flush=True)
                    self._dead_workers.add(i)
                    results.append(({}, {}, {}, {}))
                    continue
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                    print(f"  [WARN] Worker {i} error: {msg[1][:200]}", flush=True)
                    self._dead_workers.add(i)
                    results.append(({}, {}, {}, {}))
                else:
                    results.append(msg)
            else:
                raise TimeoutError(
                    f"Worker {i} (PID {self.processes[i].pid}) "
                    f"did not respond within {timeout:.0f}s")

        # Auto-restart dead workers and fix up dummy results
        for i in list(self._dead_workers):
            try:
                new_obs = self._restart_worker(i, self._env_kwargs)
                self.worker_restart_count += 1
                results[i] = (
                    new_obs,
                    {aid: 0.0 for aid in new_obs},
                    {aid: True for aid in new_obs},
                    {
                        "__episode__": {
                            "worker_restart_episode": True,
                            "invalid_numerical_episode": True,
                            "invalid_numerical_reasons": ["WorkerRestart"],
                            "episode_end_reason": "worker_restart",
                            "winner": "",
                        }
                    },
                )
            except Exception as e:
                print(f"  [WARN] Failed to restart worker {i}: {e}", flush=True)

        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def env_method(self, method_name: str, *args, timeout: float = 30.0, **kwargs):
        """Call a method on every remote env and return the list of results."""
        results = []
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results.append(set())
                continue
            try:
                remote.send(("call", (method_name, args, kwargs)))
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died during env_method send", flush=True)
                self._dead_workers.add(i)
                results.append(set())
                continue
            try:
                if remote.poll(timeout):
                    msg = remote.recv()
                    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                        print(f"  [WARN] Worker {i} error in {method_name}: {msg[1][:200]}", flush=True)
                        self._dead_workers.add(i)
                        results.append(set())
                    else:
                        results.append(msg)
                else:
                    print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                          f"timed out during env_method {method_name}", flush=True)
                    self._dead_workers.add(i)
                    results.append(set())
            except (BrokenPipeError, OSError, EOFError):
                print(f"  [WARN] Worker {i} (PID {self.processes[i].pid}) "
                      f"died during env_method recv", flush=True)
                self._dead_workers.add(i)
                results.append(set())
        return results

    def env_method_each(self, method_name: str, calls: list[tuple[tuple, dict]],
                        timeout: float = 30.0):
        """Call one method with per-environment arguments."""
        if len(calls) != len(self.remotes):
            raise ValueError("calls length must match number of environments")
        for i, (remote, (args, kwargs)) in enumerate(zip(self.remotes, calls)):
            if i not in self._dead_workers:
                remote.send(("call", (method_name, args, kwargs)))
        results = []
        for i, remote in enumerate(self.remotes):
            if i in self._dead_workers:
                results.append({})
                continue
            if not remote.poll(timeout):
                raise TimeoutError(f"Worker {i} timed out during {method_name}")
            msg = remote.recv()
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
                raise RuntimeError(f"Worker {i} error in {method_name}: {msg[1]}")
            results.append(msg)
        return results

    def _restart_worker(self, i: int, env_kwargs: dict):
        """Restart a dead worker and return its initial observation."""
        # Clean up old process
        old_p = self.processes[i]
        if old_p.is_alive():
            old_p.terminate()
            old_p.join(timeout=5)
        # Close old remote
        try:
            self.remotes[i].close()
        except Exception:
            pass
        # Create new pipe and process
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_worker,
                        args=(child_conn, parent_conn, env_kwargs),
                        daemon=True)
        p.start()
        child_conn.close()
        self.processes[i] = p
        self.remotes[i] = parent_conn
        # Wait for ready signal (with protection against immediate worker death)
        try:
            if not parent_conn.poll(600.0):
                raise RuntimeError(f"Worker {i} ready signal timed out")
            msg = parent_conn.recv()
            if not (isinstance(msg, tuple) and msg[0] == "ready"):
                raise RuntimeError(f"Worker {i} unexpected ready msg: {str(msg)[:200]}")
            # Send reset command
            parent_conn.send(("reset", None))
            if not parent_conn.poll(300.0):
                raise RuntimeError(f"Worker {i} reset timed out")
            obs = parent_conn.recv()
            if isinstance(obs, tuple) and len(obs) == 2 and obs[0] == "error":
                raise RuntimeError(f"Worker {i} error during reset: {obs[1][:200]}")
            self._dead_workers.discard(i)
            print(f"  [INFO] Worker {i} restarted (PID {p.pid})", flush=True)
            return obs
        except (BrokenPipeError, OSError, EOFError) as e:
            raise RuntimeError(f"Worker {i} pipe broken during restart: {e}") from e

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _fetch_blue_own_positions(vec_env, timeout: float = 30.0) -> list[dict]:
    """Fetch blue ownship positions from worker envs for rule-policy patrol."""

    raw = vec_env.env_method("get_blue_own_positions", timeout=timeout)
    return [item if isinstance(item, dict) else {} for item in raw]


def _fetch_blue_own_kinematics(
    vec_env,
    timeout: float = 30.0,
) -> tuple[list[dict], list[dict]]:
    """Fetch blue ownship positions/headings from worker envs."""

    raw = vec_env.env_method("get_blue_own_kinematics", timeout=timeout)
    positions_list: list[dict] = []
    headings_list: list[dict] = []
    for item in raw:
        pos: dict = {}
        hdg: dict = {}
        if isinstance(item, dict):
            for bid, data in item.items():
                if isinstance(data, dict):
                    if "position" in data:
                        pos[bid] = data["position"]
                    if "heading" in data:
                        hdg[bid] = data["heading"]
        positions_list.append(pos)
        headings_list.append(hdg)
    return positions_list, headings_list


# ==============================================================================
#  Rollout Buffer (无掩码字段，精简版)
# ==============================================================================
class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, num_red: int,
                 action_dim: int, rnn_hidden_size: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_red = num_red
        T, E, A = num_steps, num_envs, num_red
        H = rnn_hidden_size

        # 展平观测存储 (可变长度 obs_dim 由实际数据决定)
        self.obs: list[list[list[np.ndarray]]] = [
            [[None for _ in range(A)] for _ in range(E)] for _ in range(T)]
        self.actions = np.zeros((T, E, A, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, E, A), dtype=np.float32)
        self.action_stds = np.zeros((T, E, A, action_dim), dtype=np.float32)
        self.policy_mean_actions = np.zeros(
            (T, E, A, action_dim), dtype=np.float32)
        self.alive = np.zeros((T, E, A), dtype=bool)
        self.actor_rnn_states_before = np.zeros((T, E, A, H), dtype=np.float32)
        self.actor_sequence_start = np.zeros((T, E, A), dtype=bool)

        # Paper joint trajectory: exactly one state/reward/value/done per env-step.
        self.global_states: list[list[np.ndarray | None]] = [
            [None for _ in range(E)] for _ in range(T)]
        self.joint_rewards = np.zeros((T, E), dtype=np.float32)
        self.team_values = np.zeros((T, E), dtype=np.float32)
        self.episode_dones = np.zeros((T, E), dtype=np.float32)
        self.valid_transitions = np.ones((T, E), dtype=bool)

        # GAE bootstrap value is environment-level; critic has one team value.
        self.bootstrap_value = np.zeros(E, dtype=np.float32)

    def invalidate_episode(self, env_idx: int, start_step: int, end_step: int) -> int:
        """Invalidate one episode's transitions within the current rollout."""
        start = max(0, int(start_step))
        end = min(self.num_steps, int(end_step) + 1)
        previous = int(np.count_nonzero(~self.valid_transitions[start:end, env_idx]))
        self.valid_transitions[start:end, env_idx] = False
        return max(0, end - start - previous)

    def store_team_step(self, step: int, env_idx: int, global_state: np.ndarray,
                        joint_reward: float, value: float, episode_done: float):
        self.global_states[step][env_idx] = np.asarray(global_state, dtype=np.float32)
        self.joint_rewards[step, env_idx] = float(joint_reward)
        self.team_values[step, env_idx] = float(value)
        self.episode_dones[step, env_idx] = float(episode_done)

    def store_step(self, step: int, env_idx: int, agent_idx: int,
                   obs_np: np.ndarray, action: np.ndarray,
                   log_prob: float, alive: bool,
                   actor_rnn_state_before: np.ndarray,
                   sequence_start: bool,
                   action_std: np.ndarray | None = None,
                   policy_mean_action: np.ndarray | None = None):
        self.obs[step][env_idx][agent_idx] = obs_np
        self.actions[step, env_idx, agent_idx] = action
        self.log_probs[step, env_idx, agent_idx] = log_prob
        if action_std is not None:
            self.action_stds[step, env_idx, agent_idx] = np.asarray(
                action_std, dtype=np.float32)
        if policy_mean_action is not None:
            self.policy_mean_actions[step, env_idx, agent_idx] = np.asarray(
                policy_mean_action, dtype=np.float32)
        self.alive[step, env_idx, agent_idx] = alive
        self.actor_rnn_states_before[step, env_idx, agent_idx] = np.asarray(
            actor_rnn_state_before, dtype=np.float32)
        self.actor_sequence_start[step, env_idx, agent_idx] = bool(sequence_start)


# ==============================================================================
#  GAE
# ==============================================================================
def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float, lam: float):
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


def _compute_joint_gae_by_env(buffer: RolloutBuffer, gamma: float, lam: float,
                              device: torch.device):
    """Compute exactly one joint advantage/return trajectory per environment."""
    advantages_by_env = []
    returns_by_env = []
    for env_idx in range(buffer.joint_rewards.shape[1]):
        rewards = torch.as_tensor(
            buffer.joint_rewards[:, env_idx], dtype=torch.float32, device=device)
        values = torch.as_tensor(
            np.concatenate([
                buffer.team_values[:, env_idx],
                np.array([buffer.bootstrap_value[env_idx]], dtype=np.float32),
            ]), dtype=torch.float32, device=device)
        dones = torch.as_tensor(
            buffer.episode_dones[:, env_idx], dtype=torch.float32, device=device)
        valid = torch.as_tensor(
            buffer.valid_transitions[:, env_idx], dtype=torch.bool, device=device)
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=device)
        for t in reversed(range(buffer.num_steps)):
            if not bool(valid[t]):
                gae = torch.tensor(0.0, device=device)
                continue
            next_value = values[t + 1]
            if t + 1 < buffer.num_steps and not bool(valid[t + 1]):
                next_value = torch.tensor(0.0, device=device)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:buffer.num_steps]
        returns = torch.where(valid, returns, torch.zeros_like(returns))
        advantages_by_env.append(advantages)
        returns_by_env.append(returns)
    return advantages_by_env, returns_by_env


# ==============================================================================
#  PPO Update (无掩码)
# ==============================================================================
def _grad_has_nan(module: nn.Module) -> bool:
    """Return True if any parameter gradient contains NaN or Inf."""
    for p in module.parameters():
        if p.grad is not None:
            if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                return True
    return False


def _current_entropy_coef(config, _total_steps: int = 0) -> float:
    """Constant entropy coefficient (paper Table 3)."""
    return config.entropy_coef


def _safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0)


def _action_std_growth_ratio(action_std_mean: float) -> float:
    """Return true growth relative to the configured 0.3 initial std."""
    value = float(action_std_mean)
    return value / ACTION_STD_INIT if np.isfinite(value) else float("nan")


def _csv_optional_float(value: float, digits: int = 6) -> str:
    value = float(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else ""


def _result_optional_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _ratio_with_denominator_zero(num: float, den: float) -> tuple[float, bool]:
    """Return a true ratio plus an explicit zero-denominator indicator."""
    numerator = float(num)
    denominator = float(den)
    if denominator != 0.0:
        return numerator / denominator, False
    return (float("inf") if numerator > 0.0 else 0.0), True


def _classify_death_reason(reason: str | None) -> str:
    if not reason:
        return "other"
    r = str(reason).lower()
    if "shot" in r or "missile" in r or "hit" in r:
        return "missile"
    if "crash" in r or "ground" in r or "altitude" in r:
        return "crash"
    return "other"


def _episode_outcome(red_alive: int, blue_alive: int) -> str:
    if red_alive > blue_alive:
        return "red"
    if blue_alive > red_alive:
        return "blue"
    return "draw"


def _episode_is_invalid(info: dict) -> bool:
    return bool(info.get("__episode__", {}).get(
        "invalid_numerical_episode", False))


def _joint_team_reward_once(rewards: dict, team_ids: list[str]) -> float:
    """Read a replicated team reward once and reject inconsistent values."""
    values = [float(rewards[aid]) for aid in team_ids if aid in rewards]
    if not values:
        return 0.0
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        raise ValueError("team agents received inconsistent joint rewards")
    return values[0]


def _reward_component_log_metrics(
    team_component_sums: dict[str, float], team_size: int,
) -> dict[str, float]:
    """Name team component sums and their fixed-team per-agent means."""
    denominator = max(int(team_size), 1)
    metrics = {}
    for key in LOCAL_REWARD_COMPONENT_KEYS:
        team_sum = float(team_component_sums.get(key, 0.0))
        metrics[f"{key}_team_sum"] = team_sum
        metrics[f"{key}_per_agent_mean"] = team_sum / denominator
    metrics["r_end_team"] = float(team_component_sums.get("r_end", 0.0))
    return metrics


def _actor_std_stats(actor, sampled_stds: list[np.ndarray] | None = None) -> dict:
    """Return diagnostic std / log_std metrics for entropy monitoring.

    Does not modify any parameter or alter training behaviour.
    """
    if sampled_stds is not None and not sampled_stds:
        return {
            "action_std_mean": float("nan"),
            "action_std_min": float("nan"),
            "action_std_max": float("nan"),
            "action_log_std_mean": float("nan"),
            "action_std_lower_bound_frac": float("nan"),
            "action_std_upper_bound_frac": float("nan"),
        }
    if sampled_stds:
        std = torch.as_tensor(
            np.concatenate([np.asarray(x).reshape(-1) for x in sampled_stds]),
            dtype=torch.float32)
    else:
        with torch.no_grad():
            raw = actor.action_log_std_head.bias
            lower = float(np.log(ACTION_STD_MIN))
            upper = float(np.log(ACTION_STD_MAX))
            std = torch.exp(lower + 0.5 * (torch.tanh(raw) + 1.0) * (
                upper - lower))
    log_std = torch.log(std)
    tolerance = 1e-4
    return {
        "action_std_mean": float(std.mean().item()),
        "action_std_min":  float(std.min().item()),
        "action_std_max":  float(std.max().item()),
        "action_log_std_mean": float(log_std.mean().item()),
        "action_std_lower_bound_frac": float(
            (std <= ACTION_STD_MIN + tolerance).float().mean().item()),
        "action_std_upper_bound_frac": float(
            (std >= ACTION_STD_MAX - tolerance).float().mean().item()),
    }


def _ppo_update_legacy(actor, critic, actor_opt, critic_opt, buffer, config, device,
                       total_steps: int = 0):
    """Disabled pre-joint-reward update retained only for migration errors.

    It references the removed per-agent return buffer and must never be called.
    """
    raise RuntimeError("legacy per-agent PPO update is disabled")
    num_steps = buffer.num_steps
    num_envs = buffer.rnn_actor_init.shape[0]
    num_red = buffer.rnn_actor_init.shape[1]

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_agents = 0

    actor_opt.zero_grad()
    critic_opt.zero_grad()

    for env_idx in range(num_envs):
        # ---- Build paper CTDE state (all red native ego states) ----
        global_obs_seq = []
        for t in range(num_steps):
            parts = [buffer.obs[t][env_idx][i] for i in range(num_red)]
            global_obs_seq.append(_global_state_from_local_obs_flats(
                parts, obs_mode=config.obs_mode))

        # ---- Centralized critic: batch forward all timesteps → V(s_global_t) ----
        gobs_batch = torch.as_tensor(np.stack(global_obs_seq), dtype=torch.float32,
                                      device=device)  # (T, global_obs_dim)
        new_vals_global = critic(gobs_batch).squeeze(-1)  # (T,)

        env_critic_loss = 0.0
        env_critic_count = 0

        for agent_idx in range(num_red):
            # ---- Collect trajectory (alive steps only) ----
            t_obs_flat = []
            t_act = []
            t_rew = []
            t_val = []
            t_lp = []
            t_done = []
            alive_steps = []
            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_obs_flat.append(buffer.obs[step][env_idx][agent_idx])
                    t_act.append(buffer.actions[step, env_idx, agent_idx])
                    t_rew.append(buffer.rewards[step, env_idx, agent_idx])
                    t_val.append(buffer.values[step, env_idx, agent_idx])
                    t_lp.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_done.append(buffer.dones[step, env_idx, agent_idx])
                    alive_steps.append(step)

            if len(t_act) == 0:
                continue

            T = len(t_act)

            # ---- GAE (uses centralized values stored during rollout) ----
            bootstrap = float(buffer.bootstrap_values[env_idx, agent_idx])
            rewards = torch.tensor(t_rew, device=device)
            old_values = torch.tensor(t_val + [bootstrap], device=device)
            dones = torch.tensor(t_done, device=device)

            advantages, returns = compute_gae(rewards, old_values, dones,
                                              config.gamma, config.gae_lambda)

            if advantages.numel() > 1:
                adv_std = advantages.std()
            else:
                adv_std = torch.std(advantages, correction=0)
            if adv_std <= 1e-8 or torch.isnan(adv_std):
                adv_std = 1.0
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            # ---- Critic loss: MSE(V(s_global_t), return_i(t)) per alive step ----
            for t_idx, step in enumerate(alive_steps):
                env_critic_loss += F.mse_loss(
                    new_vals_global[step], returns[t_idx].detach(),
                    reduction='none')
                env_critic_count += 1

            # ---- Actor GRU unroll (per-agent, from rollout init state) ----
            rnn_a = torch.as_tensor(
                buffer.rnn_actor_init[env_idx, agent_idx], device=device).unsqueeze(0)

            new_lps = []
            entropies = []

            for t in range(T):
                obs_t = torch.as_tensor(t_obs_flat[t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                act_t = torch.as_tensor(t_act[t], dtype=torch.float32, device=device)

                action_dist, rnn_a = actor(obs_t, rnn_a)

                new_lp = action_dist.log_prob(act_t.unsqueeze(0)).sum(dim=-1)
                new_lps.append(new_lp)
                entropies.append(action_dist.base_entropy().sum(dim=-1).mean())

            new_lp = torch.cat(new_lps)
            old_lp = torch.tensor(t_lp, device=device)

            ent_avg = torch.stack(entropies).mean()

            # ---- PPO Clip Loss ----
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                1 + config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_coef = _current_entropy_coef(config, total_steps)
            actor_loss = policy_loss - entropy_coef * ent_avg

            actor_loss.backward()

            total_actor_loss += actor_loss.item()
            total_entropy += ent_avg.item()
            n_agents += 1

        # ---- Centralized critic loss (per-env, averaged across agents) ----
        if env_critic_count > 0:
            env_critic_loss = env_critic_loss / env_critic_count
            env_critic_loss.backward()
            total_critic_loss += env_critic_loss.item()

    if n_agents == 0:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # ---- NaN gradient guard ----
    actor_nan = _grad_has_nan(actor)
    critic_nan = _grad_has_nan(critic)
    if actor_nan or critic_nan:
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        print(f"  [WARN] NaN gradient detected (actor={actor_nan}, critic={critic_nan}) — "
              f"skipping optimizer step to preserve weights", flush=True)
        return {"actor_loss": float("nan"), "critic_loss": float("nan"), "entropy": 0.0}

    nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
    actor_opt.step()
    critic_opt.step()

    return {
        "actor_loss": total_actor_loss / n_agents,
        "critic_loss": total_critic_loss / max(num_envs, 1),
        "entropy": total_entropy / n_agents,
    }


def _build_actor_segments(
    buffer: RolloutBuffer,
    team_advantages: list[torch.Tensor],
) -> list[dict]:
    """Split alive actor samples into contiguous recurrent segments."""
    segments = []

    def append_segment(env_idx: int, agent_idx: int, steps: list[int]):
        if not steps:
            return
        indices = np.asarray(steps, dtype=np.int64)
        if indices.size > 1 and not np.all(np.diff(indices) == 1):
            raise RuntimeError("actor recurrent segment contains a time gap")
        segments.append({
            "env_idx": env_idx,
            "agent_idx": agent_idx,
            "steps": indices,
            "initial_hidden": buffer.actor_rnn_states_before[
                indices[0], env_idx, agent_idx].copy(),
            "obs": np.stack([
                buffer.obs[t][env_idx][agent_idx] for t in indices
            ]).astype(np.float32),
            "actions": buffer.actions[indices, env_idx, agent_idx].copy(),
            "old_log_probs": buffer.log_probs[indices, env_idx, agent_idx].copy(),
            "advantages": team_advantages[env_idx][indices].detach(),
        })

    for env_idx in range(buffer.num_envs):
        for agent_idx in range(buffer.num_red):
            current: list[int] = []
            for step in range(buffer.num_steps):
                alive = bool(buffer.alive[step, env_idx, agent_idx])
                valid = bool(buffer.valid_transitions[step, env_idx])
                starts = bool(buffer.actor_sequence_start[step, env_idx, agent_idx])
                previous_episode_done = bool(
                    step > 0 and buffer.episode_dones[step - 1, env_idx])
                if not alive or not valid:
                    append_segment(env_idx, agent_idx, current)
                    current = []
                    continue
                if current and (starts or previous_episode_done
                                or step != current[-1] + 1):
                    append_segment(env_idx, agent_idx, current)
                    current = []
                current.append(step)
            append_segment(env_idx, agent_idx, current)
    return segments


def _recompute_segment_log_probs(
    actor: VanillaActor, segment: dict, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute corrected squashed log-probs from the true start hidden."""
    rnn = torch.as_tensor(
        segment["initial_hidden"], dtype=torch.float32,
        device=device).unsqueeze(0)
    obs = torch.as_tensor(segment["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(
        segment["actions"], dtype=torch.float32, device=device)
    log_probs = []
    policy_entropies = []
    for step in range(obs.shape[0]):
        distribution, rnn = actor(obs[step].unsqueeze(0), rnn)
        log_probs.append(
            distribution.log_prob(actions[step].unsqueeze(0)).sum(dim=-1))
        policy_entropies.append(distribution.base_entropy().sum(dim=-1))
    return torch.cat(log_probs), torch.cat(policy_entropies)


# ==============================================================================
#  Main paper-joint PPO update
# ==============================================================================
def ppo_update(actor, critic, actor_opt, critic_opt, buffer, config, device,
               total_steps: int = 0):
    """Paper joint-reward MAPPO update with one team GAE per environment."""
    num_steps = buffer.num_steps
    num_envs = buffer.num_envs
    team_advantages, team_returns = _compute_joint_gae_by_env(
        buffer, config.gamma, config.gae_lambda, device)

    valid_advantages = [
        team_advantages[env_idx][torch.as_tensor(
            buffer.valid_transitions[:, env_idx], dtype=torch.bool,
            device=device)]
        for env_idx in range(num_envs)
    ]
    valid_advantages = [adv for adv in valid_advantages if adv.numel()]
    valid_transition_count = int(np.count_nonzero(buffer.valid_transitions))
    invalid_transition_count = int(buffer.valid_transitions.size - valid_transition_count)
    if not valid_advantages:
        return {
            "policy_loss": 0.0, "entropy_bonus": 0.0,
            "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
            "ActorUpdateAttempts": 0, "ActorUpdatesApplied": 0,
            "ActorUpdatesSkipped": 0, "CriticUpdateAttempts": 0,
            "CriticUpdatesApplied": 0, "CriticUpdatesSkipped": 0,
            "InvalidTransitionsDropped": invalid_transition_count,
            "UpdateSkipReason": "no_valid_transitions",
        }
    all_adv = torch.cat(valid_advantages)
    adv_std = torch.std(all_adv, correction=0)
    if not torch.isfinite(adv_std) or adv_std <= 1e-8:
        adv_std = torch.tensor(1.0, device=device)
    adv_mean = all_adv.mean()
    team_advantages = [
        (adv - adv_mean) / (adv_std + 1e-8) for adv in team_advantages]

    actor_trajectories = _build_actor_segments(buffer, team_advantages)

    if not actor_trajectories:
        return {
            "policy_loss": 0.0,
            "entropy_bonus": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "ActorUpdateAttempts": 0,
            "ActorUpdatesApplied": 0,
            "ActorUpdatesSkipped": 0,
            "CriticUpdateAttempts": 0,
            "CriticUpdatesApplied": 0,
            "CriticUpdatesSkipped": 0,
            "InvalidTransitionsDropped": invalid_transition_count,
            "UpdateSkipReason": "no_valid_actor_samples",
        }

    critic_pairs = [
        (env_idx, t)
        for env_idx in range(num_envs)
        for t in range(num_steps)
        if buffer.valid_transitions[t, env_idx]
    ]
    critic_states = torch.as_tensor(np.stack([
        buffer.global_states[t][env_idx]
        for env_idx, t in critic_pairs
    ]), dtype=torch.float32, device=device)
    critic_targets = torch.stack([
        team_returns[env_idx][t] for env_idx, t in critic_pairs
    ]).detach()
    entropy_coef = _current_entropy_coef(config, total_steps)
    policy_loss_log = []
    actor_loss_log = []
    critic_loss_log = []
    entropy_log = []
    entropy_bonus_log = []
    actor_update_attempts = 0
    actor_updates_applied = 0
    actor_updates_skipped = 0
    critic_update_attempts = 0
    critic_updates_applied = 0
    critic_updates_skipped = 0

    for _epoch in range(config.n_update_epochs):
        order = np.random.permutation(len(actor_trajectories))
        minibatches = np.array_split(
            order, max(1, min(config.n_minibatches, len(order))))
        for mb in minibatches:
            if len(mb) == 0:
                continue
            actor_opt.zero_grad()
            sample_losses = []
            sample_entropies = []
            for traj_idx in mb:
                traj = actor_trajectories[int(traj_idx)]
                old_log_probs = torch.as_tensor(
                    traj["old_log_probs"], dtype=torch.float32, device=device)
                new_log_probs, policy_entropies = _recompute_segment_log_probs(
                    actor, traj, device)
                ratio = torch.exp(new_log_probs - old_log_probs)
                advantage = traj["advantages"].to(device)
                surrogate = torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - config.clip_epsilon,
                                1.0 + config.clip_epsilon) * advantage)
                sample_losses.append(-surrogate)
                sample_entropies.append(policy_entropies)
            policy_loss = torch.cat(sample_losses).mean()
            policy_entropy = torch.cat(sample_entropies).mean()
            entropy_bonus = entropy_coef * policy_entropy
            actor_loss = policy_loss - entropy_bonus
            actor_update_attempts += 1
            actor_loss.backward()
            if not _grad_has_nan(actor):
                nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                actor_opt.step()
                actor_updates_applied += 1
                policy_loss_log.append(float(policy_loss.item()))
                actor_loss_log.append(float(actor_loss.item()))
                entropy_log.append(float(policy_entropy.item()))
                entropy_bonus_log.append(float(entropy_bonus.item()))
            else:
                actor_updates_skipped += 1

        critic_opt.zero_grad()
        critic_values = critic(critic_states).squeeze(-1)
        critic_loss = F.mse_loss(critic_values, critic_targets)
        critic_update_attempts += 1
        critic_loss.backward()
        if not _grad_has_nan(critic):
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_opt.step()
            critic_updates_applied += 1
            critic_loss_log.append(float(critic_loss.item()))
        else:
            critic_updates_skipped += 1

    return {
        "policy_loss": float(np.mean(policy_loss_log)) if policy_loss_log else float("nan"),
        "entropy_bonus": float(np.mean(entropy_bonus_log)) if entropy_bonus_log else float("nan"),
        "actor_loss": float(np.mean(actor_loss_log)) if actor_loss_log else float("nan"),
        "critic_loss": float(np.mean(critic_loss_log)) if critic_loss_log else float("nan"),
        "entropy": float(np.mean(entropy_log)) if entropy_log else 0.0,
        "ActorUpdateAttempts": actor_update_attempts,
        "ActorUpdatesApplied": actor_updates_applied,
        "ActorUpdatesSkipped": actor_updates_skipped,
        "CriticUpdateAttempts": critic_update_attempts,
        "CriticUpdatesApplied": critic_updates_applied,
        "CriticUpdatesSkipped": critic_updates_skipped,
        "InvalidTransitionsDropped": invalid_transition_count,
        "UpdateSkipReason": "",
    }


def _ppo_update_agent_legacy(actor, critic, actor_opt, critic_opt, buffer, config, device,
                             total_steps: int = 0):
    """Disabled discontinuous-alive-step PPO update; not a training path."""
    raise RuntimeError("legacy discontinuous actor PPO update is disabled")
    num_steps = buffer.num_steps
    num_envs = buffer.rnn_actor_init.shape[0]
    num_red = buffer.rnn_actor_init.shape[1]

    global_obs_by_env = []
    trajectories = []

    for env_idx in range(num_envs):
        global_obs_seq = []
        for t in range(num_steps):
            parts = [buffer.obs[t][env_idx][i] for i in range(num_red)]
            global_obs_seq.append(_global_state_from_local_obs_flats(
                parts, obs_mode=config.obs_mode))
        global_obs_by_env.append(np.stack(global_obs_seq).astype(np.float32))

        for agent_idx in range(num_red):
            t_obs_flat = []
            t_act = []
            t_rew = []
            t_val = []
            t_lp = []
            t_done = []
            alive_steps = []
            for step in range(num_steps):
                if buffer.alive[step, env_idx, agent_idx]:
                    t_obs_flat.append(buffer.obs[step][env_idx][agent_idx])
                    t_act.append(buffer.actions[step, env_idx, agent_idx])
                    t_rew.append(buffer.rewards[step, env_idx, agent_idx])
                    t_val.append(buffer.values[step, env_idx, agent_idx])
                    t_lp.append(buffer.log_probs[step, env_idx, agent_idx])
                    t_done.append(buffer.dones[step, env_idx, agent_idx])
                    alive_steps.append(step)

            if not t_act:
                continue

            bootstrap = float(buffer.bootstrap_values[env_idx, agent_idx])
            rewards = torch.tensor(t_rew, device=device)
            old_values = torch.tensor(t_val + [bootstrap], device=device)
            dones = torch.tensor(t_done, device=device)
            advantages, returns = compute_gae(rewards, old_values, dones,
                                              config.gamma, config.gae_lambda)

            adv_std = advantages.std() if advantages.numel() > 1 else torch.std(
                advantages, correction=0)
            if adv_std <= 1e-8 or torch.isnan(adv_std):
                adv_std = 1.0
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            trajectories.append({
                "env_idx": env_idx,
                "agent_idx": agent_idx,
                "alive_steps": alive_steps,
                "obs": np.stack(t_obs_flat).astype(np.float32),
                "actions": np.stack(t_act).astype(np.float32),
                "old_log_probs": np.asarray(t_lp, dtype=np.float32),
                "advantages": advantages.detach(),
                "returns": returns.detach(),
            })

    if not trajectories:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    actor_losses_log = []
    critic_losses_log = []
    entropies_log = []
    entropy_coef = _current_entropy_coef(config, total_steps)

    for _epoch in range(config.n_update_epochs):
        order = np.random.permutation(len(trajectories))
        minibatches = np.array_split(order, max(1, min(config.n_minibatches, len(order))))

        for mb in minibatches:
            if len(mb) == 0:
                continue

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            actor_losses = []
            critic_losses = []
            entropies = []

            for traj_idx in mb:
                traj = trajectories[int(traj_idx)]
                env_idx = traj["env_idx"]
                agent_idx = traj["agent_idx"]

                rnn_a = torch.as_tensor(
                    buffer.rnn_actor_init[env_idx, agent_idx],
                    dtype=torch.float32, device=device).unsqueeze(0)
                obs = torch.as_tensor(traj["obs"], dtype=torch.float32, device=device)
                acts = torch.as_tensor(traj["actions"], dtype=torch.float32,
                                       device=device)
                old_lp = torch.as_tensor(traj["old_log_probs"],
                                         dtype=torch.float32, device=device)
                adv = traj["advantages"].to(device)
                ret = traj["returns"].to(device)

                new_lps = []
                traj_entropies = []
                for t in range(obs.shape[0]):
                    action_dist, rnn_a = actor(obs[t].unsqueeze(0), rnn_a)
                    new_lps.append(
                        action_dist.log_prob(acts[t].unsqueeze(0)).sum(dim=-1))
                    traj_entropies.append(
                        action_dist.base_entropy().sum(dim=-1).mean())

                new_lp = torch.cat(new_lps)
                ent_avg = torch.stack(traj_entropies).mean()
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(policy_loss - entropy_coef * ent_avg)
                entropies.append(ent_avg.detach())

                gobs = torch.as_tensor(
                    global_obs_by_env[env_idx][traj["alive_steps"]],
                    dtype=torch.float32, device=device)
                values = critic(gobs).squeeze(-1)
                critic_losses.append(F.mse_loss(values, ret))

            actor_loss = torch.stack(actor_losses).mean()
            critic_loss = torch.stack(critic_losses).mean()
            (actor_loss + critic_loss).backward()

            actor_nan = _grad_has_nan(actor)
            critic_nan = _grad_has_nan(critic)
            if actor_nan or critic_nan:
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                print(f"  [WARN] NaN gradient detected (actor={actor_nan}, "
                      f"critic={critic_nan}) - skipping minibatch", flush=True)
                continue

            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            actor_opt.step()
            critic_opt.step()

            actor_losses_log.append(float(actor_loss.item()))
            critic_losses_log.append(float(critic_loss.item()))
            entropies_log.append(float(torch.stack(entropies).mean().item()))

    if not actor_losses_log:
        return {"actor_loss": float("nan"), "critic_loss": float("nan"), "entropy": 0.0}

    return {
        "actor_loss": float(np.mean(actor_losses_log)),
        "critic_loss": float(np.mean(critic_losses_log)),
        "entropy": float(np.mean(entropies_log)),
    }


def parse_args():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Train vanilla MAPPO baseline for my_uav_env.")
    parser.add_argument("--preset", type=str, default=None,
                        help="Experiment preset name (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", default=False,
                        help="List available presets and exit")
    parser.add_argument("--num-red", type=int, default=defaults.num_red)
    parser.add_argument("--num-blue", type=int, default=defaults.num_blue)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--total-env-steps", type=int,
                        default=defaults.total_env_steps)
    parser.add_argument("--max-episode-length", type=int,
                        default=defaults.max_episode_length)
    parser.add_argument("--replay-buffer-size", type=int,
                        default=defaults.replay_buffer_size)
    parser.add_argument("--n-minibatches", type=int,
                        default=defaults.n_minibatches)
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--entropy-coef", type=float,
                        default=defaults.entropy_coef)
    parser.add_argument("--mlp-hidden", type=int, default=defaults.mlp_hidden)
    parser.add_argument("--rnn-hidden-size", type=int,
                        default=defaults.rnn_hidden_size)
    parser.add_argument("--blue-policy-profile",
        choices=(PAPER_BLUE_POLICY_PROFILE,),
        default=defaults.blue_policy_profile)
    parser.add_argument("--environment-profile",
        choices=(PAPER_ENVIRONMENT_PROFILE,),
        default=defaults.environment_profile)
    parser.add_argument("--obs-mode", type=str,
                        choices=("paper_strict",),
                        default=defaults.obs_mode)
    parser.add_argument("--obs-normalization", type=str,
                        choices=("paper_fixed_v1", "none"),
                        default=defaults.obs_normalization)
    parser.add_argument("--pid-profile", type=str,
                        choices=(PAPER_PID_PROFILE,),
                        default=defaults.pid_profile)
    parser.add_argument("--pid-throttle-base", type=float,
                        default=defaults.pid_throttle_base)
    parser.add_argument("--reward-mode", type=str,
                        choices=(PAPER_REWARD_MODE,),
                        default=defaults.reward_mode)
    parser.add_argument("--missile-guidance-mode", type=str,
                        choices=(PAPER_MISSILE_GUIDANCE_MODE,),
                        default=defaults.missile_guidance_mode)
    parser.add_argument("--initial-condition-randomization-mode",
                        choices=("deterministic_v1",),
                        default=defaults.initial_condition_randomization_mode)
    parser.add_argument("--resume-from-best", action="store_true",
                        default=defaults.resume_from_best)
    parser.add_argument("--resume-latest", action="store_true",
                        default=defaults.resume_latest)
    parser.add_argument("--resume-state", type=str, default=defaults.resume_state)
    parser.add_argument("--overwrite-existing", action="store_true",
                        default=defaults.overwrite_existing)
    parser.add_argument("--eval-during-training", action="store_true",
                        default=defaults.eval_during_training)
    parser.add_argument("--no-eval-during-training", action="store_false",
                        dest="eval_during_training")
    parser.add_argument("--eval-interval-steps", type=int,
                        default=defaults.eval_interval_steps)
    parser.add_argument("--eval-episodes", type=int,
                        default=defaults.eval_episodes)
    parser.add_argument("--eval-log-file", type=str,
                        default=defaults.eval_log_file)
    parser.add_argument("--log-file", type=str, default=defaults.log_file)
    parser.add_argument("--results-file", type=str,
                        default=defaults.results_file)
    parser.add_argument("--launch-quality-file", type=str,
                        default=defaults.launch_quality_file)
    parser.add_argument("--extreme-load-trace-file", type=str,
                        default=defaults.extreme_load_trace_file)
    parser.add_argument("--checkpoint-dir", type=str,
                        default=defaults.checkpoint_dir)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"),
                        default=defaults.device)
    return parser.parse_args()


_VANILLA_PRESET_CLI_FLAGS = {
    "num_red", "num_blue", "num_envs", "total_env_steps",
    "max_episode_length", "replay_buffer_size", "n_minibatches",
    "actor_lr", "critic_lr", "entropy_coef", "mlp_hidden", "rnn_hidden_size",
    "blue_policy_profile", "environment_profile",
    "obs_mode", "obs_normalization", "pid_profile",
    "pid_throttle_base",
    "reward_mode", "missile_guidance_mode", "resume_from_best",
    "initial_condition_randomization_mode", "resume_latest", "resume_state",
    "overwrite_existing", "eval_during_training", "eval_interval_steps",
    "eval_episodes", "eval_log_file",
    "log_file", "results_file", "launch_quality_file",
    "extreme_load_trace_file",
    "checkpoint_dir", "seed", "device",
}

_FORMAL_PAPER_3V3_PRESETS = (
    "vanilla_3v3_paper_smoke",
    "vanilla_3v3_paper_main",
    "vanilla_3v3_paper_100k_diag",
)


def _apply_preset_vanilla(args, preset: dict):
    """Apply preset values to args for any key not explicitly given on CLI."""
    cli_flags = set()
    for item in sys.argv:
        if item.startswith("--"):
            name = item.lstrip("-").replace("-", "_")
            cli_flags.add(name)
    for key, value in preset.items():
        if key not in _VANILLA_PRESET_CLI_FLAGS:
            continue
        if key == "device":
            flag = "device"
        else:
            flag = key
        if flag not in cli_flags:
            setattr(args, key, value)


def _validate_preset_resume_semantics(args) -> None:
    return


def make_config_from_args(args) -> Config:
    config = Config()
    config.num_red = args.num_red
    config.num_blue = args.num_blue
    config.num_envs = args.num_envs
    config.total_env_steps = args.total_env_steps
    config.max_episode_length = args.max_episode_length
    config.replay_buffer_size = args.replay_buffer_size
    config.n_minibatches = args.n_minibatches
    config.actor_lr = args.actor_lr
    config.critic_lr = args.critic_lr
    config.entropy_coef = args.entropy_coef
    config.mlp_hidden = args.mlp_hidden
    config.rnn_hidden_size = args.rnn_hidden_size
    config.blue_policy_profile = args.blue_policy_profile
    config.environment_profile = args.environment_profile
    config.environment_version = args.environment_profile
    config.altitude_reward_config = _minimal_altitude_reward_config()
    config.obs_mode = args.obs_mode
    config.obs_normalization = args.obs_normalization
    config.pid_profile = args.pid_profile
    config.pid_throttle_base = args.pid_throttle_base
    config.reward_mode = args.reward_mode
    config.missile_guidance_mode = args.missile_guidance_mode
    config.initial_condition_randomization_mode = (
        args.initial_condition_randomization_mode)
    config.resume_from_best = args.resume_from_best
    config.resume_latest = args.resume_latest
    config.resume_state = args.resume_state
    config.overwrite_existing = args.overwrite_existing
    config.eval_during_training = args.eval_during_training
    config.eval_interval_steps = args.eval_interval_steps
    config.eval_episodes = args.eval_episodes
    config.eval_log_file = args.eval_log_file
    config.log_file = args.log_file
    config.results_file = args.results_file
    config.launch_quality_file = args.launch_quality_file
    config.extreme_load_trace_file = args.extreme_load_trace_file
    config.checkpoint_dir = args.checkpoint_dir
    config.seed = args.seed
    config.device = args.device
    return config


def _set_main_process_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] --device cuda requested but CUDA is unavailable; "
              "falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _default_launch_quality_file(results_file: str) -> str:
    results_dir = os.path.dirname(results_file) or "results"
    stem = os.path.splitext(os.path.basename(results_file))[0]
    return os.path.join(results_dir, f"{stem}_launch_quality.csv")


def _default_eval_log_file(results_file: str) -> str:
    results_dir = os.path.dirname(results_file) or "results"
    stem = os.path.splitext(os.path.basename(results_file))[0]
    return os.path.join(results_dir, f"{stem}_eval.csv")


def _default_extreme_load_trace_file(results_file: str) -> str:
    results_dir = os.path.dirname(results_file) or "results"
    stem = os.path.splitext(os.path.basename(results_file))[0]
    return os.path.join(results_dir, f"{stem}_extreme_load_traces.jsonl")


EVAL_LOG_FIELDS = (
    "Iteration", "Step", "FinalCheckpoint", "Episodes", "RedWins",
    "BlueWins", "Draws", "RedWinRate", "MeanRedReward",
    "EnvironmentProfile", "EnvironmentConfigFingerprint",
    "ActionDistribution", "EntropyEstimator",
    "MeanRedAlive", "MeanBlueAlive", "RedMissilesFired",
    "BlueMissilesFired", "RedMissileHits", "BlueMissileHits",
    "MissileLifetimeMeanS", "RedMWSDetectedAgentDecisions",
    "RedMWSOverrideAgentDecisions", "BlueMWSDetectedAgentDecisions",
    "BlueMWSOverrideAgentDecisions", "InvalidEpisodes",
)


def _run_periodic_evaluation(
    actor, config, device, checkpoint_meta: dict, iteration: int,
    total_steps: int, final_checkpoint: bool,
) -> dict:
    from evaluate_vanilla_mappo import run_one_episode

    rng_state = _capture_rng_state()
    was_training = actor.training
    actor.eval()
    try:
        rows = []
        base_seed = int(config.seed or 0) + 1_000_000
        for episode in range(config.eval_episodes):
            rows.append(run_one_episode(
                actor=actor,
                rnn_hidden_size=config.rnn_hidden_size,
                num_red=config.num_red,
                num_blue=config.num_blue,
                max_steps=config.max_episode_length,
                device=device,
                episode_idx=episode + 1,
                obs_mode=config.obs_mode,
                obs_normalization=config.obs_normalization,
                pid_profile=config.pid_profile,
                pid_throttle_base=config.pid_throttle_base,
                reward_mode=config.reward_mode,
                missile_guidance_mode=config.missile_guidance_mode,
                blue_policy_profile=config.blue_policy_profile,
                environment_profile=config.environment_profile,
                initial_condition_randomization_mode=(
                    config.initial_condition_randomization_mode),
                seed=base_seed + episode,
                deterministic=True))
        red_wins = sum(int(row["RedWin"]) for row in rows)
        blue_wins = sum(int(row["BlueWin"]) for row in rows)
        draws = sum(int(row["Draw"]) for row in rows)
        result = {
            "Iteration": int(iteration),
            "Step": int(total_steps),
            "FinalCheckpoint": int(final_checkpoint),
            "Episodes": len(rows),
            "RedWins": red_wins,
            "BlueWins": blue_wins,
            "Draws": draws,
            "RedWinRate": _safe_div(red_wins, len(rows)),
            "MeanRedReward": float(np.mean([
                float(row["EpisodeRewardRed"]) for row in rows])) if rows else 0.0,
            "EnvironmentProfile": config.environment_profile,
            "EnvironmentConfigFingerprint": checkpoint_meta[
                "environment_config_fingerprint"],
            "ActionDistribution": ACTION_DISTRIBUTION_VERSION,
            "EntropyEstimator": ENTROPY_ESTIMATOR_VERSION,
            "MeanRedAlive": float(np.mean([
                float(row.get("RedAlive", 0)) for row in rows])) if rows else 0.0,
            "MeanBlueAlive": float(np.mean([
                float(row.get("BlueAlive", 0)) for row in rows])) if rows else 0.0,
            "RedMissilesFired": sum(float(row.get("RedMissilesFired", 0)) for row in rows),
            "BlueMissilesFired": sum(float(row.get("BlueMissilesFired", 0)) for row in rows),
            "RedMissileHits": sum(float(row.get("RedMissileHits", 0)) for row in rows),
            "BlueMissileHits": sum(float(row.get("BlueMissileHits", 0)) for row in rows),
            "MissileLifetimeMeanS": _mean_or_zero([
                float(row["MissileLifetimeMeanS"])
                for row in rows if row.get("MissileLifetimeMeanS") is not None]),
            "RedMWSDetectedAgentDecisions": sum(int(
                row.get("RedMWSDetectedAgentDecisions", 0)) for row in rows),
            "RedMWSOverrideAgentDecisions": sum(int(
                row.get("RedMWSOverrideAgentDecisions", 0)) for row in rows),
            "BlueMWSDetectedAgentDecisions": sum(int(
                row.get("BlueMWSDetectedAgentDecisions", 0)) for row in rows),
            "BlueMWSOverrideAgentDecisions": sum(int(
                row.get("BlueMWSOverrideAgentDecisions", 0)) for row in rows),
            "InvalidEpisodes": sum(int(
                row.get("InvalidNumericalEpisode", 0)) for row in rows),
        }
        exists = os.path.exists(config.eval_log_file)
        with open(config.eval_log_file, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVAL_LOG_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(result)
            handle.flush()
            os.fsync(handle.fileno())
        return result
    finally:
        actor.train(was_training)
        _restore_rng_state(rng_state)


def main():
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGINT, _request_safe_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_safe_stop)
    args = parse_args()

    if args.list_presets:
        print("Available presets:")
        for name in _FORMAL_PAPER_3V3_PRESETS:
            print(f"  {name}")
        return

    if args.preset is not None:
        if args.preset not in _FORMAL_PAPER_3V3_PRESETS:
            choices = ", ".join(_FORMAL_PAPER_3V3_PRESETS)
            raise ValueError(
                f"formal vanilla MAPPO only accepts paper_3v3_v1 presets: {choices}"
            )
        from configs.experiment_presets import get_preset
        preset = get_preset(args.preset)
        _apply_preset_vanilla(args, preset)
    _validate_preset_resume_semantics(args)

    config = make_config_from_args(args)
    if config.launch_quality_file is None:
        config.launch_quality_file = _default_launch_quality_file(config.results_file)
    if config.eval_log_file is None:
        config.eval_log_file = _default_eval_log_file(config.results_file)
    if config.extreme_load_trace_file is None:
        config.extreme_load_trace_file = _default_extreme_load_trace_file(
            config.results_file)
    _set_main_process_seed(config.seed)
    device = _select_device(config.device)

    # 计算展平观测维度 (红方视角)
    obs_dim = _compute_obs_dim(
        config.num_red, config.num_blue, is_red=True,
        obs_mode=config.obs_mode)
    global_obs_dim = _compute_global_state_dim(config.num_red, config.obs_mode)
    rollout_layout = _rollout_layout(config.replay_buffer_size, config.num_envs)
    checkpoint_meta = _checkpoint_metadata(config, obs_dim, global_obs_dim)
    resume_path = config.resume_state
    if config.resume_latest:
        if resume_path is not None:
            raise ValueError("use only one of --resume-latest and --resume-state")
        resume_path = os.path.join(
            config.checkpoint_dir, "latest_training_state.pt")
    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False)
        _validate_training_state(resume_payload, config, checkpoint_meta)

    # ---- 持久化：创建 checkpoint 目录 ----
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if (resume_payload is None and os.path.exists(config.results_file)
            and not config.overwrite_existing):
        raise FileExistsError(
            f"results file already exists: {config.results_file}")
    if (config.eval_during_training and resume_payload is None
            and os.path.exists(config.eval_log_file)
            and not config.overwrite_existing):
        raise FileExistsError(f"evaluation log already exists: {config.eval_log_file}")
    if (config.eval_during_training and resume_payload is not None
            and os.path.exists(config.eval_log_file)):
        _validate_resume_csv_header(
            config.eval_log_file, list(EVAL_LOG_FIELDS))
    manifest_path = os.path.join(config.checkpoint_dir, "run_manifest.json")
    if (resume_payload is None and os.path.exists(manifest_path)
            and not config.overwrite_existing):
        raise FileExistsError(
            f"checkpoint run manifest already exists: {manifest_path}")
    if resume_payload is not None and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["target_total_env_steps"] = int(config.total_env_steps)
    else:
        manifest = {
            "run_id": f"vanilla-{time.time_ns()}",
            "seed": config.seed,
            "start_unix_time": time.time(),
            "code_version": CHECKPOINT_SCHEMA_VERSION,
            "training_state_schema": TRAINING_STATE_SCHEMA_VERSION,
            "training_log_schema": _training_log_fields(),
            "environment_config_fingerprint": checkpoint_meta[
                "environment_config_fingerprint"],
            "core_config": _training_core_config(config, checkpoint_meta),
            "target_total_env_steps": int(config.total_env_steps),
            "resume_count": 0,
        }
    _atomic_json_save(manifest, manifest_path)

    # ---- 持久化：CSV 日志 ----
    _ensure_parent_dir(config.log_file)
    train_log_fields = _training_log_fields()
    if resume_payload is not None:
        _validate_resume_csv(
            config.log_file, train_log_fields,
            int(resume_payload["runtime"]["total_steps"]))
    elif os.path.exists(config.log_file) and not config.overwrite_existing:
        raise FileExistsError(
            f"training log already exists; use --resume-latest, --resume-state, "
            f"or --overwrite-existing: {config.log_file}")
    csv_file = open(
        config.log_file, "a" if resume_payload is not None else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if resume_payload is None:
        csv_writer.writerow(train_log_fields)
    csv_file.flush()

    _ensure_parent_dir(config.extreme_load_trace_file)
    if (resume_payload is None
            and os.path.exists(config.extreme_load_trace_file)
            and not config.overwrite_existing):
        raise FileExistsError(
            "extreme-load trace file already exists: "
            f"{config.extreme_load_trace_file}")
    if resume_payload is None:
        open(config.extreme_load_trace_file, "w", encoding="utf-8").close()

    launch_quality_file = config.launch_quality_file
    launch_quality_csv_file = None
    launch_quality_writer = None
    if launch_quality_file and str(launch_quality_file).lower() not in ("none", "off", "false"):
        _ensure_parent_dir(launch_quality_file)
        if resume_payload is not None:
            _validate_resume_csv_header(
                launch_quality_file, list(LAUNCH_QUALITY_DETAIL_FIELDS))
        elif os.path.exists(launch_quality_file) and not config.overwrite_existing:
            raise FileExistsError(
                f"launch-quality log already exists: {launch_quality_file}")
        launch_quality_csv_file = open(
            launch_quality_file,
            "a" if resume_payload is not None else "w", newline="")
        launch_quality_writer = csv.DictWriter(
            launch_quality_csv_file, fieldnames=LAUNCH_QUALITY_DETAIL_FIELDS)
        if resume_payload is None:
            launch_quality_writer.writeheader()
        launch_quality_csv_file.flush()

    print(f"设备: {device}")
    print("Final config:")
    print(f"  num_red / num_blue: {config.num_red} / {config.num_blue}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  total_env_steps: {config.total_env_steps}")
    print(f"  max_episode_length: {config.max_episode_length}")
    print(f"  replay_buffer_size: {config.replay_buffer_size}")
    print(f"  requested_replay_buffer_size: {rollout_layout['requested_replay_buffer_size']}")
    print(f"  rollout_horizon_per_env: {rollout_layout['rollout_horizon_per_env']}")
    print(f"  transitions_per_update: {rollout_layout['transitions_per_update']}")
    print(f"  unused_replay_slots: {rollout_layout['unused_replay_slots']}")
    if rollout_layout["unused_replay_slots"] > 0:
        print("[WARN] replay_buffer_size is not divisible by num_envs; "
              f"{rollout_layout['unused_replay_slots']} requested slots are unused "
              "because rollout_horizon_per_env uses integer division.",
              flush=True)
    print(f"  log_file: {config.log_file}")
    print(f"  results_file: {config.results_file}")
    print(f"  launch_quality_file: {launch_quality_file}")
    print(f"  extreme_load_trace_file: {config.extreme_load_trace_file}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  seed: {config.seed}")
    print(f"  device: {device}")
    print("  reward_version: paper_3v3_joint_eq15_23_v1")
    print(f"  environment_profile: {config.environment_profile}")
    print(f"  obs_mode: {config.obs_mode}")
    print(f"  obs_normalization: {config.obs_normalization}")
    print(f"  pid_profile: {config.pid_profile}")
    print(f"  pid_throttle_base: {config.pid_throttle_base}")
    print(f"  reward_mode: {config.reward_mode}")
    print(f"  missile_guidance_mode: {config.missile_guidance_mode}")
    print(f"  action_distribution: {ACTION_DISTRIBUTION_VERSION}")
    print(f"  checkpoint_schema: {CHECKPOINT_SCHEMA_VERSION}")
    print("  altitude_reward_config: "
          f"{json.dumps(asdict(config.altitude_reward_config), sort_keys=True)}")
    print(f"架构: Vanilla MLP + GRU (无注意力, 无掩码)")
    print(f"场景: {config.num_red}v{config.num_blue} (红方 RL, 蓝方规则)")
    print(f"展平 obs 维度: {obs_dim}")
    print(f"buffer: {config.replay_buffer_size} 步 ({config.num_envs} env × "
          f"{config.replay_buffer_size // config.num_envs} steps)")
    print(f"MLP hidden: {config.mlp_hidden},  RNN hidden: {config.rnn_hidden_size}")
    print(f"CSV 日志: {config.log_file}")
    print(f"模型存档: {config.checkpoint_dir}/ (每 10 iter, 保留最新 5 个)")

    # ---- 评估准入准则 ----
    MIN_EPISODES_TO_EVAL = 50  # 最少完成 50 局后才允许覆盖 best 模型

    # ---- 1. 创建并行环境 ----
    num_steps = rollout_layout["rollout_horizon_per_env"]
    env_kwargs = dict(max_num_blue=config.num_blue, max_num_red=config.num_red,
                      max_steps=config.max_episode_length,
                      environment_profile=config.environment_profile,
                      initial_condition_randomization_mode=(
                          config.initial_condition_randomization_mode),
                      obs_mode=config.obs_mode,
                      pid_profile=config.pid_profile,
                      pid_throttle_base=config.pid_throttle_base,
                      reward_mode=config.reward_mode,
                      missile_guidance_mode=config.missile_guidance_mode,
                      altitude_reward_config=config.altitude_reward_config,
                      blue_policy_profile=config.blue_policy_profile)
    print(f"正在启动 {config.num_envs} 个 worker 进程...", flush=True)
    vec_env = SubprocVecEnv(config.num_envs, env_kwargs, base_seed=config.seed)

    red_ids = [f"red_{i}" for i in range(config.num_red)]
    blue_ids = [f"blue_{i}" for i in range(config.num_blue)]

    # ---- 2. 初始化网络 ----
    actor = VanillaActor(obs_dim=obs_dim, action_dim=config.action_dim,
                         hidden=config.mlp_hidden,
                         rnn_hidden=config.rnn_hidden_size).to(device)
    global_obs_dim = _compute_global_state_dim(config.num_red, config.obs_mode)
    critic = CentralizedCritic(global_obs_dim=global_obs_dim,
                                hidden=config.mlp_hidden).to(device)

    print(f"Actor  params:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params:  {sum(p.numel() for p in critic.parameters()):,}  "
          f"(centralized, global_obs_dim={global_obs_dim})")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # ---- 3. 从 best checkpoint 恢复训练 (消融 r_ceil 后重新起航) ----
    actor_best_path = os.path.join(config.checkpoint_dir, "vanilla_actor_best.pt")
    critic_best_path = os.path.join(config.checkpoint_dir, "centralized_critic_best.pt")
    if resume_payload is not None:
        actor.load_state_dict(resume_payload["actor_state_dict"])
        critic.load_state_dict(resume_payload["critic_state_dict"])
        actor_opt.load_state_dict(resume_payload["actor_optimizer_state_dict"])
        critic_opt.load_state_dict(resume_payload["critic_optimizer_state_dict"])
        print(f"[OK] loaded full training state: {resume_path}")
    elif (config.resume_from_best and os.path.exists(actor_best_path)
            and os.path.exists(critic_best_path)):
        actor_payload = torch.load(
            actor_best_path, map_location=device, weights_only=False)
        critic_payload = torch.load(
            critic_best_path, map_location=device, weights_only=False)
        actor.load_state_dict(_unpack_and_validate_checkpoint(
            actor_payload, checkpoint_meta, "actor"))
        critic.load_state_dict(_unpack_and_validate_checkpoint(
            critic_payload, checkpoint_meta, "critic"))
        print(f"[OK] 已加载 best checkpoint 权重 (actor + critic)")
    else:
        print(f"[WARN] best checkpoint 不存在，使用随机初始化权重")

    # ---- 4. 初始 RNN 状态 ----
    rnn_hidden_actor = np.zeros(
        (config.num_envs, config.num_red, config.rnn_hidden_size), dtype=np.float32)
    actor_reset_before = np.ones(
        (config.num_envs, config.num_red), dtype=bool)

    # ---- 5. 重置 ----
    print(f"正在重置 {config.num_envs} 个环境...", flush=True)
    t_reset = time.perf_counter()
    raw_obs_list = vec_env.reset(timeout=300.0)
    print(f"重置完成 ({time.perf_counter() - t_reset:.0f}s)", flush=True)
    print("=" * 70)

    # ---- 6. 训练循环 ----
    total_steps = 0
    iteration = 1
    total_episodes = 0
    red_wins = 0
    blue_wins = 0
    draws = 0
    invalid_numerical_episodes = 0
    death_stats = {"red": Counter(), "blue": Counter()}
    red_missiles_total = 0.0
    blue_missiles_total = 0.0
    best_reward_value = -float("inf")
    best_reward_win_rate = 0.0
    best_winrate_value = -float("inf")
    best_winrate_reward = -float("inf")

    # Episodic reward trackers — only fully-completed episodes contribute (Red only)
    recent_ep_rewards_red = deque(maxlen=50)
    # Per-component episodic trackers for red team diagnostics
    COMP_KEYS = EPISODE_REWARD_COMPONENT_KEYS
    recent_ep_comps_red: deque[dict] = deque(maxlen=50)
    recent_ep_missiles_red = deque(maxlen=50)
    recent_ep_missiles_blue = deque(maxlen=50)
    recent_ep_red_alive = deque(maxlen=50)
    recent_ep_blue_alive = deque(maxlen=50)
    current_ep_reward_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_comp_red = {k: np.zeros(config.num_envs, dtype=np.float64)
                           for k in COMP_KEYS}
    current_ep_missiles_red = np.zeros(config.num_envs, dtype=np.float32)
    current_ep_missiles_blue = np.zeros(config.num_envs, dtype=np.float32)
    # Results log for offline plotting — accumulates all key metrics per iteration
    results_log: list[dict] = []
    cumulative_environment_diag = Counter()
    cumulative_missile_term = Counter()
    last_eval_steps = -1
    if resume_payload is not None:
        runtime = resume_payload["runtime"]
        total_steps = int(runtime["total_steps"])
        iteration = int(runtime["iteration"])
        total_episodes = int(runtime["total_episodes"])
        red_wins = int(runtime["red_wins"])
        blue_wins = int(runtime["blue_wins"])
        draws = int(runtime["draws"])
        invalid_numerical_episodes = int(runtime["invalid_numerical_episodes"])
        death_stats = {
            "red": Counter(runtime["death_stats"]["red"]),
            "blue": Counter(runtime["death_stats"]["blue"]),
        }
        red_missiles_total = float(runtime["red_missiles_total"])
        blue_missiles_total = float(runtime["blue_missiles_total"])
        best_reward_value = float(runtime["best_reward_value"])
        best_reward_win_rate = float(runtime["best_reward_win_rate"])
        best_winrate_value = float(runtime["best_winrate_value"])
        best_winrate_reward = float(runtime["best_winrate_reward"])
        recent_ep_rewards_red.extend(runtime.get("recent_ep_rewards_red", []))
        recent_ep_comps_red.extend(runtime.get("recent_ep_comps_red", []))
        recent_ep_missiles_red.extend(runtime.get("recent_ep_missiles_red", []))
        recent_ep_missiles_blue.extend(runtime.get("recent_ep_missiles_blue", []))
        recent_ep_red_alive.extend(runtime.get("recent_ep_red_alive", []))
        recent_ep_blue_alive.extend(runtime.get("recent_ep_blue_alive", []))
        cumulative_environment_diag.update(
            runtime.get("cumulative_environment_diag", {}))
        cumulative_missile_term.update(runtime.get("cumulative_missile_term", {}))
        last_eval_steps = int(runtime.get("last_eval_steps", -1))
        vec_env.worker_restart_count += int(
            runtime.get("worker_restart_count", 0))
        if os.path.exists(config.results_file):
            with open(config.results_file, newline="") as handle:
                results_log = list(csv.DictReader(handle))
        _restore_rng_state(resume_payload["rng_state"])

    manifest["worker_restart_count"] = int(vec_env.worker_restart_count)
    _atomic_json_save(manifest, manifest_path)

    def _runtime_state(next_iteration: int) -> dict:
        return {
            "run_id": manifest["run_id"],
            "total_steps": int(total_steps),
            "iteration": int(next_iteration),
            "total_episodes": int(total_episodes),
            "red_wins": int(red_wins),
            "blue_wins": int(blue_wins),
            "draws": int(draws),
            "invalid_numerical_episodes": int(invalid_numerical_episodes),
            "death_stats": {
                "red": dict(death_stats["red"]),
                "blue": dict(death_stats["blue"]),
            },
            "red_missiles_total": float(red_missiles_total),
            "blue_missiles_total": float(blue_missiles_total),
            "best_reward_value": float(best_reward_value),
            "best_reward_win_rate": float(best_reward_win_rate),
            "best_winrate_value": float(best_winrate_value),
            "best_winrate_reward": float(best_winrate_reward),
            "recent_ep_rewards_red": list(recent_ep_rewards_red),
            "recent_ep_comps_red": list(recent_ep_comps_red),
            "recent_ep_missiles_red": list(recent_ep_missiles_red),
            "recent_ep_missiles_blue": list(recent_ep_missiles_blue),
            "recent_ep_red_alive": list(recent_ep_red_alive),
            "recent_ep_blue_alive": list(recent_ep_blue_alive),
            "cumulative_environment_diag": dict(cumulative_environment_diag),
            "cumulative_missile_term": dict(cumulative_missile_term),
            "last_eval_steps": int(last_eval_steps),
            "worker_restart_count": int(vec_env.worker_restart_count),
            "resume_starts_new_episode": True,
            "ResumeEnvironmentReset": True,
        }

    while total_steps < config.total_env_steps:
        t_start = time.perf_counter()

        buffer = RolloutBuffer(
            num_steps=num_steps, num_envs=config.num_envs,
            num_red=config.num_red, action_dim=config.action_dim,
            rnn_hidden_size=config.rnn_hidden_size,
        )
        # Per-iteration episode counters (for sliding-window win rate)
        iter_episodes = 0
        iter_red_wins = 0
        iter_launch_diag = _empty_launch_diag_totals()
        iter_launch_quality_records: list[dict] = []
        iter_launch_quality_done_records: list[dict] = []
        iter_action_bound = _empty_action_bound_totals()
        iter_blue_policy_diag = Counter()
        iter_environment_diag = Counter()
        iter_episode_lengths: list[int] = []
        iter_episode_first_events: list[dict] = []
        iter_aircraft_diag_records: list[dict] = []
        iter_episode_physics_frames = 0
        iter_invalid_episodes_dropped = 0
        episode_start_in_rollout = np.zeros(config.num_envs, dtype=np.int64)

        # ---- Rollout ----
        for step in range(num_steps):
            blue_calls = []
            for env_obs in raw_obs_list:
                blue_obs = ({bid: env_obs[bid] for bid in blue_ids}
                            if env_obs else {})
                blue_calls.append(((blue_obs,), {}))
            blue_actions_list = vec_env.env_method_each(
                "blue_policy_actions", blue_calls)

            # ---- Phase 1: per-env blue actions + collect red obs for batching ----
            # env_actions_builders stores per-env data needed to finalize actions after batched inference
            env_actions_builders = []  # list of dicts
            # Batched actor inputs
            all_red_obs_flat = []        # list of np.ndarray for alive agents across ALL envs
            all_rnn_hidden_in = []       # corresponding RNN hidden states
            all_sequence_start = []      # segment start before current action
            alive_map = []               # list of (env_idx, agent_idx) tuples for split-back
            # Batched critic inputs
            all_global_obs_np = []       # per-env global obs for centralized critic batch
            # Per-env dead-agent records (store_step for dead agents)
            env_dead_records = []        # per-env list of (agent_idx, obs_flat)

            for env_idx in range(config.num_envs):
                env_obs = raw_obs_list[env_idx]
                if not env_obs or len(env_obs) == 0:
                    env_actions_builders.append({"dead": True})
                    all_global_obs_np.append(np.zeros(global_obs_dim, dtype=np.float32))
                    env_dead_records.append([])
                    rnn_hidden_actor[env_idx].fill(0.0)
                    actor_reset_before[env_idx].fill(True)
                    continue

                # ---- Blue rule-based actions (fast CPU, per-env) ----
                blue_actions = blue_actions_list[env_idx]

                # ---- Red: collect alive agents for batched actor forward pass ----
                red_obs_flat_all = []
                alive_agent_indices = []
                dead_agent_records = []
                for i, rid in enumerate(red_ids):
                    obs_np = env_obs[rid]
                    obs_flat = _flatten_obs(
                        obs_np, obs_mode=config.obs_mode,
                        obs_normalization=config.obs_normalization)
                    red_obs_flat_all.append(obs_flat)
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        if actor_reset_before[env_idx, i]:
                            rnn_hidden_actor[env_idx, i].fill(0.0)
                        all_red_obs_flat.append(obs_flat)
                        all_rnn_hidden_in.append(
                            rnn_hidden_actor[env_idx, i].copy())
                        all_sequence_start.append(
                            bool(step == 0 or actor_reset_before[env_idx, i]))
                        alive_map.append((env_idx, i))
                        alive_agent_indices.append(i)
                    else:
                        rnn_hidden_actor[env_idx, i].fill(0.0)
                        actor_reset_before[env_idx, i] = True
                        dead_agent_records.append((i, obs_flat))

                all_global_obs_np.append(_global_state_from_local_obs_flats(
                    red_obs_flat_all, obs_mode=config.obs_mode))
                env_actions_builders.append({
                    "dead": False,
                    "blue_actions": blue_actions,
                    "alive_indices": alive_agent_indices,
                })
                env_dead_records.append(dead_agent_records)

            # ---- Phase 2: batched actor forward pass (all alive agents, all envs) ----
            if all_red_obs_flat:
                obs_batch = torch.as_tensor(
                    np.stack(all_red_obs_flat), dtype=torch.float32, device=device)
                batch_rnn_a = torch.as_tensor(
                    np.stack(all_rnn_hidden_in), dtype=torch.float32, device=device)

                with torch.no_grad():
                    action_dist, new_rnn_a = actor(obs_batch, batch_rnn_a)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                action_stds_np = action_dist.scale.cpu().numpy()
                policy_means_np = action_dist.mode.cpu().numpy()

                actions_np = action.cpu().numpy()
                env_actions_np = actions_np
                log_probs_np = log_prob.cpu().numpy()
                new_rnn_np = new_rnn_a.cpu().numpy()
            else:
                actions_np = np.array([])
                env_actions_np = np.array([])
                policy_means_np = np.array([])
                log_probs_np = np.array([])
                new_rnn_np = np.array([])

            # ---- Phase 3: batched centralized critic forward pass (all envs) ----
            global_obs_batch = torch.as_tensor(
                np.stack(all_global_obs_np), dtype=torch.float32, device=device)
            with torch.no_grad():
                v_global_all = critic(global_obs_batch).squeeze(-1).cpu().numpy()  # [num_envs]

            # ---- Phase 4: build per-env action dicts and fill buffer ----
            actions_list = []
            for env_idx in range(config.num_envs):
                builder = env_actions_builders[env_idx]
                v_global = float(v_global_all[env_idx])
                buffer.global_states[step][env_idx] = np.asarray(
                    all_global_obs_np[env_idx], dtype=np.float32)
                buffer.team_values[step, env_idx] = v_global

                if builder.get("dead"):
                    env_actions = {aid: np.zeros(config.action_dim, dtype=np.float32)
                                   for aid in red_ids + blue_ids}
                    actions_list.append(env_actions)
                    continue

                env_actions = dict(builder["blue_actions"])

                # Dead agents: zero actions, store to buffer
                for agent_idx_i, obs_flat in env_dead_records[env_idx]:
                    rid = red_ids[agent_idx_i]
                    env_actions[rid] = np.zeros(config.action_dim, dtype=np.float32)
                    buffer.store_step(step, env_idx, agent_idx_i, obs_flat,
                                      np.zeros(config.action_dim, dtype=np.float32),
                                      0.0, alive=False,
                                      actor_rnn_state_before=np.zeros(
                                          config.rnn_hidden_size, dtype=np.float32),
                                      sequence_start=True)

                # Alive agents: assign batched actions, store to buffer
                for k, (batch_env, agent_idx_i) in enumerate(alive_map):
                    if batch_env != env_idx:
                        continue
                    rid = red_ids[agent_idx_i]
                    env_actions[rid] = env_actions_np[k]
                    rnn_hidden_actor[env_idx, agent_idx_i] = new_rnn_np[k]
                    # Get obs_flat for buffer storage
                    obs_flat = all_red_obs_flat[k]
                    buffer.store_step(step, env_idx, agent_idx_i, obs_flat,
                                      env_actions_np[k], float(log_probs_np[k]),
                                      alive=True,
                                      actor_rnn_state_before=all_rnn_hidden_in[k],
                                      sequence_start=all_sequence_start[k],
                                      action_std=action_stds_np[k],
                                      policy_mean_action=policy_means_np[k])

                actions_list.append(env_actions)

            # ---- 环境步进 ----
            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(actions_list)

            # 填充 reward / done + 追踪 episode 结局
            for env_idx in range(config.num_envs):
                rew = rewards_list[env_idx]
                don = dones_list[env_idx]
                info = infos_list[env_idx]
                red_joint_reward = _joint_team_reward_once(rew, red_ids)
                buffer.joint_rewards[step, env_idx] = red_joint_reward
                buffer.episode_dones[step, env_idx] = float(all(don.values()))
                current_ep_reward_red[env_idx] += red_joint_reward
                _accumulate_launch_diag_totals(
                    iter_launch_diag, info.get("__launch_diag__", {}))
                launch_quality_step = info.get("__launch_quality_step__", [])
                if isinstance(launch_quality_step, list):
                    iter_launch_quality_records.extend(
                        r for r in launch_quality_step if isinstance(r, dict))
                launch_quality_done = info.get("__launch_quality_done__", [])
                if isinstance(launch_quality_done, list):
                    done_records = [r for r in launch_quality_done if isinstance(r, dict)]
                    iter_launch_quality_done_records.extend(done_records)
                    if launch_quality_writer is not None:
                        for record in done_records:
                            launch_quality_writer.writerow({
                                field: record.get(field, "")
                                for field in LAUNCH_QUALITY_DETAIL_FIELDS
                            })
                        launch_quality_csv_file.flush()

                # ---- accumulate step rewards FIRST (incl. terminal r_end for dead agents) ----
                for i, rid in enumerate(red_ids):
                    if don.get(rid, False):
                        rnn_hidden_actor[env_idx, i] = np.zeros(
                            config.rnn_hidden_size, dtype=np.float32)
                        actor_reset_before[env_idx, i] = True
                    elif buffer.alive[step, env_idx, i]:
                        actor_reset_before[env_idx, i] = False
                    # Accumulate per-component diagnostics
                    rcinfo = info.get(rid, {})
                    for k in COMP_KEYS:
                        if k == "r_end":
                            continue
                        current_ep_comp_red[k][env_idx] += rcinfo.get(k, 0.0)

                reward_summary = info.get("__reward_summary__", {})
                if reward_summary:
                    current_ep_comp_red["r_end"][env_idx] += float(
                        reward_summary.get("red_team_terminal_reward", 0.0))
                else:
                    current_ep_comp_red["r_end"][env_idx] += sum(
                        float(info.get(rid, {}).get("r_end", 0.0))
                        for rid in red_ids)

                inf = infos_list[env_idx]
                for rid in red_ids:
                    fired = inf.get(rid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_red[env_idx] += fired
                    red_missiles_total += fired
                for bid in blue_ids:
                    fired = inf.get(bid, {}).get("missiles_fired_this_step", 0)
                    current_ep_missiles_blue[env_idx] += fired
                    blue_missiles_total += fired

                # ---- episodic settlement AFTER accumulation (terminal r_end is included) ----
                if all(don.values()):
                    invalid_episode = _episode_is_invalid(inf)
                    episode_info = inf.get("__episode__", {})
                    target_diag = inf.get("__target_assignment_diag__", {})
                    for field in (
                            "target_reallocations",
                            "target_reallocations_after_death",
                            "target_switches_while_alive",
                            "engaged_wait_frames", "no_alive_target_frames"):
                        iter_environment_diag[field] += int(
                            target_diag.get(field, 0))
                    mws_diag = inf.get("__mws_diag__", {})
                    for field in (
                            "red_detected_agent_decisions",
                            "red_override_agent_decisions",
                            "blue_detected_agent_decisions",
                            "blue_override_agent_decisions",
                            "red_warning_generations",
                            "red_direction_changes_within_same_missile",
                            "red_suppressed_direction_flip_attempts"):
                        iter_environment_diag[field] += int(mws_diag.get(field, 0))
                    iter_environment_diag["red_maximum_continuous_decisions"] = max(
                        iter_environment_diag["red_maximum_continuous_decisions"],
                        int(mws_diag.get("red_maximum_continuous_decisions", 0)))
                    iter_environment_diag["red_target_heading_delta_max_deg"] = max(
                        iter_environment_diag["red_target_heading_delta_max_deg"],
                        float(mws_diag.get("red_target_heading_delta_max_deg", 0.0)))
                    load_diag = inf.get("__load_diag__", {})
                    for field in (
                            "invalid_nonfinite_load_count",
                            "invalid_catastrophic_finite_load_count",
                            "invalid_persistent_extreme_finite_load_count"):
                        iter_environment_diag[field] += int(load_diag.get(field, 0))
                    for field in (
                            "red_warning_to_terminal_mean_s",
                            "red_warning_to_terminal_p50_s",
                            "red_warning_to_hit_mean_s",
                            "blue_warning_to_terminal_mean_s",
                            "blue_warning_to_terminal_p50_s",
                            "blue_warning_to_hit_mean_s"):
                        value = mws_diag.get(field)
                        if value is not None:
                            iter_environment_diag[field] += float(value)
                            iter_environment_diag[f"{field}_count"] += 1
                    blue_diag = inf.get("__blue_policy_diag__", {})
                    for field in BLUE_POLICY_DIAG_CSV_FIELDS:
                        iter_blue_policy_diag[field] += int(blue_diag.get(field, 0))
                    blue_alive = sum(
                        1 for bid in blue_ids
                        if inf.get(bid, {}).get("alive", False))
                    red_alive = sum(
                        1 for rid in red_ids
                        if inf.get(rid, {}).get("alive", False))
                    iter_aircraft_diag_records.extend(
                        inf.get(aid, {}) for aid in red_ids + blue_ids)
                    iter_episode_physics_frames += int(
                        inf.get("__episode__", {}).get("EpisodeLength", 0)) * 12
                    if invalid_episode:
                        _append_invalid_trace_jsonl(
                            config.extreme_load_trace_file,
                            run_id=str(manifest["run_id"]), seed=config.seed,
                            total_step=total_steps + config.num_envs,
                            env_index=env_idx, episode_info=episode_info,
                            traces=list(inf.get("__extreme_load_traces__", [])))
                        invalid_numerical_episodes += 1
                        iter_invalid_episodes_dropped += 1
                        buffer.invalidate_episode(
                            env_idx, episode_start_in_rollout[env_idx], step)
                        reasons = inf.get("__episode__", {}).get(
                            "invalid_numerical_reasons", [])
                        for label in reasons:
                            if ":" not in str(label):
                                continue
                            invalid_aid, invalid_reason = str(label).split(":", 1)
                            team = "blue" if invalid_aid.startswith("blue") else "red"
                            death_stats[team][f"Invalid_{invalid_reason}"] += 1
                        print(
                            f"  [INVALID EPISODE] env={env_idx} "
                            f"total_steps={total_steps} reasons={reasons}",
                            flush=True)
                    else:
                        iter_episode_lengths.append(int(
                            episode_info.get(
                                "EpisodeLength",
                                episode_info.get("episode_length", 0))))
                        iter_episode_first_events.append({
                            key: episode_info.get(key)
                            for key in (
                                "red_first_launch_step",
                                "blue_first_launch_step",
                                "red_first_hit_step",
                                "blue_first_hit_step")
                        })
                        total_episodes += 1
                        iter_episodes += 1
                        outcome = _episode_outcome(red_alive, blue_alive)
                        if outcome == "red":
                            red_wins += 1
                            iter_red_wins += 1
                        elif outcome == "blue":
                            blue_wins += 1
                        else:
                            draws += 1
                        for bid in blue_ids:
                            dr = inf.get(bid, {}).get("death_reason")
                            if dr:
                                death_stats["blue"][dr] += 1
                        for rid in red_ids:
                            dr = inf.get(rid, {}).get("death_reason")
                            if dr:
                                death_stats["red"][dr] += 1
                        recent_ep_rewards_red.append(
                            float(current_ep_reward_red[env_idx]))
                        recent_ep_comps_red.append(
                            {k: float(current_ep_comp_red[k][env_idx])
                             for k in COMP_KEYS})
                        recent_ep_missiles_red.append(
                            float(current_ep_missiles_red[env_idx]))
                        recent_ep_missiles_blue.append(
                            float(current_ep_missiles_blue[env_idx]))
                        recent_ep_red_alive.append(float(red_alive))
                        recent_ep_blue_alive.append(float(blue_alive))
                    current_ep_reward_red[env_idx] = 0.0
                    for k in COMP_KEYS:
                        current_ep_comp_red[k][env_idx] = 0.0
                    current_ep_missiles_red[env_idx] = 0.0
                    current_ep_missiles_blue[env_idx] = 0.0
                    episode_start_in_rollout[env_idx] = step + 1

            raw_obs_list = next_obs_list
            total_steps += config.num_envs

        # 计算 GAE bootstrap 值: centralized V(s_T) — batched across all envs
        bootstrap_global_obs_list = []
        for env_idx in range(config.num_envs):
            env_obs = raw_obs_list[env_idx]
            if not env_obs or len(env_obs) == 0:
                bootstrap_global_obs_list.append(np.zeros(global_obs_dim, dtype=np.float32))
                continue
            global_obs_parts = []
            for rid in red_ids:
                if rid in env_obs:
                    global_obs_parts.append(_flatten_obs(
                        env_obs[rid], obs_mode=config.obs_mode,
                        obs_normalization=config.obs_normalization))
                else:
                    global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
            bootstrap_global_obs_list.append(_global_state_from_local_obs_flats(
                global_obs_parts, obs_mode=config.obs_mode))
        bootstrap_obs_batch = torch.as_tensor(
            np.stack(bootstrap_global_obs_list), dtype=torch.float32, device=device)
        with torch.no_grad():
            v_bootstrap_all = critic(bootstrap_obs_batch).squeeze(-1).cpu().numpy()
        for env_idx in range(config.num_envs):
            v_bootstrap = (0.0 if buffer.episode_dones[-1, env_idx]
                           else float(v_bootstrap_all[env_idx]))
            buffer.bootstrap_value[env_idx] = v_bootstrap

        valid_actor_mask = (
            buffer.alive & buffer.valid_transitions[:, :, np.newaxis])
        iter_action_bound = _empty_action_bound_totals()
        valid_executed_actions = buffer.actions[valid_actor_mask]
        valid_policy_means = buffer.policy_mean_actions[valid_actor_mask]
        if valid_executed_actions.size:
            _accumulate_action_bound_totals(
                iter_action_bound, valid_executed_actions, valid_policy_means)
        valid_std_values = buffer.action_stds[valid_actor_mask]
        iter_sampled_stds = (
            [valid_std_values] if valid_std_values.size else [])

        # ---- PPO 更新 ----
        stats = ppo_update(actor, critic, actor_opt, critic_opt,
                           buffer, config, device, total_steps=total_steps)

        t_elapsed = time.perf_counter() - t_start
        avg_r_red = np.mean(recent_ep_rewards_red) if recent_ep_rewards_red else 0.0
        avg_m_red = np.mean(recent_ep_missiles_red) if recent_ep_missiles_red else 0.0
        avg_m_blue = np.mean(recent_ep_missiles_blue) if recent_ep_missiles_blue else 0.0
        red_win_rate = red_wins / max(total_episodes, 1)
        std_r_red = float(np.std(recent_ep_rewards_red)) if len(recent_ep_rewards_red) > 1 else 0.0
        iter_win_rate = iter_red_wins / max(iter_episodes, 1)
        red_alive_mean = np.mean(recent_ep_red_alive) if recent_ep_red_alive else 0.0
        blue_alive_mean = np.mean(recent_ep_blue_alive) if recent_ep_blue_alive else 0.0

        red_deaths_missile = sum(
            v for k, v in death_stats["red"].items()
            if _classify_death_reason(k) == "missile")
        red_deaths_crash = sum(
            v for k, v in death_stats["red"].items()
            if _classify_death_reason(k) == "crash")
        blue_deaths_missile = sum(
            v for k, v in death_stats["blue"].items()
            if _classify_death_reason(k) == "missile")
        blue_deaths_crash = sum(
            v for k, v in death_stats["blue"].items()
            if _classify_death_reason(k) == "crash")
        red_missile_hits = blue_deaths_missile
        blue_missile_hits = red_deaths_missile
        red_total_deaths = sum(death_stats["red"].values())
        blue_total_deaths = sum(death_stats["blue"].values())
        red_missile_hit_rate = _safe_div(red_missile_hits, red_missiles_total)
        blue_missile_hit_rate = _safe_div(blue_missile_hits, blue_missiles_total)
        kd_red_all, _ = _ratio_with_denominator_zero(
            blue_total_deaths, red_total_deaths)
        kd_red_missile, _ = _ratio_with_denominator_zero(
            blue_deaths_missile, red_deaths_missile)
        rwr, rwr_denominator_zero = _ratio_with_denominator_zero(
            red_wins, blue_wins)

        std_stats = _actor_std_stats(actor, iter_sampled_stds)
        action_std_delta = std_stats["action_std_mean"] - ACTION_STD_INIT
        action_std_growth = _action_std_growth_ratio(
            std_stats["action_std_mean"])
        launch_diag_metrics = _launch_diag_metrics(iter_launch_diag)
        launch_quality_metrics = _launch_quality_metrics(
            iter_launch_quality_records,
            iter_launch_quality_done_records,
        )
        learnability_metrics = _learnability_iteration_metrics(
            iter_launch_quality_records, iter_launch_quality_done_records,
            iter_environment_diag, iter_episode_lengths,
            iter_episode_first_events,
            float(checkpoint_meta["missile_hit_radius_m"]), iter_episodes,
            t_elapsed, config.num_envs * num_steps,
            max(config.total_env_steps - total_steps, 0),
            vec_env.worker_restart_count,
            int(manifest.get("resume_count", 0)))
        manifest["worker_restart_count"] = int(vec_env.worker_restart_count)
        _atomic_json_save(manifest, manifest_path)
        cumulative_environment_diag.update(iter_environment_diag)
        cumulative_missile_term.update(
            (str(record.get("team", "")),
             str(record.get("raw_termination_reason", "unknown")))
            for record in iter_launch_quality_done_records)
        action_bound_metrics = _action_bound_metrics(iter_action_bound)
        def _diag_max(field: str) -> float:
            values = [float(row.get(field, float("nan")))
                      for row in iter_aircraft_diag_records]
            finite = [value for value in values if np.isfinite(value)]
            return max(finite) if finite else float("nan")
        speed_limiter_activations = sum(
            int(row.get("speed_limiter_activations", 0))
            for row in iter_aircraft_diag_records)
        load_limiter_activations = sum(
            int(row.get("load_limiter_activations", 0))
            for row in iter_aircraft_diag_records)
        speed_limiter_rate = (
            1000.0 * speed_limiter_activations / iter_episode_physics_frames
            if iter_episode_physics_frames > 0 else float("nan"))
        envelope_metrics = {
            "MaximumSpeedBeforeLimiterMps": _diag_max(
                "maximum_speed_before_limit_mps"),
            "MaximumSpeedAfterLimiterMps": _diag_max(
                "maximum_speed_after_limit_mps"),
            "SpeedLimiterActivations": speed_limiter_activations,
            "SpeedLimiterActivationRatePer1000PhysicsSteps": speed_limiter_rate,
            "MaximumLoadG": _diag_max("maximum_load_g_seen"),
            "LoadLimiterActivations": load_limiter_activations,
            "EnvironmentDynamicsWarning": (
                "frequent_speed_limiter_activation"
                if np.isfinite(speed_limiter_rate) and speed_limiter_rate > 100.0
                else ""),
        }
        for team, prefix in (("red", "Red"), ("blue", "Blue")):
            rows = [row for row in iter_aircraft_diag_records
                    if row.get("team") == team]
            def _team_max(field):
                values = [float(row.get(field, 0.0)) for row in rows]
                return max(values, default=0.0)
            learnability_metrics.update({
                f"{prefix}MaximumGSeen": _team_max("maximum_load_g_seen"),
                f"{prefix}FramesAbove9G": sum(
                    int(row.get("frames_above_9g", 0)) for row in rows),
                f"{prefix}MaximumConsecutiveAbove9GFrames": _team_max(
                    "maximum_consecutive_above_9g_frames"),
                f"{prefix}EpisodeEverExceeded9G": int(any(
                    bool(row.get("episode_ever_exceeded_9g", False))
                    for row in rows)),
                f"{prefix}TransientAbove30GEvents": sum(
                    int(row.get("transient_above_30g_events", 0)) for row in rows),
                f"{prefix}MaximumConsecutiveAbove30GFrames": _team_max(
                    "maximum_consecutive_above_30g_frames"),
                f"{prefix}LoadProtectionActiveFrames": sum(
                    int(row.get("load_protection_active_frames", 0)) for row in rows),
                f"{prefix}SetpointRateLimitActivations": sum(
                    int(row.get("setpoint_rate_limit_activations", 0)) for row in rows),
                f"{prefix}RequestedHeadingJumpMaxDeg": np.rad2deg(_team_max(
                    "requested_heading_jump_max_rad")),
                f"{prefix}AppliedHeadingJumpMaxDeg": np.rad2deg(_team_max(
                    "applied_heading_jump_max_rad")),
                f"{prefix}RequestedPitchJumpMaxDeg": np.rad2deg(_team_max(
                    "requested_pitch_jump_max_rad")),
                f"{prefix}AppliedPitchJumpMaxDeg": np.rad2deg(_team_max(
                    "applied_pitch_jump_max_rad")),
                f"{prefix}MaximumAbsoluteEPhi": _team_max(
                    "maximum_absolute_e_phi"),
                f"{prefix}MaximumAbsoluteETheta": _team_max(
                    "maximum_absolute_e_theta"),
                f"{prefix}MaximumAbsoluteDerivativeTerm": _team_max(
                    "maximum_absolute_derivative_term"),
                f"{prefix}PIDOutputSaturationFrames": sum(
                    int(row.get("pid_output_saturation_frames", 0))
                    for row in rows),
                f"{prefix}DegenerateArctanRatioCount": sum(
                    int(row.get("degenerate_arctan_ratio_count", 0))
                    for row in rows),
            })

        # Average per-component breakdown across completed episodes
        if recent_ep_comps_red:
            avg_comps = {k: float(np.mean([ep[k] for ep in recent_ep_comps_red]))
                         for k in COMP_KEYS}
        else:
            avg_comps = {k: 0.0 for k in COMP_KEYS}
        component_log_metrics = _reward_component_log_metrics(
            avg_comps, config.num_red)
        altitude_config_json = json.dumps(
            asdict(config.altitude_reward_config), sort_keys=True,
            separators=(",", ":"))

        # Build breakdown string: [Alt:+12.3 Pitch:-0.5 Roll:0.0 Vel:-0.3 Adv:+0.0 End:-180.0]
        comp_str = " ".join(
            f"{k.replace('r_','').capitalize()}:{avg_comps[k]:+.1f}"
            for k in COMP_KEYS)

        # ---- 持久化：CSV 写入 + flush ----
        csv_writer.writerow([iteration, total_steps,
                             f"{stats['actor_loss']:.6f}",
                             f"{stats['critic_loss']:.6f}",
                             f"{stats['entropy']:.6f}",
                             f"{stats['policy_loss']:.6f}",
                             f"{stats['entropy_bonus']:.6f}",
                             stats["ActorUpdateAttempts"],
                             stats["ActorUpdatesApplied"],
                             stats["ActorUpdatesSkipped"],
                             stats["CriticUpdateAttempts"],
                             stats["CriticUpdatesApplied"],
                             stats["CriticUpdatesSkipped"],
                             _csv_optional_float(action_std_delta),
                             _csv_optional_float(action_std_growth),
                             ("" if iter_episodes == 0 else f"{avg_r_red:.4f}"),
                             f"{red_win_rate:.6f}",
                             ("" if iter_episodes == 0 else f"{std_r_red:.4f}"),
                             ("" if iter_episodes == 0 else f"{iter_win_rate:.6f}"),
                             ("" if iter_episodes == 0 else f"{avg_m_red:.1f}"),
                             ("" if iter_episodes == 0 else f"{avg_m_blue:.1f}"),
                             *[
                                 rollout_layout[field]
                                 for field in ROLLOUT_LAYOUT_CSV_FIELDS
                             ],
                             total_episodes, invalid_numerical_episodes,
                             stats["InvalidTransitionsDropped"],
                             iter_invalid_episodes_dropped,
                             stats["UpdateSkipReason"],
                             red_wins, blue_wins, draws,
                             ("" if iter_episodes == 0 else f"{red_alive_mean:.4f}"),
                             ("" if iter_episodes == 0 else f"{blue_alive_mean:.4f}"),
                             red_deaths_missile,
                             red_deaths_crash,
                             blue_deaths_missile,
                             blue_deaths_crash,
                             red_missile_hits,
                             blue_missile_hits,
                             f"{red_missile_hit_rate:.6f}",
                             f"{blue_missile_hit_rate:.6f}",
                             f"{kd_red_all:.6f}",
                             f"{kd_red_missile:.6f}",
                             f"{rwr:.6f}",
                             int(rwr_denominator_zero),
                             "paper_3v3_joint_eq15_23_v1",
                             config.reward_mode,
                             config.environment_profile,
                             config.obs_normalization,
                             config.pid_profile,
                             f"{config.pid_throttle_base:.6f}",
                             config.missile_guidance_mode,
                             CHECKPOINT_SCHEMA_VERSION,
                             ACTION_DISTRIBUTION_VERSION,
                             ENTROPY_ESTIMATOR_VERSION,
                             checkpoint_meta["environment_config_fingerprint"],
                             config.blue_policy_profile,
                             checkpoint_meta.get("red_mws_mode", ""),
                             checkpoint_meta.get("blue_mws_mode", ""),
                             config.num_red,
                             config.num_blue,
                             config.max_episode_length,
                             config.altitude_reward_config.version,
                             altitude_config_json,
                             *[
                                 ("" if iter_episodes == 0
                                  else f"{component_log_metrics[field]:.6f}")
                                 for field in REWARD_COMPONENT_LOG_FIELDS
                             ],
                             _csv_optional_float(std_stats['action_std_mean']),
                             _csv_optional_float(std_stats['action_std_min']),
                             _csv_optional_float(std_stats['action_std_max']),
                             _csv_optional_float(std_stats['action_log_std_mean']),
                             _csv_optional_float(std_stats['action_std_mean']),
                             _csv_optional_float(std_stats['action_std_min']),
                             _csv_optional_float(std_stats['action_std_max']),
                             _csv_optional_float(
                                 std_stats['action_std_lower_bound_frac']),
                             _csv_optional_float(
                                 std_stats['action_std_upper_bound_frac']),
                             *[
                                 (f"{launch_diag_metrics[field]:.6f}"
                                  if "Rate" in field else launch_diag_metrics[field])
                                 for field in LAUNCH_DIAG_CSV_FIELDS
                             ],
                             *[
                                 f"{launch_quality_metrics[field]:.6f}"
                                 for field in LAUNCH_QUALITY_AGG_CSV_FIELDS
                             ],
                             *[
                                 _csv_optional_float(action_bound_metrics[field])
                                 for field in ACTION_BOUND_CSV_FIELDS
                             ],
                             *[
                                 (envelope_metrics[field]
                                  if field == "EnvironmentDynamicsWarning"
                                  else _csv_optional_float(envelope_metrics[field]))
                                 for field in AIRCRAFT_ENVELOPE_CSV_FIELDS
                             ],
                             *[iter_blue_policy_diag[field]
                               for field in BLUE_POLICY_DIAG_CSV_FIELDS],
                             *[
                                 _csv_optional_float(learnability_metrics[field])
                                 for field in LEARNABILITY_DIAG_CSV_FIELDS
                             ]])
        _flush_and_periodic_fsync(csv_file, iteration)

        # ---- 持久化：results/ 绘图数据 (累计 + 每 1M 步自动保存) ----
        results_log.append({
            "Step":           total_steps,
            "Iteration":      iteration,
            "RedMeanReward":  None if iter_episodes == 0 else avg_r_red,
            "RedRewardStd":   None if iter_episodes == 0 else std_r_red,
            "WinRateRecent":  None if iter_episodes == 0 else iter_win_rate,
            "WinRateCumul":   red_win_rate,
            "RedMissiles":    None if iter_episodes == 0 else avg_m_red,
            "BlueMissiles":   None if iter_episodes == 0 else avg_m_blue,
            "requested_replay_buffer_size": rollout_layout["requested_replay_buffer_size"],
            "rollout_horizon_per_env": rollout_layout["rollout_horizon_per_env"],
            "transitions_per_update": rollout_layout["transitions_per_update"],
            "unused_replay_slots": rollout_layout["unused_replay_slots"],
            "Episodes":       total_episodes,
            "InvalidNumericalEpisodes": invalid_numerical_episodes,
            "InvalidTransitionsDropped": stats["InvalidTransitionsDropped"],
            "InvalidEpisodesDropped": iter_invalid_episodes_dropped,
            "UpdateSkipReason": stats["UpdateSkipReason"],
            "RedWins":        red_wins,
            "BlueWins":       blue_wins,
            "Draws":          draws,
            "RedAliveMean":   None if iter_episodes == 0 else red_alive_mean,
            "BlueAliveMean":  None if iter_episodes == 0 else blue_alive_mean,
            "RedDeathsMissile": red_deaths_missile,
            "RedDeathsCrash": red_deaths_crash,
            "BlueDeathsMissile": blue_deaths_missile,
            "BlueDeathsCrash": blue_deaths_crash,
            "RedMissileHits": red_missile_hits,
            "BlueMissileHits": blue_missile_hits,
            "RedMissileHitRate": red_missile_hit_rate,
            "BlueMissileHitRate": blue_missile_hit_rate,
            "KD_Red_AllDeaths": kd_red_all,
            "KD_Red_MissileOnly": kd_red_missile,
            "RWR":            rwr,
            "RWRDenominatorZero": rwr_denominator_zero,
            "RewardVersion": "paper_3v3_joint_eq15_23_v1",
            "RewardMode": config.reward_mode,
            "EnvironmentProfile": config.environment_profile,
            "ObsNormalization": config.obs_normalization,
            "PIDProfile": config.pid_profile,
            "PIDThrottleBase": config.pid_throttle_base,
            "MissileGuidanceMode": config.missile_guidance_mode,
            "CheckpointSchema": CHECKPOINT_SCHEMA_VERSION,
            "ActionDistribution": ACTION_DISTRIBUTION_VERSION,
            "EntropyEstimator": ENTROPY_ESTIMATOR_VERSION,
            "EnvironmentConfigFingerprint": checkpoint_meta[
                "environment_config_fingerprint"],
            "BluePolicyProfile": config.blue_policy_profile,
            "RedMWSMode": checkpoint_meta.get("red_mws_mode", ""),
            "BlueMWSMode": checkpoint_meta.get("blue_mws_mode", ""),
            "NumRed": config.num_red,
            "NumBlue": config.num_blue,
            "MaxSteps": config.max_episode_length,
            "AltitudeRewardConfigVersion": config.altitude_reward_config.version,
            "AltitudeRewardConfig": altitude_config_json,
            "ActionStdMean":  _result_optional_float(std_stats["action_std_mean"]),
            "ActionStdMin":   _result_optional_float(std_stats["action_std_min"]),
            "ActionStdMax":   _result_optional_float(std_stats["action_std_max"]),
            "ActionLogStdMean": _result_optional_float(
                std_stats["action_log_std_mean"]),
            "StateDependentStdMean": _result_optional_float(
                std_stats["action_std_mean"]),
            "StateDependentStdMin": _result_optional_float(
                std_stats["action_std_min"]),
            "StateDependentStdMax": _result_optional_float(
                std_stats["action_std_max"]),
            "StateDependentStdLowerBoundFrac": std_stats[
                "action_std_lower_bound_frac"] if np.isfinite(std_stats[
                    "action_std_lower_bound_frac"]) else None,
            "StateDependentStdUpperBoundFrac": std_stats[
                "action_std_upper_bound_frac"] if np.isfinite(std_stats[
                    "action_std_upper_bound_frac"]) else None,
            "ActionStdDeltaFromInit": _result_optional_float(action_std_delta),
            "ActionStdGrowthRatio": _result_optional_float(action_std_growth),
            "ActorLoss":      stats["actor_loss"],
            "CriticLoss":     stats["critic_loss"],
            "BaseNormalEntropy": stats["entropy"],
            "PolicyLoss":     stats["policy_loss"],
            "EntropyBonus":   stats["entropy_bonus"],
            "ActorUpdateAttempts": stats["ActorUpdateAttempts"],
            "ActorUpdatesApplied": stats["ActorUpdatesApplied"],
            "ActorUpdatesSkipped": stats["ActorUpdatesSkipped"],
            "CriticUpdateAttempts": stats["CriticUpdateAttempts"],
            "CriticUpdatesApplied": stats["CriticUpdatesApplied"],
            "CriticUpdatesSkipped": stats["CriticUpdatesSkipped"],
            "r_pitch":        avg_comps.get("r_pitch", 0.0),
            "r_roll":         avg_comps.get("r_roll", 0.0),
            "r_alt":          avg_comps.get("r_alt", 0.0),
            "r_bound":        avg_comps.get("r_bound", 0.0),
            "r_vel":          avg_comps.get("r_vel", 0.0),
            "r_adv":          avg_comps.get("r_adv", 0.0),
            "r_end":          avg_comps.get("r_end", 0.0),
            "r_death":        avg_comps.get("r_death", 0.0),
        })
        results_log[-1].update({
            key: (None if iter_episodes == 0 else value)
            for key, value in component_log_metrics.items()})
        results_log[-1].update(launch_diag_metrics)
        results_log[-1].update(launch_quality_metrics)
        results_log[-1].update(learnability_metrics)
        results_log[-1].update({
            key: _result_optional_float(value)
            for key, value in action_bound_metrics.items()})
        results_log[-1].update({
            key: (value if key == "EnvironmentDynamicsWarning"
                  else _result_optional_float(value))
            for key, value in envelope_metrics.items()})
        milestone_cur = total_steps // 1_000_000
        milestone_prev = (total_steps - config.num_envs * num_steps) // 1_000_000
        if milestone_cur > milestone_prev or total_steps >= config.total_env_steps:
            results_dir = os.path.dirname(config.results_file)
            if results_dir:
                os.makedirs(results_dir, exist_ok=True)
            with open(config.results_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(results_log[0].keys())
                for row in results_log:
                    w.writerow(row.values())
            print(f"  [Results saved] {config.results_file} "
                  f"({len(results_log)} rows)", flush=True)

        # ---- 终端打印 ----
        def _fmt_death(counter: Counter) -> str:
            order = ["Missile_Kill", "Crash_LowAlt", "Crash_HighAlt",
                     "Crash_OverG", "Crash_Extreme", "Invalid_NonFiniteState",
                     "Invalid_NonFiniteLoad", "Invalid_CatastrophicFiniteLoad",
                     "Invalid_PersistentExtremeFiniteLoad"]
            parts = []
            for k in order:
                parts.append(f"{k.replace('Crash_','').replace('Missile_','')}:{counter.get(k, 0)}")
            # Catch any unexpected reason
            for k in sorted(counter.keys()):
                if k not in order:
                    parts.append(f"{k}:{counter[k]}")
            return ", ".join(parts)

        print(f"Iter {iteration:5d} | "
              f"total_steps={total_steps:9d} | "
              f"t={t_elapsed:5.1f}s | "
              f"R_red={avg_r_red:+8.1f} [{comp_str}] | "
              f"M_red={avg_m_red:.0f} M_blue={avg_m_blue:.0f} | "
              f"PolicyLoss={stats['policy_loss']:+.4f} "
              f"EntropyBonus={stats['entropy_bonus']:+.4f} "
              f"ActorLoss={stats['actor_loss']:+.4f} "
              f"CriticLoss={stats['critic_loss']:+.4f} "
              f"EntCoef={_current_entropy_coef(config, total_steps):.4f} "
              f"BaseNormalEntropy={stats['entropy']:.4f} "
              f"Std={std_stats['action_std_mean']:.4f} | "
              f"LaunchDiag R={launch_diag_metrics['LaunchDiagRedGeometryOk']}/"
              f"{launch_diag_metrics['LaunchDiagRedLaunches']} "
              f"B={launch_diag_metrics['LaunchDiagBlueGeometryOk']}/"
              f"{launch_diag_metrics['LaunchDiagBlueLaunches']} | "
              f"WinRate_red={red_win_rate:.3f} "
              f"(Ep={total_episodes} W={red_wins}/{blue_wins}/{draws}) | "
              f"Deaths: Red[{_fmt_death(death_stats['red'])}] "
              f"Blue[{_fmt_death(death_stats['blue'])}]")

        # ---- 持久化：高频轮转存档 (每 10 iter, 保留最新 5 个) ----
        if iteration % 10 == 0:
            actor_path = os.path.join(
                config.checkpoint_dir, f"vanilla_actor_latest_{iteration:06d}.pt")
            critic_path = os.path.join(
                config.checkpoint_dir, f"centralized_critic_latest_{iteration:06d}.pt")
            _save_model_checkpoint(actor_path, actor, checkpoint_meta, "actor")
            _save_model_checkpoint(critic_path, critic, checkpoint_meta, "critic")
            # 轮转清理：删除超出保留数量的旧 checkpoint
            _cleanup_rotating_checkpoints(config.checkpoint_dir,
                                          "vanilla_actor_latest", keep=5)
            _cleanup_rotating_checkpoints(config.checkpoint_dir,
                                          "centralized_critic_latest", keep=5)

        # ---- 持久化：最佳模型拦截 (需满足评估准入准则) ----
        # best_reward: selects by recent average reward
        # best_winrate: selects by recent iteration win rate, reward as tie-breaker
        # legacy best.pt aliases best_winrate for evaluator compatibility
        if total_episodes >= MIN_EPISODES_TO_EVAL:
            # ---- best_reward checkpoint ----
            if avg_r_red > best_reward_value:
                best_reward_value = avg_r_red
                best_reward_win_rate = red_win_rate
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "vanilla_actor_best_reward.pt"),
                    actor, checkpoint_meta, "actor")
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "centralized_critic_best_reward.pt"),
                    critic, checkpoint_meta, "critic")
                print(f"  *** New Best Reward Model Saved! "
                      f"(Reward={best_reward_value:+.2f}, "
                      f"RecentWinRate={iter_win_rate:.4f}, "
                      f"CumulWinRate={red_win_rate:.4f}) ***")

            # ---- best_winrate checkpoint ----
            winrate_is_better = (
                iter_win_rate > best_winrate_value
                or (abs(iter_win_rate - best_winrate_value) < 1e-6
                    and avg_r_red > best_winrate_reward)
            )
            if winrate_is_better:
                best_winrate_value = iter_win_rate
                best_winrate_reward = avg_r_red
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "vanilla_actor_best_winrate.pt"),
                    actor, checkpoint_meta, "actor")
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "centralized_critic_best_winrate.pt"),
                    critic, checkpoint_meta, "critic")
                # legacy compatibility alias
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "vanilla_actor_best.pt"),
                    actor, checkpoint_meta, "actor")
                _save_model_checkpoint(
                    os.path.join(config.checkpoint_dir,
                                 "centralized_critic_best.pt"),
                    critic, checkpoint_meta, "critic")
                print(f"  *** New Best WinRate Model Saved! "
                      f"(RecentWinRate={best_winrate_value:.4f}, "
                      f"Reward={best_winrate_reward:+.2f}, "
                      f"CumulWinRate={red_win_rate:.4f}) ***")

        if (config.eval_during_training
                and total_steps >= config.eval_interval_steps
                and total_steps - last_eval_steps >= config.eval_interval_steps):
            _run_periodic_evaluation(
                actor, config, device, checkpoint_meta, iteration,
                total_steps, final_checkpoint=False)
            last_eval_steps = total_steps

        if iteration % 10 == 0 or _STOP_REQUESTED:
            training_state = _build_training_state(
                actor, critic, actor_opt, critic_opt, config, checkpoint_meta,
                _runtime_state(iteration + 1))
            _atomic_torch_save(
                training_state,
                os.path.join(config.checkpoint_dir, "latest_training_state.pt"))
        if _STOP_REQUESTED:
            print("[STOP] safe stop requested; latest training state saved.",
                  flush=True)
            break
        iteration += 1

    # ---- 持久化：最终模型存档 ----
    _save_model_checkpoint(
        os.path.join(config.checkpoint_dir, "vanilla_actor_final.pt"),
        actor, checkpoint_meta, "actor")
    _save_model_checkpoint(
        os.path.join(config.checkpoint_dir, "centralized_critic_final.pt"),
        critic, checkpoint_meta, "critic")
    _atomic_torch_save(
        _build_training_state(
            actor, critic, actor_opt, critic_opt, config, checkpoint_meta,
            _runtime_state(iteration + (1 if _STOP_REQUESTED else 0))),
        os.path.join(config.checkpoint_dir, "latest_training_state.pt"))
    if (config.eval_during_training and not _STOP_REQUESTED
            and last_eval_steps != total_steps):
        _run_periodic_evaluation(
            actor, config, device, checkpoint_meta, iteration,
            total_steps, final_checkpoint=True)
        last_eval_steps = total_steps
        _atomic_torch_save(
            _build_training_state(
                actor, critic, actor_opt, critic_opt, config, checkpoint_meta,
                _runtime_state(iteration)),
            os.path.join(config.checkpoint_dir, "latest_training_state.pt"))
    if results_log:
        _ensure_parent_dir(config.results_file)
        fieldnames = list(results_log[-1].keys())
        with open(config.results_file, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_log:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
    print("=" * 70)
    print(f"最终模型已保存至 {config.checkpoint_dir}/")
    print(f"Results 已保存至 {config.results_file} ({len(results_log)} rows)")
    print(f"总 Episodes: {total_episodes}  "
          f"红方胜: {red_wins}  蓝方胜: {blue_wins}  平局: {draws}  "
          f"红方胜率: {red_win_rate:.4f}")
    csv_file.close()
    if launch_quality_csv_file is not None:
        launch_quality_csv_file.close()

    # ---- 清理 ----
    vec_env.close()
    print("基线训练完成！")


if __name__ == "__main__":
    mp.freeze_support()
    main()
