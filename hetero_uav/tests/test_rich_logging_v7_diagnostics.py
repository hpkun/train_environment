"""Lightweight tests for v7 rich logging diagnostics. No JSBSim."""
from pathlib import Path
import csv, json, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestV7Schema:
    def test_reward_components_has_v7_fields(self):
        from scripts.experiment_logging_schema import REWARD_COMPONENT_COLUMNS
        for k in ["tam_v7_total", "tam_v7_uav_situation", "tam_v7_mav_safety",
                  "tam_v7_uav_first_out_of_zone", "tam_v7_mav_team_credit_delta",
                  "tam_v7_terminal_per_agent", "tam_v7_shared_track_usage_log"]:
            assert k in REWARD_COMPONENT_COLUMNS, f"missing {k}"

    def test_episode_reward_components_schema_exists(self):
        from scripts.experiment_logging_schema import FILE_SCHEMAS, EPISODE_REWARD_COMPONENTS_COLUMNS
        assert "episode_reward_components.csv" in FILE_SCHEMAS
        assert "tam_v7_total_sum" in EPISODE_REWARD_COMPONENTS_COLUMNS

    def test_aircraft_timeseries_has_team_role(self):
        from scripts.experiment_logging_schema import AIRCRAFT_TIMESERIES_COLUMNS
        for k in ["team", "role", "agent_id", "alive", "altitude", "speed", "is_mav", "is_uav"]:
            assert k in AIRCRAFT_TIMESERIES_COLUMNS, f"missing {k}"


class TestV7RichLoggerMethods:
    def test_has_aircraft_timeseries_method(self):
        from scripts.rich_logging import RichExperimentLogger
        assert hasattr(RichExperimentLogger, "write_aircraft_timeseries")

    def test_has_reward_components_method(self):
        from scripts.rich_logging import RichExperimentLogger
        assert hasattr(RichExperimentLogger, "write_reward_components")

    def test_has_episode_reward_components_method(self):
        from scripts.rich_logging import RichExperimentLogger
        assert hasattr(RichExperimentLogger, "write_episode_reward_components")


class TestV7MockWriting:
    def test_aircraft_timeseries_writes_red_and_blue(self):
        import numpy as np
        from scripts.rich_logging import RichExperimentLogger
        from scripts.experiment_logging_schema import ensure_schema_files
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ensure_schema_files(d)
            logger = RichExperimentLogger(d, run_id="test", method_name="test",
                                          scenario_name="test", device="cpu",
                                          num_envs=1, rollout_length_per_env=256,
                                          transitions_per_rollout=256)

            class MockSim:
                is_alive = True
                def get_geodetic(self): return (120.0, 30.0, 6000.0)
                def get_rpy(self): return (0.0, 0.1, 1.5)
                def get_velocity(self): return np.array([250.0, 0.0, 0.0])
                def get_position(self): return np.array([0.0, 0.0, 6000.0])

            class MockEnv:
                red_ids = ["red_0", "red_1"]
                blue_ids = ["blue_0"]
                agent_roles = {"red_0": "mav", "red_1": "attack_uav", "blue_0": "attack_uav"}
                red_planes = {"red_0": MockSim(), "red_1": MockSim()}
                blue_planes = {"blue_0": MockSim()}

            logger.write_aircraft_timeseries(MockEnv(), scenario="test", episode_id=0, step=1, sim_time=0.2)
            logger.close()

            with open(d / "aircraft_timeseries.csv") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3, f"expected 3 rows (red_0,red_1,blue_0), got {len(rows)}"
            agents = {r["agent_id"] for r in rows}
            assert "red_0" in agents
            assert "blue_0" in agents
            teams = {r["team"] for r in rows}
            assert "red" in teams and "blue" in teams

    def test_reward_components_writes_v7_keys(self):
        import numpy as np
        from scripts.rich_logging import RichExperimentLogger
        from scripts.experiment_logging_schema import ensure_schema_files
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ensure_schema_files(d)
            logger = RichExperimentLogger(d, run_id="test", method_name="test",
                                          scenario_name="test", device="cpu",
                                          num_envs=1, rollout_length_per_env=256,
                                          transitions_per_rollout=256)
            info = {"reward_components": {
                "red_0": {"tam_v7_total": 0.05, "tam_v7_mav_safety": -0.02, "tam_v7_mav_support": 0.07,
                          "tam_v7_uav_situation": 0.0, "tam_v7_uav_first_out_of_zone": 0.0,
                          "tam_v7_shared_track_usage_log": 2.0},
                "red_1": {"tam_v7_total": -0.10, "tam_v7_uav_situation": -0.03, "tam_v7_uav_flight": -0.07,
                          "tam_v7_mav_safety": 0.0, "tam_v7_mav_support": 0.0},
            }}
            logger.write_reward_components(info, scenario="test", episode_id=0, step=1, sim_time=0.2)
            logger.close()

            with open(d / "reward_components.csv") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            vals = {r["agent_id"]: r["tam_v7_total"] for r in rows}
            assert vals["red_0"] == "0.05"
            assert vals["red_1"] == "-0.1"

    def test_episode_reward_components_writes_scenario(self):
        import numpy as np
        from scripts.rich_logging import RichExperimentLogger
        from scripts.experiment_logging_schema import ensure_schema_files
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ensure_schema_files(d)
            logger = RichExperimentLogger(d, run_id="test", method_name="test",
                                          scenario_name="test", device="cpu",
                                          num_envs=1, rollout_length_per_env=256,
                                          transitions_per_rollout=256)
            logger.write_episode_reward_components(
                scenario="test_scenario",
                episode_id=0, agent_id="red_0", role="mav", team="red",
                episode_length=500, episode_return=-200.0,
                component_sums={"tam_v7_total_sum": -200.0, "tam_v7_mav_safety_sum": -30.0,
                                "tam_v7_mav_support_sum": 20.0},
                outcome="timeout", end_reason="max_steps"
            )
            logger.close()
            with open(d / "episode_reward_components.csv") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["scenario"] == "test_scenario"
            assert rows[0]["tam_v7_total_sum"] == "-200.0"
            assert rows[0]["outcome"] == "timeout"
            assert rows[0]["end_reason"] == "max_steps"


class TestV7SingleRunnerPresence:
    def test_single_runner_has_per_agent_accumulators(self):
        import inspect
        import scripts.train_happo_reference as mod
        src = inspect.getsource(mod)
        assert "current_ep_reward_comp_by_agent" in src
        assert "write_episode_reward_components" in src
        assert "tam_v7_mav_team_credit_used_max" in src
        # loss fields are dynamically keyed (key + "_last") in the runner
        from scripts.experiment_logging_schema import EPISODE_REWARD_COMPONENTS_COLUMNS
        assert "tam_v7_red_loss_weighted_last" in EPISODE_REWARD_COMPONENTS_COLUMNS
        assert "tam_v7_blue_loss_frac_last" in EPISODE_REWARD_COMPONENTS_COLUMNS
        assert "current_ep_launch_stats" in src
