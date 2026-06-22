"""Tests for combat-oriented checkpoint selection scoring (v2)."""
import pytest
from eval_checkpoint_selection import score_record, compute_eval_scores


class TestScoreRecord:
    """Verify combat score prefers elimination wins over timeout-advantage."""

    def test_elimination_beats_timeout(self):
        elim = {"red_elimination_win_rate": 1.0, "red_timeout_alive_advantage_rate": 0.0,
                "blue_dead_mean": 2.0, "red_dead_mean": 1.0,
                "red_missile_hits_mean": 2.0,
                "blue_timeout_alive_advantage_rate": 0.0}
        tout = {"red_elimination_win_rate": 0.0, "red_timeout_alive_advantage_rate": 1.0,
                "blue_dead_mean": 0.0, "red_dead_mean": 1.0,
                "red_missile_hits_mean": 0.0,
                "blue_timeout_alive_advantage_rate": 0.0}
        assert score_record(elim) > score_record(tout), "elimination should score higher than timeout"

    def test_red_loss_penalised(self):
        good_kill = {"red_elimination_win_rate": 0.5, "blue_dead_mean": 1.0, "red_dead_mean": 0.5,
                     "red_missile_hits_mean": 1.0,
                     "red_timeout_alive_advantage_rate": 0.0,
                     "blue_timeout_alive_advantage_rate": 0.0}
        mutual_kill = {"red_elimination_win_rate": 0.5, "blue_dead_mean": 1.5, "red_dead_mean": 2.5,
                       "red_missile_hits_mean": 1.5,
                       "red_timeout_alive_advantage_rate": 0.0,
                       "blue_timeout_alive_advantage_rate": 0.0}
        assert score_record(good_kill) > score_record(mutual_kill), \
            "fewer red losses should score higher"

    def test_blue_timeout_win_penalised(self):
        red_timeout = {"red_elimination_win_rate": 0.0, "blue_dead_mean": 0.0, "red_dead_mean": 1.0,
                       "red_missile_hits_mean": 0.0,
                       "red_timeout_alive_advantage_rate": 0.5,
                       "blue_timeout_alive_advantage_rate": 0.0}
        blue_timeout = {"red_elimination_win_rate": 0.0, "blue_dead_mean": 0.0, "red_dead_mean": 1.0,
                        "red_missile_hits_mean": 0.0,
                        "red_timeout_alive_advantage_rate": 0.0,
                        "blue_timeout_alive_advantage_rate": 0.5}
        assert score_record(red_timeout) > score_record(blue_timeout), \
            "blue timeout win should be penalised below red timeout win"

    def test_empty_record_is_zero(self):
        assert score_record({}) == 0.0

    def test_combined_scores_weighted(self):
        rec_3v2 = {"config": "3v2.yaml", "red_elimination_win_rate": 0.8, "blue_dead_mean": 1.6,
                   "red_dead_mean": 0.5, "red_missile_hits_mean": 1.6,
                   "red_timeout_alive_advantage_rate": 0.0,
                   "blue_timeout_alive_advantage_rate": 0.0}
        rec_5v4 = {"config": "5v4.yaml", "red_elimination_win_rate": 0.4, "blue_dead_mean": 2.0,
                   "red_dead_mean": 1.0, "red_missile_hits_mean": 2.0,
                   "red_timeout_alive_advantage_rate": 0.0,
                   "blue_timeout_alive_advantage_rate": 0.0}
        rec_7v6 = {"config": "7v6.yaml", "red_elimination_win_rate": 0.2, "blue_dead_mean": 3.0,
                   "red_dead_mean": 2.0, "red_missile_hits_mean": 3.0,
                   "red_timeout_alive_advantage_rate": 0.0,
                   "blue_timeout_alive_advantage_rate": 0.0}
        scores = compute_eval_scores([rec_3v2, rec_5v4, rec_7v6])
        assert scores["score_3v2"] > 0
        assert scores["score_5v4"] > 0
        assert scores["score_7v6"] > 0
        assert scores["score_combined"] == pytest.approx(
            0.4 * scores["score_3v2"] + 0.3 * scores["score_5v4"] + 0.3 * scores["score_7v6"]
        )
