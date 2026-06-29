from scripts.audit_environment_rationality import (
    BLOCKED_FIELDS,
    LAUNCH_DIAG_FIELDS,
    _diag_summary,
    _episode_outcome,
)


def test_diag_summary_reads_real_launch_diag_fields_and_leaves_missing_blank():
    info = {
        "__launch_diag__": {
            "red": {
                "range_ok_pairs": 3,
                "ao_ok_pairs": 2,
                "track_unobserved_blocked": 1,
            }
        }
    }

    row = _diag_summary(info, "red")

    assert row["red_range_ok_pairs"] == 3
    assert row["red_ao_ok_pairs"] == 2
    assert row["red_track_unobserved_blocked"] == 1
    assert row["red_ta_ok_pairs"] == ""
    for key in LAUNCH_DIAG_FIELDS:
        assert f"red_{key}" in row
    for key in BLOCKED_FIELDS:
        assert f"red_{key}" in row


def test_episode_outcome_uses_episode_local_final_counts():
    assert _episode_outcome(red_alive=3, blue_alive=0) == "red_win"
    assert _episode_outcome(red_alive=0, blue_alive=2) == "blue_win"
    assert _episode_outcome(red_alive=3, blue_alive=2) == "draw_or_timeout"
