import csv

import pytest

from experiments.isj_supplement.summarize_sensor_u_factorial import (
    MASK_TO_GROUP,
    MODEL_CASES,
    U_MASKS,
    extract,
    process_sequence,
    source_case,
)


def test_eight_sensor_uncertainty_combinations_are_complete():
    assert set(U_MASKS) == {
        "000", "001", "010", "011", "100", "101", "110", "111"
    }
    assert set(MASK_TO_GROUP) == set(U_MASKS)


def test_full_factorial_source_case_names_match_original_results():
    assert source_case("GRU", "000") == "ABL_V5_GRU_RELIABILITY_ONLY"
    assert source_case("Mamba", "111") == "FINAL_S2_v6_all_mamba"
    assert source_case("GRU", "100") == "ISJ_SENSU_GRU_ONLY_GPS"
    assert source_case("Mamba", "011") == "ISJ_SENSU_MAMBA_DROP_GPS"
    assert source_case("GRU", "101") == "ISJ_SENSU_GRU_DROP_IMU"


def test_extract_rejects_nonconverged_status_format():
    with pytest.raises(ValueError):
        extract({"continued_ATE3D": "2", "stationary_under_rule": "unknown"},
                "bad")
    with pytest.raises(ValueError):
        extract({"continued_ATE3D": "-1", "stationary_under_rule": "True"},
                "bad")


def test_full_factorial_summary_and_conditional_gps_gain(tmp_path):
    seq = "2011_10_03_drive_0047_sync"
    folder = tmp_path / seq
    folder.mkdir(parents=True)
    by_group = {k: [] for k in set(MASK_TO_GROUP.values())}
    for model in MODEL_CASES:
        for mask in U_MASKS:
            by_group[MASK_TO_GROUP[mask]].append({
                "case": source_case(model, mask),
                "continued_ATE3D": 1.0 if mask == "000" else
                  2.0 if mask == "100" else
                  3.0 if mask == "011" else
                  4.0 if mask == "111" else 2.5,
                "stationary_under_rule": "True",
            })
    for group, rows in by_group.items():
        path = folder / f"isj_convergence_{group}_results.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summaries, details = process_sequence(tmp_path, seq)
    assert len(summaries) == 2
    assert len(details) == 16
    for row in summaries:
        assert row["all_8_cases_stationary"] is True
        # Relative error increases by 100% when switching on GPS alone.
        assert row["GPS_U_gain_when_IMU_CAMERA_off_pct"] == pytest.approx(-100.)
        # With IMU+Camera on, GPS makes 3m become 4m, gain -33.33%.
        assert row["GPS_U_gain_when_IMU_CAMERA_on_pct"] == pytest.approx(
            100 * (3. - 4.) / 3.
        )
