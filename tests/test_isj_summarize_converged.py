import csv

import pytest

from experiments.isj_supplement.summarize_converged import (
    FIXED_CASE, MODELS, SEEDS, gain, get, summarize_sequence,
)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "continued_ATE3D", "stationary_under_rule"]
        )
        writer.writeheader()
        writer.writerows(rows)


def case_row(name, value, stable=True):
    return {
        "case": name,
        "continued_ATE3D": str(value),
        "stationary_under_rule": str(stable),
    }


def test_get_rejects_incomplete_and_invalid_status():
    with pytest.raises(KeyError):
        get({}, "not_here")
    with pytest.raises(ValueError):
        get({"bad": case_row("bad", 1.0, stable="unknown")}, "bad")
    with pytest.raises(ValueError):
        get({"bad": case_row("bad", -1.0)}, "bad")


def test_gain_signed():
    assert gain(10.0, 8.0) == pytest.approx(20.0)
    assert gain(10.0, 11.0) == pytest.approx(-10.0)


def test_sequence_summary_preserves_unstable_fixed_flag(tmp_path):
    seq = "2011_10_03_drive_0034_sync"
    folder = tmp_path / seq
    core = [case_row(FIXED_CASE, 3.27, False)]
    ablations = []
    for model, names in MODELS.items():
        core.append(case_row(names["dynamic"], 2.7))
        ablations.extend([
            case_row(names["reliability"], 2.8),
            case_row(names["mean"], 3.0),
        ])
        ablations.extend([
            case_row(names["permutation_prefix"] + str(seed), 2.9 + seed * 0.001)
            for seed in SEEDS
        ])
    write_rows(folder / "isj_convergence_core_results.csv", core)
    write_rows(folder / "isj_convergence_ablations_results.csv", ablations)
    result = summarize_sequence(tmp_path, seq)
    assert len(result) == 2
    assert {r["model"] for r in result} == {"GRU", "Mamba"}
    for row in result:
        assert row["fixed2_stable"] is False
        assert row["all_cases_stable_under_rule"] is False
        assert row["dynamic_improvement_vs_reliability_pct"] == pytest.approx(
            100 * (2.8 - 2.7) / 2.8
        )
