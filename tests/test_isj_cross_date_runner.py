import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cross_sequence import stage7_config as cfg
from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, date_for, validate_destination, crossdate_bindings,
    result_is_complete, prediction_preflight, MODELS,
)


SEQ = "2011_09_30_drive_0016_sync"


def test_only_cross_date_synchronized_names_allowed():
    assert date_for(SEQ) == "2011_09_30"
    with pytest.raises(ValueError):
        date_for("2011_10_03_drive_0034_sync")
    for bad in ("", "../2011_09_30_drive_0016_sync",
                "2011_09_30_drive_0016_extract",
                "2011_09_30_drive_0016_sync/../../results"):
        with pytest.raises(ValueError):
            date_for(bad)


def test_crossdate_output_is_isolated_from_historical_results():
    path = validate_destination(SEQ, RESULT_ROOT)
    assert path.name == SEQ
    assert path.parent == RESULT_ROOT.resolve()
    with pytest.raises(ValueError):
        validate_destination(SEQ, cfg.ROOT / "results" / "cross_sequence")
    with pytest.raises(ValueError):
        validate_destination(SEQ, cfg.ROOT / "results")


def test_in_memory_bindings_are_restored():
    old_test = cfg.TEST_SEQUENCES
    old_date = cfg.DATE_DIR
    old_out = cfg.OUT_ROOT
    old_lookup = cfg.sequence_out

    import experiments.cross_sequence.run_stage7_factor_graph as s7
    old_s7_out = s7.sequence_out
    with crossdate_bindings(SEQ, RESULT_ROOT, "fgo"):
        assert cfg.DATE_DIR.name == "2011_09_30"
        assert cfg.TEST_SEQUENCES == (SEQ,)
        assert cfg.sequence_out(SEQ) == RESULT_ROOT.resolve() / SEQ
        assert s7.sequence_out(SEQ) == RESULT_ROOT.resolve() / SEQ
        with pytest.raises(ValueError):
            s7.sequence_out("2011_10_03_drive_0034_sync")
    assert cfg.TEST_SEQUENCES == old_test
    assert cfg.DATE_DIR == old_date
    assert cfg.OUT_ROOT == old_out
    assert cfg.sequence_out is old_lookup
    assert s7.sequence_out is old_s7_out


def test_cached_fgo_requires_both_pose_and_metric_files(tmp_path):
    assert not result_is_complete(tmp_path, "FINAL_S0_fixed", 2)
    folder = tmp_path / "factor_graph" / "FINAL_S0_fixed"
    folder.mkdir(parents=True)
    np.save(folder / "poses.npy", np.tile(np.eye(4), (2, 1, 1)))
    with pytest.raises(ValueError):
        result_is_complete(tmp_path, "FINAL_S0_fixed", 2)
    (folder / "metrics.json").write_text(
        json.dumps({"ATE3D": 1.0}), encoding="utf-8"
    )
    assert result_is_complete(tmp_path, "FINAL_S0_fixed", 2)


def test_prediction_check_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run --stage gru first"):
        prediction_preflight(tmp_path, ("gru",))
    assert MODELS["mamba"]["source"] == "v6"
