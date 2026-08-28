from __future__ import annotations

"""
Compatibility runner.

This version reuses your already-working
experiments/factor_graph/run_degraded_four_sensor_predictive_fg.py
without modifying that file.

It imports the module, replaces only its prediction/output directory constants,
then calls main(). This guarantees that the exact degraded-measurement loading
path that already produced ATE3D=2.850053 is reused.
"""

import importlib.util
import os
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


ORIGINAL_RUNNER = os.path.join(
    ROOT,
    "experiments",
    "factor_graph",
    "run_degraded_four_sensor_predictive_fg.py",
)

NEW_PREDICTION_DIR = os.path.join(
    ROOT,
    "results",
    "conservative_predictive_reliability",
)

NEW_OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "conservative_predictive_four_sensor_fg",
)


def _set_first_existing(module, names, value):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, value)
            return name
    return None


def main():
    print("=" * 104)
    print("CONSERVATIVE PREDICTIVE FOUR-SENSOR FG - COMPATIBILITY RUNNER")
    print("=" * 104)

    if not os.path.isfile(ORIGINAL_RUNNER):
        raise FileNotFoundError(ORIGINAL_RUNNER)
    if not os.path.isdir(NEW_PREDICTION_DIR):
        raise FileNotFoundError(
            "Conservative reliability directory missing. Run:\n"
            "python experiments/reliability/build_conservative_reliability.py"
        )

    spec = importlib.util.spec_from_file_location(
        "_existing_degraded_predictive_runner",
        ORIGINAL_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import existing degraded predictive runner.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pred_name = _set_first_existing(
        module,
        (
            "PREDICTION_DIR",
            "PREDICT_DIR",
            "RELIABILITY_DIR",
            "PREDICTIVE_RELIABILITY_DIR",
        ),
        NEW_PREDICTION_DIR,
    )
    out_name = _set_first_existing(
        module,
        (
            "OUTPUT",
            "OUTPUT_DIR",
            "RESULT_DIR",
            "SAVE_DIR",
        ),
        NEW_OUTPUT_DIR,
    )

    if pred_name is None:
        raise RuntimeError(
            "Could not find prediction-directory constant in "
            "run_degraded_four_sensor_predictive_fg.py. "
            "Use the direct runner or inspect that file."
        )
    if out_name is None:
        raise RuntimeError(
            "Could not find output-directory constant in "
            "run_degraded_four_sensor_predictive_fg.py."
        )

    print(f"Patched {pred_name} -> {NEW_PREDICTION_DIR}")
    print(f"Patched {out_name} -> {NEW_OUTPUT_DIR}")
    print("Existing source file is NOT modified.")

    if not hasattr(module, "main"):
        raise RuntimeError("Existing degraded predictive runner has no main().")

    module.main()


if __name__ == "__main__":
    main()
