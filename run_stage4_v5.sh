#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pytest -q tests/test_reliability_uncertainty_v5.py
python experiments/uncertainty_v5/train_reliability_uncertainty_v5.py
python experiments/uncertainty_v5/check_v5_outputs.py
echo
echo "V5 training complete. Send the full terminal output before factor-graph integration."
