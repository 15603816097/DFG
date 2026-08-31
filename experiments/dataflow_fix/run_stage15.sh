#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STAGE 1.5: SAME-SOURCE FACTOR LABEL REBUILD"
echo "============================================================"

python experiments/dataflow_fix/build_same_source_factor_labels_v3.py
python experiments/dataflow_fix/validate_same_source_factor_labels_v3.py
python experiments/dataflow_fix/train_same_source_factor_reliability_v3.py
python experiments/dataflow_fix/check_same_source_predictions_v3.py

echo
echo "============================================================"
echo "STAGE 1.5 COMPLETE"
echo "============================================================"
echo "Send the full terminal output to ChatGPT."
