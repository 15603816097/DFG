#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/dataflow_fix/build_corrected_shared_factor_features_v2.py
python experiments/dataflow_fix/validate_corrected_shared_dataflow.py
python experiments/dataflow_fix/train_corrected_shared_factor_reliability_v2.py
echo "STAGE 1 COMPLETE"
