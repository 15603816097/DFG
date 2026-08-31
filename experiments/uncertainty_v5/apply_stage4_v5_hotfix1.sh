#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STAGE 4 V5 HOTFIX 1"
echo "============================================================"

python -m py_compile experiments/uncertainty_v5/train_reliability_uncertainty_v5.py
echo "Syntax PASS"

pytest -q tests/test_reliability_uncertainty_v5.py
python experiments/uncertainty_v5/train_reliability_uncertainty_v5.py
python experiments/uncertainty_v5/check_v5_outputs.py
