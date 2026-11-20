#!/usr/bin/env bash
# ONE COMMAND: frozen GRU t+3 forecast vs no-forecast persistence,
# prediction accuracy + same-information FGO + convergence.
# Run with: bash experiments/isj_supplement/run_h3_gru_now.sh
# Uses ORIGINAL frozen checkpoints, 0016 processed data, and fixed FGO.
# No retraining, no raw preprocessing, no overwrite of earlier results.

set -Eeuo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

echo "=== H3 core experiment; GRU; $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Python: $(command -v python)"

python -m pytest \
  tests/test_isj_cross_date_future_vs_persistence.py \
  tests/test_isj_future_h3_fgo.py \
  -v 2>&1 | tee logs/isj_h3_gru_tests.log

python -u experiments/isj_supplement/audit_future_prediction_vs_persistence.py \
  --dataset dev --model gru --device cpu \
  2>&1 | tee logs/isj_h3_dev_gru_accuracy.log

python -u experiments/isj_supplement/audit_future_prediction_vs_persistence.py \
  --dataset 2011_09_30_drive_0016_sync --model gru --device cpu \
  2>&1 | tee logs/isj_h3_0016_gru_accuracy.log

python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage predict --model gru --device cpu \
  2>&1 | tee logs/isj_h3_0016_gru_predict.log

python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage audit --model gru \
  2>&1 | tee logs/isj_h3_0016_gru_audit.log

python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage fgo --model gru \
  2>&1 | tee logs/isj_h3_0016_gru_fgo.log

python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage converge --model gru \
  2>&1 | tee logs/isj_h3_0016_gru_converge.log

echo "=== H3 GRU EXPERIMENT COMPLETE ==="
echo "Accuracy: logs/isj_h3_0016_gru_accuracy.log"
echo "FGO: logs/isj_h3_0016_gru_fgo.log"
echo "Convergence: logs/isj_h3_0016_gru_converge.log"
