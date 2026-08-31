#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STAGE 1.6 / V4: SENSOR-SPECIFIC PREDICTIVE RELIABILITY"
echo "============================================================"

python experiments/dataflow_fix/train_sensor_specific_reliability_v4.py
python experiments/dataflow_fix/check_sensor_specific_reliability_v4.py

echo
echo "============================================================"
echo "STAGE 1.6 COMPLETE"
echo "============================================================"
