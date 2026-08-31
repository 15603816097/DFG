#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/factor_graph/run_stage3_confidence_aware_mapping.py
