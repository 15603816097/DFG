# Stage 6 — GPS development comparison and repeated CUDA seeds

This stage does NOT modify any model, dataset, or earlier checkpoint.
The compare script validates source hashes, horizon labels, split boundaries,
and paired predictions from the same 680 validation frames on 0027.
All results remain development-only, not independent paper tests.

## Current seed-42 results (user verified)

| Head | GRU MAE | Mamba MAE | Relative MAE decrease |
|---|---:|---:|---:|
| t | 0.181271 | 0.131580 | 27.41% |
| t+1 | 0.188393 | 0.129633 | 31.19% |
| t+2 | 0.194372 | 0.136340 | 29.86% |

GRU: 114630 params, CPU, best epoch 1.
Mamba: 139590 params, V100 CUDA, best epoch 8.
Models therefore are not parameter-count matched, and devices differ.

## Comparison mechanics

- Requires stored dataset and provenance SHA256 to agree with frozen GPS candidate.
- Reads exactly 3180..3859 validation anchors, 680x3 predictions and labels.
- Recomputes MAE and RMSE from prediction archives and verifies metrics JSON.
- Enforces same run settings and same seed for both models.
- Computes per-head MAE/RMSE reduction and mean evidence proxy.
- Reports Spearman correlation between evidence proxy U and absolute error.
- Estimates nominal 80% coverage from reconstructed Beta output.
- Provides exploratory paired 32-frame moving-block bootstrap intervals;
  these are not independent-drive confidence intervals.
- No file writes by default; --write-report creates a new write-once report directory.

## Run seed 42 comparison

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo TEST_EXIT=$?
python -u -m experiments.isj_v7.compare_dev_gps --seeds 42
echo PAIR_READONLY_EXIT=$?
# Optionally save once after checking the read-only result:
python -u -m experiments.isj_v7.compare_dev_gps --seeds 42 --write-report
echo PAIR_REPORT_EXIT=$?
~~~

## Matched-device repeat-seed comparison

On the now functioning V100, run BOTH GRU and Mamba with seeds 43,44,45.
Both use the same CUDA device, hidden=96, layers=2, epochs=40, patience=8,
batch-size=64. Parameter counts will still differ, which must be reported.
Never overwrite the prior seed42 CPU GRU or GPU Mamba checkpoints.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
mkdir -p logs
for seed in 43 44 45; do
  for backend in gru mamba; do
    dest="results/isj_v7/dev_gps_baseline_v1/${backend}/seed_${seed}"
    logfile="logs/isj_v7_${backend}_seed_${seed}.log"
    if [ -e "$dest" ]; then
      echo "EXISTING_RUN_SKIP: $dest"
      continue
    fi
    echo "TRAIN backend=$backend seed=$seed"
    python -u -m experiments.isj_v7.train_gps_dev \
      --backend "$backend" --device cuda \
      --seed "$seed" --hidden 96 --layers 2 \
      --epochs 40 --patience 8 --batch-size 64 \
      > "$logfile" 2>&1
    rc=$?
    echo "TRAIN_EXIT backend=$backend seed=$seed rc=$rc"
    if [ "$rc" -ne 0 ]; then
      tail -n 80 "$logfile"
      break 2
    fi
    tail -n 8 "$logfile"
  done
done
python -u -m experiments.isj_v7.compare_dev_gps --seeds 43 44 45
~~~

The next scientific step is capacity-matched controls and independent
state-estimation reference evaluation, not declaring publication-ready
ATE or uncertainty from these development-only results.
