# V7 GPS-only GRU vs Mamba — development baseline

> Scope: one **already explored** KITTI 0027 synthetic GPS-injection train/validation candidate. Not a final IEEE Sensors Journal result. Final-supervision policy `final_training_approved=false` remains unchanged.

## Inputs and safeguards

- Input: `results/isj_v7/dev_gps_0027_same_source_v1/{gps_train_val.npz,provenance.json}`. The loader enforces exact source SHA256 lineage, archive array hashes, shapes, dtypes, train/VAL anchors and labels.
- Model sees only `gps_features` (past/current degraded GPS derived features). The offline clean GPS used to compute synthetic supervision is neither loaded nor exported to training inputs.
- Shared architecture protocol: 64 input time steps, 21 features, three prediction heads at time offsets 0, 1 and 2, fixed train-only median and IQR and `[-20,20]` normalized feature clipping.
- Both backends: same input projection, Beta evidential output head, same horizon-weighted Beta negative log-likelihood plus weak evidence penalty, AdamW, early stopping based on **validation** NLL, minibatch shuffle using a backend-independent fixed generator seed. Same hidden widths/layer counts *do not guarantee equal parameter counts*; actual counts are printed.
- Prediction reliability `alpha/(alpha+beta)`; concentration/evidence proxy `U=2/(alpha+beta)`. `U` is **not automatically calibrated model uncertainty nor sensor observation covariance**.
- The exported development archive contains only 0027 train and val; the 0027 internal test frames 3862..4543 are omitted. Reserved 0016 and 0020 are not accessed. Do not use validation results as independent generalization evidence.
- `--backend mamba` must import and run a **real Mamba** layer from `mamba_ssm` and uses NVIDIA CUDA; it fails clearly if unavailable. There is no GRU fallback.

## Run on the Ubuntu server

```bash
cd /root/projects/DFG
conda activate dfg-isj
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -v
echo "V7_TEST_EXIT=$?"

# First, check which compute backends actually exist:
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
try:
    from mamba_ssm import Mamba
    print("mamba_ssm_import=PASS")
except Exception as exc:
    print("mamba_ssm_import=FAIL", repr(exc))
PY

# Both smoke commands are non-training, non-writing forward/backward tests:
python -u -m experiments.isj_v7.train_gps_dev --backend gru --smoke-only
python -u -m experiments.isj_v7.train_gps_dev --backend mamba --smoke-only

# Train each backend SEPARATELY only when its smoke test passes;
# both use exactly the same predeclared run settings:
python -u -m experiments.isj_v7.train_gps_dev \
  --backend gru --seed 42 --hidden 96 --layers 2 \
  --epochs 40 --patience 8 --batch-size 64 \
  2>&1 | tee logs/isj_v7_dev_gps_gru_seed42.log

python -u -m experiments.isj_v7.train_gps_dev \
  --backend mamba --seed 42 --hidden 96 --layers 2 \
  --epochs 40 --patience 8 --batch-size 64 \
  2>&1 | tee logs/isj_v7_dev_gps_mamba_seed42.log
```

**Important shell note:** with `set +o pipefail`, a command piped to `tee` can fail while the pipeline exits 0. Inspect logs for `V7_GPS_DEV_COMPLETE` and check that the corresponding `metrics.json` exists before counting a model as trained. Do not rerun the same seed after a failed partial run without inspecting the output path; the trainer deliberately refuses to overwrite.

If Mamba/CUDA is missing, do NOT install arbitrary mismatched libraries or report the GRU run as a Mamba comparison. Provide the exact environment import error and GPU details, then fix the environment specifically.

## Generated development artifacts

- `results/isj_v7/dev_gps_baseline_v1/gru/seed_42/` and `.../mamba/seed_42/` contain `best.pt`, `history.json`, `metrics.json`, `val_predictions.npz`.
- Output metadata includes dataset NPZ and manifest SHA256, Git commit, PyTorch version, model backend, train settings, best epoch, parameter count, and only validation MAE/RMSE, train-mean and fixed-reliability baselines.
- Source datasets and V5/V6 checkpoints are immutable. Each backend/seed path is write-once; use a **different seed** for repeated experiments, and keep the same matched seed list across backends.
- This stage does not run FGO, calculate ATE/RPE, prove uncertainty calibration, establish cross-drive generalization, or approve final training.

## Immediate next decision gate

If both models train and the baseline reports look reasonable, compare `h0/h1/h2` validation MAE/RMSE against train-mean and always-one baselines, and check uncertainty-vs-error information **conditional on reliability**. An impressive MAE or small U alone is not a publishable improvement. Only then integrate **pre-update** factor-graph covariance logic and add a scientifically independent evaluation reference.
