# ISJ supplementary experiments (Stage 7)

This directory adds **exploratory, post-hoc controls** to the original Stage-7
vehicle pose-estimation experiments. It does not change Stage-7
model weights, noise mapping functions, or original scripts/results.

## Scope

* **Fixed second pass**: rerun the identical fixed-noise FGO from its own saved
  Stage-1 pose solution. This checks whether merely running the optimizer again
  explains apparent improvements.
* **Constant-mean uncertainty**: for each sensor and model, replace the
  post-warmup uncertainty curve with its own temporal mean, leaving its
  reliability predictions unchanged.
* **Time-permuted uncertainty**: shuffle only post-warmup uncertainty values
  within each sensor, preserving the distribution but destroying time alignment.
  Seeds are fixed in advance (11, 23, 47).

Both GRU (V5) and Mamba (V6) use the **same saved Stage-1 fixed-FGO poses** as
their initialization. LiDAR weights remain fixed. No training or test-informed
parameter search is performed by these controls. Their outcomes are
**exploratory** because the Stage-7 test results were already examined.

## Fresh server: first check what is present

Clone the GitHub experimental branch, then from the repository root:

~~~bash
python3 experiments/isj_supplement/preflight.py --require clone
python3 experiments/isj_supplement/preflight.py --require raw
python3 experiments/isj_supplement/preflight.py --require ready
~~~

The checks answer different questions:

* clone: code and V5/V6 checkpoints exist (expected after cloning).
* raw: raw KITTI image, OXTS, Velodyne and calibration paths exist.
* ready: all Stage-7 processed arrays, V5/V6 predictions and
  FINAL_S0_fixed poses exist, so the supplementary experiments can run.

**Important:** dataset/ and results/cross_sequence/ are intentionally
ignored by git. They are **not** downloaded by git clone. Either copy your
previously generated results/cross_sequence/ tree to the server, or copy
KITTI Raw there and reproduce Stage 7. Keep the exact expected layout:
dataset/kitti/2011_10_03/<sequence_name>/.

This repository does not contain a tested universal requirements.txt.
Use a controlled Python environment and record package versions. Stage-7
processing and inference may require NumPy, SciPy, OpenCV, GTSAM, PyTorch,
and a compatible CUDA/PyTorch/Mamba SSM installation. Do not blindly
install CUDA wheels before checking server drivers and PyTorch version.

## If processed data is already available

After the ready check passes and NumPy/GTSAM are installed:

~~~bash
python experiments/isj_supplement/run_controls.py --sequence 2011_10_03_drive_0034_sync --controls fixed2
python experiments/isj_supplement/run_controls.py --sequence 2011_10_03_drive_0034_sync --controls all
python experiments/isj_supplement/run_controls.py --sequence all --controls all
~~~

The first line runs one smoke-test control; second runs all controls for that
sequence; third covers all three sequences and reuses cached control results.

To intentionally rerun **only** the new controls, pass --rerun.
Original FINAL_*, ABL_*, ROBUST_* and stage7 CSV files are never overwritten.
New results are saved to:

~~~text
results/cross_sequence/<sequence>/
    factor_graph/ISJ_CTRL_*/metrics.json
    factor_graph/ISJ_CTRL_*/poses.npy
    isj_controls_results.csv
    isj_controls_protocol.json
~~~

The permuted control defaults to three predetermined seeds; these are
repeated permutations of a **fixed inference signal**, not independent
training runs. Do not report them as model-training variability.

## If Stage 7 must be reconstructed from KITTI Raw

With KITTI data in paths expected by stage7_config.py and dependencies installed:

~~~bash
bash experiments/cross_sequence/run_stage7_local.sh
bash experiments/cross_sequence/run_stage7_v6_inference.sh
for seq in 2011_10_03_drive_0034_sync 2011_10_03_drive_0042_sync 2011_10_03_drive_0047_sync; do
    python experiments/cross_sequence/run_stage7_two_stage.py --sequence "$seq"
done
python3 experiments/isj_supplement/preflight.py --require ready
python experiments/isj_supplement/run_controls.py --sequence all
~~~

Image/stereo and LiDAR odometry recomputation can be expensive. If prior
processed Stage-7 files can be recovered, copying them is preferable to
recomputation. Never change parameters based on these already inspected test
sequences and label the resulting analysis as post hoc.

## Research limitations

1. KITTI OXTS/GPS-derived reference trajectory is not an independent
   external localization ground truth relative to GPS input.
2. All three existing Stage-7 sequences share the same acquisition date;
   they are cross-sequence but not cross-date validation.
3. These controls isolate the contribution of time-varying evidential
   uncertainty, but do not replace new degradation-level studies,
   an independent runtime study, or a tuned adaptive-FGO comparison.
