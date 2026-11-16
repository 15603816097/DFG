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


## Next phase: fixed-graph convergence diagnosis

After three-sequence controls have been generated, evaluate the ORIGINAL
fixed-noise graph objective at Stage-1 and Stage-2 solutions:

~~~bash
python experiments/isj_supplement/diagnose_fixed_convergence.py \
  --sequence 2011_10_03_drive_0042_sync
~~~

If necessary, run an additional unchanged-graph optimizer pass (this does
**not** overwrite the original fixed-stage or supplemental controls):

~~~bash
python experiments/isj_supplement/diagnose_fixed_convergence.py \
  --sequence 2011_10_03_drive_0042_sync --third-pass
~~~

Results: results/cross_sequence/<sequence>/isj_fixed_convergence_diagnostic.json
and, if requested, factor_graph/ISJ_DIAG_FIXED_THIRD_PASS/.

ATE3D can change even when graph loss barely changes; objective values are
needed to discuss numerical convergence. A third pass with the original
100-iteration LM limit is exploratory and cannot prove global optimality.

## Next phase: residual-adaptive covariance conventional baseline

Run just one sequence for a smoke test:

~~~bash
python -m pytest tests/test_isj_adaptive_common.py -v
python experiments/isj_supplement/run_residual_adaptive.py \
  --sequence 2011_10_03_drive_0034_sync --mode gps
~~~

Then all three sequences, two frozen schemes (GPS-only or GPS/IMU/Camera):

~~~bash
python experiments/isj_supplement/run_residual_adaptive.py \
  --sequence all --mode both
~~~

This benchmark computes the predeclared normalized measurement residuals
using FINAL_S0_fixed/poses.npy, inflates covariance once via
sigma/sigma0 = clip(sqrt(max(1, norm(residual)/c)),1,4), then runs the
same second-stage GTSAM LM optimizer from the SAME Stage-1 poses.
c=sqrt(chi-square 0.95 quantile), 2.795 for 3D and 3.548 for 6D.
LiDAR always keeps its original fixed covariance.
The results appear under factor_graph/ISJ_RESADAPT_*/ and in
isj_residual_adaptive_results.csv. No trained model or ground-truth
positions are used in weight computation. This is a **post-hoc,
conventional baseline**, not a claimed SOTA implementation.

The new script neither imports raw KITTI image data nor re-runs odometry.
It depends on saved Stage-7 measurements and fixed-stage poses, and
requires NumPy and GTSAM in the current Python environment.


## Diagnose fixed-FGO convergence (especially drive 0042)

The original fixed FGO improved considerably when run a second time on
drive 0042. ATE alone does not indicate whether the solver minimized the
same objective or how well the first pass converged. This separate audit
rebuilds the fixed Stage-7 graph with exactly the frozen original factor
types, compares its **graph objectives** for the saved first- and
second-pass pose arrays, and optionally tests another 100 LM iterations.
It does not overwrite any original Stage-7 outputs.

~~~bash
git pull --ff-only
python experiments/isj_supplement/audit_fixed_convergence.py --sequence all
python experiments/isj_supplement/audit_fixed_convergence.py --sequence 2011_10_03_drive_0042_sync --refine
~~~

The resulting diagnostic JSON is stored separately at
results/cross_sequence/<sequence>/isj_fixed_convergence_audit.json.

Interpretation:
- If the second pass substantially reduces the same fixed-graph objective,
  the original first stage was not at the same optimum.
- If ATE improves but objective barely moves, inspect sensitivity and
  trajectory changes: ATE is not the objective optimized by GTSAM.
- If even the second saved fixed trajectory improves materially under
  more LM iterations, treat the baseline solver convergence as unresolved.
- Do not describe observed ATE reductions as proof of a specific optimizer
  failure mode without checking the objective differences.


## Audit learned dynamic-FGO convergence before the next comparison

Especially for 0042, fixed-FGO Stage 1 did not fully settle within its
initial optimization pass. Thus dynamic V5 and V6 second passes must also be
audited on **their own** graph objectives before claiming a fully controlled
comparison. The script below only loads frozen predictions, saved poses, and
the existing measurements; it does not alter original results.

~~~bash
python experiments/isj_supplement/audit_dynamic_convergence.py \
  --sequence 2011_10_03_drive_0042_sync --refine
python experiments/isj_supplement/audit_dynamic_convergence.py \
  --sequence all --refine
~~~

Output: results/cross_sequence/<sequence>/isj_dynamic_convergence_audit.json.
Each model's objective is only comparable across optimization passes of the
SAME model, since GRU and Mamba use different factor covariance values.
The continuation LM pass checks local numerical progress, not global optimality.


## Uniform multi-pass convergence analysis (fixed, GRU/Mamba, controls)

The 0042 dynamic audit found GRU ATE3D changing from 9.8812 m to 8.8219 m
after **one** additional unchanged LM pass; Mamba moved from 8.5036 to
8.5171 m. The audit_dynamic_convergence.py --refine command always starts
from the original saved result and **does not chain** successive invocations.

Use the script below to chain extra optimization passes on each method's own
**original graph**, applying the same predeclared stopping rule. Every method
starts from its existing saved Stage-7 second-stage solution, and the script
never modifies the original Stage-7 or ISJ_CTRL_* results.

~~~bash
python -m pytest tests/test_isj_uniform_convergence.py -v
python experiments/isj_supplement/run_uniform_convergence.py \
  --sequence 2011_10_03_drive_0042_sync --family core \
  --max-extra-passes 4
python experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family core --max-extra-passes 4

# Only after reviewing core convergence, extend to all uncertainty controls:
python experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family controls --max-extra-passes 4
~~~

Default stopping rule: both the relative improvement of the SAME factor
graph objective is <=1e-7 **and** the RMSE of translation changes is <=1 mm,
or else the maximum number of additional passes is exhausted. Each pass
uses the original GTSAM LM maxIterations=100, relativeErrorTol=1e-7.

Each result is under
factor_graph/ISJ_CONVERGENCE_<ORIGINAL_CASE>/, while per-sequence CSV files
are isj_convergence_core_results.csv / isj_convergence_controls_results.csv.
The output reports if a method was NOT stable under this strict criterion.
It does not infer global optimality, equal compute time, or ground truth
independence. Compare ATE only for methods whose numerical convergence
status is appropriately documented; objective numbers across different
noise-weighted graphs are not directly comparable. The analysis remains
post-hoc exploratory because the sequences had been viewed before.


## Reliability-only controls under the same convergence criterion

The original Stage-7 reliability-only ablations live under
factor_graph/ABL_V5_GRU_RELIABILITY_ONLY/ and
factor_graph/ABL_V6_MAMBA_RELIABILITY_ONLY/.
They used the same first-stage fixed warm start, but may have exited
LM before the same strict outer stopping condition.

The uniform convergence script now supports these cases as well.
It reconstructs the original graph by fixing s7.confidence(u)=1.0
for all sensors, retaining frozen reliability predictions and original
noise anchors. It reads the original ABL_*/poses.npy without overwriting
their metrics, and writes new convergence results in ISJ_CONVERGENCE_ABL_*.

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_uniform_convergence.py -v
# First inspect 0042's reliability-only and 8 shuffled/constant controls
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence 2011_10_03_drive_0042_sync \
  --family ablations --max-extra-passes 4

# Then execute remaining sequences with the same fixed protocol
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family ablations --max-extra-passes 4
~~~

Case families: core (fixed2 + 2 dynamic); controls (8 constant/permuted);
reliability (2 existing reliability-only); ablations (all 10 alternatives);
all (13 total). Historical, equal-budget original Stage-7 ATE metrics stay
available separately from the continuation-to-tolerance exploratory results.
