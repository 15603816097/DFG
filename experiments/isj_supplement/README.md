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


## Convergence-controlled publication table (no recomputation)

Once all sequences have isj_convergence_core_results.csv and
isj_convergence_ablations_results.csv:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_summarize_converged.py -v
python experiments/isj_supplement/summarize_converged.py
~~~

Outputs results/cross_sequence/isj_converged_summary.csv. It combines exact
saved ATE3D for Fixed2, GRU/Mamba full, reliability-only, constant uncertainty,
and the three shuffled-uncertainty seeds, reporting signed gains. It also
checks every case's \`stationary_under_rule\` field and preserves the caveat
that 0034 fixed2 has not crossed the stricter 1-mm shift threshold after four
additional passes. This is an exploratory convergence-controlled table;
do not silently replace or overwrite the original fixed-budget Stage-7 table.

## Conventional residual-adaptive baseline: recommended next sequence

The prior Stage-7 / uncertainty experiments are ready for conventional
residual-adaptive comparison. To test the predeclared GPS-only and
GPS/IMU/Camera residual-inflation schemes, without needing KITTI Raw:

~~~bash
python -m pytest tests/test_isj_adaptive_common.py -v
python -u experiments/isj_supplement/run_residual_adaptive.py \
  --sequence 2011_10_03_drive_0034_sync --mode both

# After inspecting the first run:
python -u experiments/isj_supplement/run_residual_adaptive.py \
  --sequence all --mode both
~~~

The baseline first performs one frozen covariance inflation using residuals
of the saved Fixed1 trajectory, then starts an equal initial second-stage LM
pass. Those original saved adaptive baseline results must be kept separate
from the post-hoc uniform continuation results. For apples-to-apples
convergence control on adaptive cases, assess their own graph objective
and continuation just as for GRU/Mamba; avoid claiming solver-equivalent
accuracy when convergence differences have not been tested.


## Convergence-control of traditional residual-adaptive baselines

After run_residual_adaptive.py --sequence all --mode both has produced
ISJ_RESADAPT_* original pose arrays and nested metrics.json files, use:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_uniform_convergence.py -v
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family adaptive --max-extra-passes 4
~~~

The audit **reconstructs the identical residual-adaptive graph** from
FINAL_S0_fixed/poses.npy for each sensor set and verifies that the
reconstructed objective at the SAVED solution matches the original
\`graph_final_objective\` to tight tolerance, before any continuation.
This catches accidental comparisons across differently constructed graphs.
It then applies exactly the same extra LM pass and stopping test used for
all other FGO families, without overwriting original saved results.

The new convergence files are stored in
factor_graph/ISJ_CONVERGENCE_ISJ_RESADAPT_*/ and
isj_convergence_adaptive_results.csv. Together with the existing
isj_convergence_core_results.csv, these permit a converged-method comparison.
The original residual-adaptive results are a *single predeclared outer*
covariance update: they are NOT a comprehensive literature baseline.
If the first-stage residual covariance inflation harms GPS-related ATE,
investigate sensor-specific inflation fractions from the ORIGINAL
ISJ_RESADAPT_*/metrics.json (sigma_statistics) rather than retuning based
on these already-inspected test sequences.


## Descriptive sensor-mechanism analysis (especially 0047)

All six existing conventional residual-adaptive baseline cases passed uniform
continuation with no additional changes. Their original, frozen sigma
statistics showed:
- 0034: 3.50% of GPS factors inflated; only ~0.02% of IMU and no camera.
- 0042: 27.27% of GPS factors inflated; 10.13% of camera factors in the
  GPS/IMU/camera scheme; no IMU factors inflated.
- 0047: ~2.99% of GPS factors inflated, no IMU or camera factors inflated.
  Hence the GPS-only and GPS/IMU/Camera variants have identical output.

To investigate why Dynamic U improves 0034/0042 over Reliability Only but
is worse than Reliability Only on 0047, run the read-only descriptor:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_uncertainty_mechanism.py -v
python -u experiments/isj_supplement/diagnose_uncertainty_mechanism.py \
  --sequence 2011_10_03_drive_0047_sync
python -u experiments/isj_supplement/diagnose_uncertainty_mechanism.py \
  --sequence all
~~~

This produces:
- \`results/cross_sequence/isj_uncertainty_mechanism_summary.csv\` with
  per-encoder/sensor correlations between evidence uncertainty and
  per-frame **gain** (Reliability Only position-error norm minus Dynamic U
  position-error norm), high/low uncertainty quartile ATE differences,
  and how uncertainty moves factor noise sigmas relative to reliability-only.
- \`results/cross_sequence/isj_uncertainty_frames_<drive>_<encoder>.csv\`
  for frame-by-frame visualization, with model uncertainty/reliability,
  noise sigma values, and both position-error magnitudes.

IMPORTANT: Quantile ATE differences are descriptive within a selected
sensor uncertainty quartile; they DO NOT attribute per-frame pose error
causally to the selected sensor. The optimization couples timestamps and
sensor factors. A correlation can be undefined if uncertainties are nearly
constant. No reference trajectory is used to select factors or retune
uncertainty: OXTS/GPS-derived ground truth is used for diagnostic evaluation
only. These results remain post-hoc exploratory and limited to one KITTI date.


## Targeted sensor-U intervention: why does drive 0047 degrade?

The first three-sequence mechanism audit found:
- On drive 0047, full Dynamic U is worse than Reliability Only (GRU:
  1.9881 vs 1.7891 m; Mamba: 2.0274 vs 1.8496 m).
- GRU camera uncertainty is negatively correlated with framewise gain
  (rho about -0.456); GPS also has a negative association.
- These associations cannot identify which sensor's uncertainty mapping
  causes the observed difference because the graph couples time/sensors.

To test **controlled noise mapping interventions**, without retraining or
changing reliability predictions, we added the following:

- \`DROP_CAMERA\`: only Camera uses u=0 (reliability-only mapping); GPS/IMU
  retain original uncertainties. Likewise for \`DROP_GPS\`, \`DROP_IMU\`.
- \`ONLY_CAMERA\`: Camera retains original uncertainty; GPS/IMU use u=0.
  Likewise for \`ONLY_GPS\`, \`ONLY_IMU\`.
- Setting u=0 covers all frames, including the 63-frame encoder warm-up,
  consistent with the prior Reliability Only ablation.
- All cases reuse FINAL_S0_fixed/poses.npy as their identical first-stage
  warm-start; LiDAR noise remains fixed.

First run the six leave-one-out controls on 0047, test, and then enforce
the **same numeric continuation stopping rule** as the main methods:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_sensor_u_ablation.py -v
python -u experiments/isj_supplement/run_sensor_uncertainty_ablation.py \
  --sequence 2011_10_03_drive_0047_sync --strategy drop
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence 2011_10_03_drive_0047_sync \
  --family sensor_u_drop --max-extra-passes 4
~~~

If the above succeeds, explore the complementary \`only\` controls to
study interactions, and repeat the intervention on all three sequences:

~~~bash
python -u experiments/isj_supplement/run_sensor_uncertainty_ablation.py \
  --sequence all --strategy all
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family sensor_u_drop --max-extra-passes 4
python -u experiments/isj_supplement/run_uniform_convergence.py \
  --sequence all --family sensor_u_only --max-extra-passes 4
~~~

Raw outputs:
\`factor_graph/ISJ_SENSU_*/poses.npy\`, \`metrics.json\`,
\`isj_sensor_uncertainty_<strategy>_results.csv\`, and matching protocol JSON.
Continued results:
\`factor_graph/ISJ_CONVERGENCE_ISJ_SENSU_*/poses.npy\`, \`metrics.json\`,
\`isj_convergence_sensor_u_drop_results.csv\`, and
\`isj_convergence_sensor_u_only_results.csv\`.

Interpretation: difference to full Dynamic U for one dropout estimates the
**conditional effect of that intervention in the current graph**, not a
causal effect of the sensor itself, and effects need not add up. Since these
comparisons were devised after seeing drive 0047, label them as exploratory.
Do not tune sensor gating on the same test sequence and call it held-out.


## Final 2^3 sensor-U factorial comparison (all completed)

Only-GPS, Only-IMU, Only-Camera, Drop-GPS, Drop-IMU, Drop-Camera,
Full-Dynamic and Reliability-Only now form the full 8-case U on/off
design (all three sensors' reliability signals retained). Across
three KITTI sequences and two frozen encoders there are 48 method
instances, each audited using the same numeric convergence criterion.

The complementary \`only\` results revealed:
- drive 0034: Only GPS U is slightly more accurate than Full Dynamic U
  for both encoders.
- drive 0042: Full Dynamic U is more accurate than Only GPS U for both.
- drive 0047: Only GPS U is worse than Reliability Only, consistent with
  Drop GPS U improving full-model ATE.

Generate publication-oriented **exact CSV summaries** from the prior
result files, without training or re-running any FGO:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_sensor_u_factorial_summary.py -v
python experiments/isj_supplement/summarize_sensor_u_factorial.py
~~~

Outputs:
- \`results/cross_sequence/isj_sensor_u_factorial_summary.csv\`:
  one row per sequence and encoder, all eight ATE3D values, conditional
  GPS/IMU/Camera U effects (percent ATE gain) in other-U-off vs
  other-U-on contexts, and convergence flags;
- \`results/cross_sequence/isj_sensor_u_factorial_long.csv\`:
  all 48 records with case identities, U-on/off masks and convergence flags.

The summary will report a failed convergence status if encountered; do
not hide it or declare optimality based solely on a numeric stopping
criterion. Conditional on/off comparisons are **not additive sensor
contributions**, since sensor weights interact through FGO. These
post-hoc experiments must be followed by out-of-date / cross-date
independent validation, and the GPS-linked OXTS reference remains a
known evaluation limitation. Do not tune a new gating strategy using
these same explored test sequences and label it blind held-out evidence.


## Cross-date KITTI Raw preflight (before modifying the evaluation pipeline)

All 48 existing sensor-U factorial runs were summarized with
\`summarize_sensor_u_factorial.py\`; six per-sequence/encoder summaries and
48 detailed rows passed the predeclared convergence rule. To assess whether
this evidence generalizes beyond 2011_10_03, FIRST inventory available
cross-date data instead of modifying the 2011_10_03 test definitions:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_cross_date_raw.py -v
python experiments/isj_supplement/audit_cross_date_raw.py
# Optional machine-readable inventory:
python experiments/isj_supplement/audit_cross_date_raw.py \
  --json-out results/cross_sequence/isj_cross_date_raw_inventory.json
~~~

This read-only preflight scans \`dataset/kitti/<date>/<date>_drive_<id>_sync/\`,
checks \`oxts\`, \`velodyne_points\`, \`image_02\`, \`image_03\` frame files
and timestamp counts, and the three date-specific calibration text files.
A directory is only "READY_RAW" if the raw file layout is complete and
its acquisition date differs from 2011_10_03. File readiness is not an
actual validation run or proof of correct calibration/clock synchronization.

IMPORTANT: The existing scripts \`stage7_config.py\`,
\`build_sequence_data.py\`, \`infer_v5_gru.py\`, \`infer_v6_mamba.py\`,
and \`run_stage7_two_stage.py\` restrict processing to exactly three drives
from 2011_10_03. Do NOT simply change DATE_DIR or TEST_SEQUENCES in-place
and risk overwriting historical result folders. Once genuine cross-date Raw
data is present, implement a separate date-aware evaluation entry point,
use date-specific calibration, keep frozen V5/V6 checkpoints and noise
anchors, and run all comparison methods under the same solver stopping rule.

The current processed KITTI experimental errors are judged against a
GPS/OXTS-derived local trajectory while degraded GPS is part of the input,
so the reference remains GPS-linked, not independent ground truth.
Also, the original 0027 degradation plan is resampled on the test drives;
clearly label all future results synthetic-degradation generalization and
do not claim real-world sensor-failure validation unless independently
measured sensor degradation is evaluated.


## Cross-date 2011-09-30 / 0016: isolated frozen-model pilot

The synchronized KITTI Raw drive
\`dataset/kitti/2011_09_30/2011_09_30_drive_0016_sync\` is now present
on the user's server: 279 frames each for OXTS, Velodyne, image_02 and
image_03, with matching timestamp counts and the date-specific calibration
files. The raw inventory found \`READY_RAW\`; this proves only file layout,
not quality or synchronization accuracy.

Use **one new entrypoint**; never run the old Stage-7 shell scripts on a
different acquisition date. The historical \`stage7_config.py\` remains
unchanged. New outputs are isolated under
\`results/isj_cross_date/2011_09_30_drive_0016_sync/\`.

The pipeline reuses the frozen original 0027 degradation recipe,
odometry computation, feature extraction, checkpoints/normalizers,
Stage-7 factor and covariance equations, and LM optimizer.
No model training, date-specific noise retuning or GT-based trajectory
initialization. The evaluation reference is OXTS/GPS-derived and not
independent of the GPS used as an input. This drive is an initial
cross-date **pipeline pilot**, not sufficient evidence of generalization
or naturally occurring sensor failure performance.

### A. Validate and preprocess the KITTI Raw sequence

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_runner.py tests/test_isj_cross_date_raw.py -v
python experiments/isj_supplement/run_cross_date.py --stage inspect
mkdir -p logs
set -o pipefail
python -u experiments/isj_supplement/run_cross_date.py --stage preprocess \
  2>&1 | tee logs/isj_cross_date_0930_preprocess.log
~~~

Preprocessing recomputes physical LiDAR and degraded stereo visual
odometry for 279 frames. It can consume substantial CPU/time; cached
processed results are reused when valid, and missing/partial data fail
rather than being silently ignored. It preserves the frozen normalized
development degradation schedule.

### B. Run the frozen V5 GRU and, when possible, the V6 Mamba

~~~bash
python -u experiments/isj_supplement/run_cross_date.py --stage gru \
  2>&1 | tee logs/isj_cross_date_0930_gru.log
python -u experiments/isj_supplement/run_cross_date.py --stage mamba \
  2>&1 | tee logs/isj_cross_date_0930_mamba.log
~~~

Mamba requires the official \`mamba_ssm\` implementation and its matching
frozen checkpoint/runtime; if unavailable on the current machine, do
GRU first, then run Mamba later in its existing supported environment.
Do not substitute another Mamba implementation/architecture to manufacture
a test result. The commands do NOT retrain either model.

### C. Run the identical Stage-7 FGO using one common fixed warm start

For GRU only:

~~~bash
python -u experiments/isj_supplement/run_cross_date.py \
  --stage fgo --model gru \
  2>&1 | tee logs/isj_cross_date_0930_gru_fgo.log
python -u experiments/isj_supplement/run_cross_date.py \
  --stage converge --model gru --max-extra-passes 4 \
  2>&1 | tee logs/isj_cross_date_0930_gru_convergence.log
~~~

After Mamba predictions are available, run \`--model mamba\` for
\`fgo\` and \`converge\`, or \`--model both\` for full combined reporting.
\`FINAL_S0_fixed\` is reused from cache, never overwritten.
For each model, the full Dynamic U and Reliability Only graph use
exactly the same fixed-FGO warm start. Same unchanged reliability
predictions, fixed covariance anchors, fixed LiDAR noise and same
separate uniform LM stopping rule as previous three sequences.

The convergence controller saves all continued poses into separate
\`ISJ_CONVERGENCE_*\` subfolders, and reports
\`STABLE\`/\`NOT_STABLE\` explicitly. All original Stage-7 outputs stay
untouched. This first-phase pilot covers fixed FGO, GRU, Mamba and
Reliability Only; constant/shuffled-U controls may be added subsequently,
without altering pretrained weights or hyperparameters.

Safety: new runner refuses to process 2011_10_03 as a cross-date drive
and refuses to point its output root at the old
\`results/cross_sequence\`. A new model's previously absent predictions
may be added after another model's FGO, but existing predictions cannot
be overwritten if FGO outputs already exist.


### D. Cross-date temporal-information controls: mean U vs shuffled U

Once BOTH frozen V5/GRU and V6/Mamba predictions and their FGO are
available, use the separate cross-date control entrypoint:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_controls.py tests/test_isj_cross_date_runner.py -v
mkdir -p logs
set -o pipefail
python -u experiments/isj_supplement/run_cross_date_controls.py \
  --stage fgo --model both --controls all --seeds 11,23,47 \
  2>&1 | tee logs/isj_cross_date_0930_controls_fgo.log

# Only after FGO succeeds:
python -u experiments/isj_supplement/run_cross_date_controls.py \
  --stage converge --model both --controls all --seeds 11,23,47 \
  --max-extra-passes 4 \
  2>&1 | tee logs/isj_cross_date_0930_controls_convergence.log
~~~

This adds 8 NEW FGO cases (2 model-specific constant-mean U and
6 model-specific sensor-wise temporally permuted U using predeclared seeds).
For each, all original reliability values r[t] and uncertainty values at
frames 0..62 remain unchanged. Mean or shuffle is applied to U[t] from
frame 63 onward, preserving each sensor's marginal U distribution under
shuffling and the GRU/Mamba frozen feature/normalization/checkpoints.
All controls start from the same saved FINAL_S0_fixed trajectory; stage-2
GTSAM LM has unchanged graph factors and covariance equations. Every run
is checked under the same 1e-7 relative objective and 0.001 m trajectory
shift stopping rule, with separate ISJ_CONVERGENCE_ directories. Existing
fixed, GRU, Mamba, and their Reliability Only results are NEVER replaced.

Controls are essential because Reliability Only forces confidence=1 at
all timestamps, whereas full dynamic U uses confidence=1-U; specifically
U=1 for the first 63 unpredicted frames sets dynamic-U confidence=0, and
thus fixed sigma, while Reliability Only still applies a predictive sigma.
GRU and Mamba also have DIFFERENT default warmup reliabilities (0.5 vs
1.0). Therefore Dynamic U vs Reliability Only alone does NOT isolate
the value of temporally varying U, nor does early/late segment ATE on
the jointly optimized trajectory establish causal effects.

The control script preserves the full-dynamic warmup and reliability
sequences for its temporal-information comparison. The controls are
exploratory post-hoc, not blinded held-out evaluation; mean-U does not
preserve the empirical distribution shape, and permuted U destroys
ordering rather than all possible correlations. If results merit a
revised method, evaluate that revised method on a NEW untouched drive.
The OXTS/GPS metric reference is correlated with the GPS input.


### E. Cross-date Beta-valid U-only permutation controls

The earlier cross-date constant-mean/shuffled-U controls keep r[t] but can
produce (r,U) pairs that no longer correspond to legal Beta evidence
(alpha=2*r/U, beta=2*(1-r)/U, requiring alpha,beta>=1). That fact limits
their interpretation as temporal-information-only ablations.

A new, separate runner implements a stricter control: it swaps U[t] between
two post-warmup timestamps ONLY when both resulting Beta pairs are valid.
All r values, each sensor's exact U multiset, and U at t=0..62 stay
unchanged. The script reports the fraction of actually changed U values
and refuses ineffective swaps (<5% changed). The method is a constrained
local-swap random walk, NOT a uniform random permutation; it does not
preserve the full joint (r,U) distribution.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest \
  tests/test_isj_cross_date_beta_swap.py \
  tests/test_isj_cross_date_controls.py \
  tests/test_isj_cross_date_runner.py -v

# Read-only preflight: verify zero invalid Beta pairs, changed percentages,
# accepted swaps and mean absolute uncertainty perturbations.
python -u experiments/isj_supplement/run_cross_date_beta_swap.py \
  --stage audit --model both --seeds 11,23,47

# Only if audit passes and materially perturbs U:
set +e
set -o pipefail
python -u experiments/isj_supplement/run_cross_date_beta_swap.py \
  --stage fgo --model both --seeds 11,23,47 \
  2>&1 | tee logs/isj_cross_date_0930_beta_swap_fgo.log

# Only if FGO returns code 0:
python -u experiments/isj_supplement/run_cross_date_beta_swap.py \
  --stage converge --model both --seeds 11,23,47 \
  --max-extra-passes 4 \
  2>&1 | tee logs/isj_cross_date_0930_beta_swap_convergence.log
~~~

This adds six new, uniquely named FGO cases and their separate
ISJ_CONVERGENCE_* continuations under the uniform LM rule. It does
not alter checkpoints, original predictions, former eight controls,
or preexisting GRU/Mamba/Fixed runs.

Interpretation: if original Dynamic U beats Beta-valid shuffled U on
this pilot, that supports sensitivity to original temporal assignment
beyond the **invalid Beta-combinations** confound, conditional on
perturbation strength. It still does not isolate causation or establish
calibration, robust generalization, or independent position truth.


### F. Matched first-63-frame Reliability Only control

The original Dynamic-U case uses U=1 (confidence=0; fixed factor sigma)
for the first 63 frames (zero-based t<63) before a complete encoder window.
The original Reliability Only case forces confidence=1 even there. This is
an important non-U-temporal confound, particularly in V5/GRU IMU weights.

The following *new* control matches Dynamic U's weights during warmup;
after the encoder window becomes valid (t>=63), it uses Reliability Only
weights. It keeps frozen r[t], sensor measurements, LiDAR, all other
factor/noise code, the fixed-FGO start and LM convergence rule unchanged.

The control supplies U_eff=1 for t<63 and U_eff=0 for t>=63 solely as a
**covariance control switch**, not a Beta uncertainty prediction.
The existing original predictions and cases are never modified.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set -o pipefail
git pull --ff-only
python -m pytest tests/test_isj_cross_date_warmup_matched.py \
  tests/test_isj_cross_date_runner.py -v

python -u experiments/isj_supplement/run_cross_date_warmup_matched.py \
  --stage audit --model both

mkdir -p logs
python -u experiments/isj_supplement/run_cross_date_warmup_matched.py \
  --stage fgo --model both \
  2>&1 | tee logs/isj_cross_date_0930_warmup_matched_fgo.log
# Require FGO exit 0 before convergence:
python -u experiments/isj_supplement/run_cross_date_warmup_matched.py \
  --stage converge --model both --max-extra-passes 4 \
  2>&1 | tee logs/isj_cross_date_0930_warmup_matched_convergence.log
~~~

Adds only these two new names:
- ISJ_CTRL_GRU_WARMUP_MATCHED_REL_ONLY
- ISJ_CTRL_MAMBA_WARMUP_MATCHED_REL_ONLY

Each saved convergence case starts from its own saved 2nd-stage output,
under the same numerical LM tolerance; the original FGO trajectory and
Dynamic-U FGO solutions are untouched. Both methods' warmup factor
weights now match within a model. This is an exploratory post-hoc pilot
on one OXTS/GPS-linked synthetic-degradation drive, not independent GT
or prospective validation.


### G. Next step: READ-ONLY prediction-quality audit (do not tune on 0016)

The matched-warmup Reliability Only controls outperformed original Dynamic U
for both V5 and V6 on cross-date 2011_09_30/0016, which falsifies an
unqualified claim that U always improves the current factor graph.
Before changing noise-mapping parameters, evaluate what frozen r and U
actually capture:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_prediction_quality.py -v
mkdir -p logs
set +e
set -o pipefail
python -u experiments/isj_supplement/audit_cross_date_prediction_quality.py \
  --sequence 2011_09_30_drive_0016_sync \
  2>&1 | tee logs/isj_cross_date_0930_prediction_quality.log
~~~

This is a read-only analysis of saved \`sensor_data.npz\` and frozen
\`predictions/v5\`/\`predictions/v6\` arrays; no checkpoint, FGO result,
or noise anchor is modified. The GPS target is the exactly known injected
synthetic position displacement, computed from saved
\`clean_gps_local\` and \`degraded_gps_local\`. For GPS/IMU/Camera,
the injected severity curve is used only as a SYNTHETIC PROXY, not
the original training target \`current_factor_reliability\`.
The audit reports Pearson/Spearman associations of severity or known GPS
error with 1-r (predicted impairment) and with U (Beta evidential
lack-of-evidence), plus severity-stratified statistics. Excludes the first
63 pre-window frames.

This is not formal statistical calibration: U=2/(alpha+beta) is not
automatically measurement variance; a GPS-linked OXTS source is shared
between corrupted GPS input and the reference; time series are
autocorrelated, and correlations on a previously explored 279-frame
sequence do not establish generalization. Do not select model parameters
from this audit and then report 0016 as an untouched held-out test.
After diagnostics, seek an unexamined cross-date drive and evaluate
the frozen protocol prospectively, then develop any revised mapping on
a development/validation split only.


### H. Read-only IMU injected-error and factor-time-alignment audit

The 0016 prediction-quality audit found unexpectedly high IMU reliability at
medium synthetic severity (both frozen GRU and Mamba). Severity is not the
factor reliability training label. Inspect the **actual applied synthetic**
gyro and acceleration differences against original KITTI OXTS values before
diagnosing a model failure:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_imu_injected_error.py \
  tests/test_isj_cross_date_prediction_quality.py -v
mkdir -p logs
set +e
set -o pipefail
python -u experiments/isj_supplement/audit_cross_date_imu_injected_error.py \
  --sequence 2011_09_30_drive_0016_sync \
  2>&1 | tee logs/isj_cross_date_0930_imu_injected_error.log
~~~

The Stage-7 FGO's orientation between factor currently uses the degraded
gyroscope at i, while the uncertainty/reliability covariance is read at
i+1. The audit compares both predictor[t] vs injected gyro error[t] and
predictor[i+1] vs injected gyro error[i], as well as acceleration error
for context (acceleration is not directly used in this Stage-7 IMU factor).
No output artifacts or model parameters are modified. Correlations are
descriptive, not formal calibration; original OXTS is not an independent
high-precision inertial reference.


### I. V3 label-origin audit and cross-date V3-style factor-target diagnosis

V3 label generation (\`experiments/dataflow_fix/build_same_source_factor_labels_v3.py\`)
calls \`compute_oracle_factor_reliability()\`. IMU targets are calculated
from the degraded gyro's rotation increment against the relative rotation
of the development fixed-factor-graph poses, then mapped as
\`clip(exp(-(error_deg / 2.0)**2), 0.01, 1.0)\`. It is **not**
a direct label for \`norm(degraded_gyro - clean_gyro)\`. The user's
4544-frame V3 training-file audit passed a max formula difference of
2.98e-08; this validates label construction, not learned predictions.

V5's default data path is:
\`results/predictive_factor_reliability_v3_same_source/factor_reliability_training_data.npz\`.
V6 training has a configurable \`--data\` defaulting to
\`data/factor_reliability_training_data.npz\`; the actual command used
for the frozen checkpoint must be checked separately before asserting
V5 and V6 used identical training data.

To compare 0016 frozen r predictions to the *same-formula* proxy reliability,
use the new **read-only** diagnostic (no GTSAM optimize, no checkpoint write):

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set -o pipefail
git pull --ff-only
python -m pytest tests/test_isj_cross_date_factor_targets.py \
  tests/test_isj_cross_date_runner.py -v
mkdir -p logs
python -u experiments/isj_supplement/audit_cross_date_factor_targets.py \
  --sequence 2011_09_30_drive_0016_sync \
  2>&1 | tee logs/isj_cross_date_0930_factor_targets.log
~~~

This uses the saved local \`FINAL_S0_fixed/poses.npy\` as the reference
rotation/pose trajectory, with the existing frozen Stage-7 sensor factors
and GPS data, and reports MAE/RMSE/correlations versus reference labels,
plus RMSE of a constant reference-mean predictor. It excludes initial
63 no-full-window frames. The reference is formed using the measurements
and is NOT independent truth; the original development "fixed FG"
reference may have different graph parameters or initialization from this
0016 reference, so this diagnoses a similar label construction on a new
drive, not formally equivalent or independently calibrated oracle labels.
This is exploratory analysis on an already repeatedly inspected drive.
If it reveals poor IMU label tracking, inspect training-vs-0016
feature and label distributions and the V6 training-data provenance
before retraining any network or changing test noise anchors.


### J. CORE PAPER TEST: frozen H=3 forecasting vs no-forecast persistence in FGO

This addresses the **anticipation hypothesis**, NOT whether Mamba beats GRU.

The V3 labels and both frozen training dataset builders encode:
- head0 at source time t: current reliability r[t]
- head1 at source time t: three-step-ahead reliability r[t+3]
- head2 at source time t: reliability r[t+2]

For target factor time j=t+3, the **predictive** control uses head1[t]
and the **no-forecast persistence** control uses head0[t]. Both have access
ONLY to sensor features through time t. Factor targets j=0..65 use exactly
fixed sigmas; j>=66 uses r-only sigmas with the same frozen anchors.
This is a clean horizon-specific comparison with U held constant as a
CONTROL SWITCH (Ueff=0 after target 66), rather than a test of Beta U.

New scripts:
- \`audit_future_prediction_vs_persistence.py\`: prediction RMSE/MAE,
  head1(t)->t+3 vs head0(t)->t+3 on 0027 dev tail or 0016 existing
  exploratory reference, **read-only**.
- \`run_future_h3_fgo.py\`: save immutable frozen head1 predictions, audit
  target alignment, run pairwise GTSAM FGO from SAME stored Fixed-FGO
  warm start, then run identical uniform convergence checks.
- \`tests/test_isj_cross_date_future_vs_persistence.py\`
- \`tests/test_isj_future_h3_fgo.py\`

All outputs are isolated to
\`results/isj_cross_date/<drive>/forecast_h3/<source>/\` and
\`factor_graph/ISJ_H3_{GRU,MAMBA}_{PERSIST,PREDICT}_R_ONLY/\`.
Never overwrite old FGO data or train checkpoints.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_future_vs_persistence.py \
  tests/test_isj_future_h3_fgo.py -v
mkdir -p logs

# A: CPU GRU forecast accuracy (dev tail and previously analyzed 0016).
python -u experiments/isj_supplement/audit_future_prediction_vs_persistence.py \
  --dataset dev --model gru --device cpu \
  2>&1 | tee logs/isj_h3_dev_gru_accuracy.log
python -u experiments/isj_supplement/audit_future_prediction_vs_persistence.py \
  --dataset 2011_09_30_drive_0016_sync --model gru --device cpu \
  2>&1 | tee logs/isj_h3_0016_gru_accuracy.log

# B: Save frozen head1[t] at target time t+3; verify cached head0.
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage predict --model gru --device cpu

# C: Print no-leakage/warmup check for both paired controls.
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage audit --model gru

# D: FGO, SAME measurements, SAME warm start, SAME U handling.
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage fgo --model gru \
  2>&1 | tee logs/isj_h3_0016_gru_fgo.log

# E: Uniform numerical convergence on the same graph.
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage converge --model gru \
  2>&1 | tee logs/isj_h3_0016_gru_converge.log
~~~

Run model=\`mamba\` in an environment containing the original
\`mamba_ssm\` and frozen Mamba checkpoint (do not silently fallback to GRU).
The model-specific \`--stage predict\` needs PyTorch+official Mamba;
subsequent \`--stage audit/fgo/converge\` can run in \`dfg-isj\`, because
they read cached head1 predictions and only need GTSAM+NumPy:

~~~bash
# In original Mamba-capable environment:
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage predict --model mamba --device cuda
# Back in the GTSAM/dfg-isj environment:
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage audit --model both
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage fgo --model both
python -u experiments/isj_supplement/run_future_h3_fgo.py \
  --stage converge --model both
~~~

Note: if the frozen V6 checkpoint was trained on a non-V3 data source,
its intended head1 horizon must be verified from original logs before
claiming it represents r[t+3]. If Mamba's original backend cannot be
loaded, report the GRU H3 experiment first and retain the frozen Mamba
files for later official inference; never substitute an alternate model.

This is a past-explored, 279-frame synthetic-fault pilot with an
OXTS/GPS-shared reference; it does not establish ISJ-level generalization.
For new 0020, freeze the protocol and run full preprocessing+inference+
H3 comparison before looking at sequence performance. Do not tune using
the new sequence and relabel it held-out.


### J. Cross-date conventional residual reweighting: first publication decision gate

The previously implemented conventional \`adaptive_fg_common.build_factor_graph\`
already supplies a **one-step residual-based covariance inflation** baseline,
but the old runner was restricted to same-date Stage-7 outputs. The new
\`run_cross_date_residual_baseline.py\` uses the *same* frozen Stage-7 factors,
fixed LiDAR covariances and first-stage Fixed FGO poses on the isolated
cross-date experiment. It offers GPS-only and GPS/IMU/Camera reweighting.

On the **already-inspected exploratory** 0016 sequence, run:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_residual_baseline.py \
  tests/test_isj_cross_date_runner.py -v

python -u experiments/isj_supplement/run_cross_date_residual_baseline.py \
  --sequence 2011_09_30_drive_0016_sync --stage fgo --mode both

python -u experiments/isj_supplement/run_cross_date_residual_baseline.py \
  --sequence 2011_09_30_drive_0016_sync --stage converge --mode both \
  --max-extra-passes 4
~~~

New results only:
\`results/isj_cross_date/<seq>/factor_graph/ISJ_RESADAPT_GPS/\`,
\`.../ISJ_RESADAPT_GPS_IMU_CAMERA/\` and
\`.../ISJ_CONVERGENCE_ISJ_RESADAPT_*/\`.
The frozen fixed, GRU, Mamba, and matched-warmup reliability-only outputs
are not touched; existing baseline result directories fail closed on
metadata mismatch. FGO and convergence stage summaries have their own
CSV files. Both stages can reuse consistent, complete cached results.

**Important:** this is **not** a robust Huber/Cauchy GTSAM loss or
a causal online correction. The fixed-stage poses come from the entire
input sequence and the same measurements; the baseline only inflates
selected Gaussian factor sigmas based on stage-one residuals once.
It is a relevant traditional **two-stage** baseline, not an independent
ground-truth validation. The 0016 dataset was already used for model
analysis; its result is a decision/diagnostic gate only. Do not tune or
select the proposed method on 0016 and then label it held-out.

Before a credible ISJ submission, compare the candidate against this
baseline, frozen and warmup-matched GRU/Mamba reliability-only, and
appropriately implemented conventional robust-kernel FGO, across
multiple newly obtained KITTI drives under a prospectively frozen protocol.
All methods need consistent graph topology, initialization, optimization
stopping/convergence checks, and synthetic-fault seeds. Explain that
GPS/OXTS reference and corrupted GPS input share their source.


### K. Before crediting residual-adaptive improvement: add a SAME fixed second pass

On exploratory cross-date drive \`2011_09_30_drive_0016_sync\`, the frozen
one-step residual-adaptive GPS-only and GPS/IMU/Camera variants both
obtained ATE3D **2.779791 m**, from a first-stage Fixed FGO result of
**2.811762 m**, with identical final objective **569.411554826**.
**Do not attribute the 0.032 m improvement to residual reweighting
before comparing against an unchanged FIXED noise graph from the same
first-stage warm start.** It may simply be an extra LM solve.
The tiny difference from the separately reported earlier fixed result
2.811416 m reflects a different saved stage/continuation; do not silently
mix them.

Run the isolated fixed second pass and inspect sigma inflation fractions:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_cross_date_fixed_second_pass.py \
  tests/test_isj_cross_date_residual_baseline.py -v
python -u experiments/isj_supplement/run_cross_date_fixed_second_pass.py \
  --sequence 2011_09_30_drive_0016_sync --stage audit
python -u experiments/isj_supplement/run_cross_date_fixed_second_pass.py \
  --sequence 2011_09_30_drive_0016_sync --stage fgo
python -u experiments/isj_supplement/run_cross_date_fixed_second_pass.py \
  --sequence 2011_09_30_drive_0016_sync --stage converge \
  --max-extra-passes 4
python -u experiments/isj_supplement/run_cross_date_fixed_second_pass.py \
  --sequence 2011_09_30_drive_0016_sync --stage compare
~~~

The script prints per-sensor fraction of factors that actually had
sigma inflated, mean/max sigma multipliers from the existing
residual baselines, and a matched unchanged fixed-noise second
optimization from the exact same first-stage Fixed FGO poses. It
reports raw and separately convergence-continued ATEs. Outputs
are isolated in the new \`ISJ_CTRL_CROSSDATE_FIXED_SECOND_PASS\`
and \`ISJ_CONVERGENCE_ISJ_CTRL_CROSSDATE_FIXED_SECOND_PASS\`
directories. Cached results are reused only after provenance validation.

This is a **methodological confound check**, not the new paper method,
and 0016 is exploratory. Even if a learned reliability method beats
this one-step residual approach, a conventional robust loss kernel or
better-tuned adaptive estimator is still needed for broader claims.


### L. Next fixed experiment: true Huber/Cauchy M-estimator GPS versus all sensors

After the exploratory 0016 fixed2 and residual comparison, run frozen
robust-kernel baselines that use the actual GTSAM
\`noiseModel.Robust\` (not the residual-based sigma inflation above).
Reuses the preexisting Stage-7 robust thresholds exactly:
Huber \`k=1.345\`, Cauchy \`k=2.3849\`.
Each is evaluated both with GPS-only M-estimation and with GPS/IMU/Camera
M-estimation. LiDAR remains fixed Gaussian, preserving original graph
topology and physical factor measurements.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest \
    tests/test_isj_cross_date_robust_kernels.py \
    tests/test_isj_cross_date_runner.py -v
python -u experiments/isj_supplement/run_cross_date_robust_kernels.py \
    --sequence 2011_09_30_drive_0016_sync \
    --stage fgo --kernel both --scope both
python -u experiments/isj_supplement/run_cross_date_robust_kernels.py \
    --sequence 2011_09_30_drive_0016_sync \
    --stage converge --kernel both --scope both \
    --max-extra-passes 4
~~~

Outputs are isolated and not overwritten at:
\`results/isj_cross_date/2011_09_30_drive_0016_sync/factor_graph/\`
under \`ISJ_ROBUST_HUBER_GPS\`, \`ISJ_ROBUST_HUBER_ALL\`,
\`ISJ_ROBUST_CAUCHY_GPS\`, \`ISJ_ROBUST_CAUCHY_ALL\` and separately
\`ISJ_CONVERGENCE_<name>\`. Existing output files must agree with
the frozen method specification or execution fails closed. Continuation
uses 100 LM iterations per pass, objective relative tolerance 1e-7,
and position RMSE shift <=1 mm, for up to four extra passes.

CAUTION: GTSAM's generic M-estimator on the 6-D Pose3 GPS pseudo-prior
acts on the factor's aggregate whitened residual; its rotation part is
already weakened by 1e6 rotation sigma. IMU robustification similarly
affects rotation residual predominantly given its weak translational
noise term. This is not a GPS-only 3D positional custom factor.
These are exploratory comparisons because 0016 has already been
inspected repeatedly; they are not an untouched evaluation.

The code was added through repository tooling, NOT numerically executed on
the user's GTSAM server. Run the unit tests and check the log before using
any numerical results in a manuscript. Do not tune robust cutoff values on
0016 or claim they are universally optimal. Next, implement one evidence-
guided calibration method on development data and test it prospectively on
unseen KITTI drives. No additional neural-network retraining is needed
unless the selected method actually requires it.


### M. Post-robust-baseline decision: build only ONE calibration innovation

On the repeatedly inspected \`2011_09_30_drive_0016_sync\`, frozen robust
GPS Huber/Cauchy and GPS/IMU/Camera Huber/Cauchy baselines have now been
completed on the server (unit tests 10/10 passed). Their converged ATE3D:
Huber GPS 2.768891m; Huber All 2.778402m; Cauchy GPS 2.780994m;
Cauchy All 2.784688m. Traditional residual-adaptive is 2.779791m,
fixed-second-pass continuation is 2.811071m. The GRU/Mamba
reliability-only numbers 1.608051m/1.695980m are from earlier
warmup-matched exploratory runs and must be checked against their
saved convergence protocols in a final table. These are NOT unseen
validation data and must not be used to select hyperparameters.

A deliberately small, **not yet experimentally validated**
\`src/reliability/evidence_feedback_calibration.py\` research prototype
implements a soft-count Beta-evidence feedback update of the frozen
reliability mean. It does NOT assume that Beta vacuity U is calibrated
measurement variance. The feedback score MUST come from a defensible,
pre-update measurement-consistency measurement. The unit tests verify
Beta count algebra, valid evidence range, missing feedback and first
63-frame warmup exclusion; numerical tests do not prove localization
benefit, independent residual validity or causal deployability.

Before writing any Stage-7 FGO integration or tuning on KITTI 0016,
check **development 0027** artifact availability WITHOUT changing
the original files:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_evidence_feedback_calibration.py \
    tests/test_isj_cross_date_robust_kernels.py -v
python -u experiments/isj_supplement/audit_dev_calibration_readiness.py
~~~

The development audit lists V3 labels, old 0027 sensor measurements and
factor files, existing frozen V5/V6 checkpoints and predicted arrays.
Do not confuse V3 label consistency with the actual V6 checkpoint
training data lineage: V6 checkpoint creation source does not save the
\`--data\` path, so the original training invocation/log must be found
before asserting comparable provenance.

The V3 future target corresponds to t+3 but V5/V6 datasets use it
as the first future target. In a FAST paper plan based only on current
\`r_t\` heads, do **not** claim validated future prediction or preemptive
trajectory weighting. Make current-reliability calibration the sole
candidate innovation. Next engineering milestone is independent
cross-sensor residual construction and development-only factor-graph
integration, with reliability-only and feedback-only controls. Never
use test GT to calculate weights. No new 0016 tuning; prospectively
select multiple new KITTI sequences for independent validation.


### N. Development-only GPS/IMU cross-modal feedback sanity check (no FGO)

All 25/25 original 0027 development artifacts were confirmed present on
the server, and the evidence calibration and robust-kernel unit tests passed
10/10. The next *engineering* check is the crucial previously missing
feedback observation signal: consistency between degraded GPS/IMU and the
physical LiDAR relative odometry, without any use of OXTS ground truth,
clean GPS, oracle reliability labels or first-stage fixed graph poses in
the generation of the feedback itself.

\`src/reliability/cross_modal_consistency.py\` computes:
* GPS: compare length of displacement over the previous **5** frame gaps
  with the length of the chained relative LiDAR displacement over the
  same interval. This orientation-free heuristic does **NOT** detect
  every directional error, a constant offset, or slowly drifting bias.
* IMU: for pair \`t-1 -> t\`, compare gyro-integrated SO(3) rotation with
  the corresponding LiDAR between-factor rotation, publishing quality at
  **target frame t**. No spurious frame t-1/t shift.
* Both convert a squared normalized discrepancy to a monotone quality
  score using predeclared chi-square 95% cutoffs; use frozen Stage-7
  nominal sigmas as an **approximation**, not true calibrated innovation
  covariance. A validated LiDAR pair is required. Masked records must
  contribute **no** Beta feedback, even though q is set to a harmless
  unit placeholder in those records.

\`experiments/isj_supplement/audit_dev_cross_modal_feedback.py\` runs
this construction on 2011_10_03_drive_0027_sync degraded GPS/IMU and
the existing physical LiDAR factors, then applies the already-added
Beta soft-count update to the **frozen** GRU/Mamba reliability and
uncertainty arrays. It prints coverage, NIS and q distributions, fraction
of nontrivial feedback, r delta, and descriptive (not independent)
agreement with V3 labels. V3 labels and synthetic severity are read
**only after feedback creation**, and used strictly for diagnostics.
Neither prediction source nor dataset is trained or changed here.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest \
    tests/test_isj_cross_modal_consistency.py \
    tests/test_isj_evidence_feedback_calibration.py -v
python -u experiments/isj_supplement/audit_dev_cross_modal_feedback.py
~~~

This **prints only, never writes files**, and does not run GTSAM,
train models, or evaluate on repeatedly inspected 0016. The repo
code has not been executed on the user's server: run tests first and
inspect any shape/key errors before integration. If feedback coverage,
validity or variation is inadequate, do not claim an innovation; first
correct the measurement-only consistency mechanism on development data.
The V6 training-data lineage is still unverified; the 0027 predictions
are from a sequence used for training and cannot support final held-out
accuracy claims.

The next stage, only if the above sanity audit succeeds, is a fresh,
isolated development 0027 FGO with four identical sensor factors and
four predeclared ablations: reliability-only, feedback-only,
reliability+feedback without U, and Beta-guided reliability+feedback.
No tuning on 0016 or newly held-out drives, and no rewriting historical
Stage-6/Stage-7 outputs. Validate initialization, reference alignment
and solver convergence on each new case before comparing ATE.


### O. Correcting a real development audit failure: unpredicted V5/V6 tail frames

On 2026-11-18 the server ran the read-only development consistency audit:
tests 10/10 passed, 0027 had 4544 frames and GPS feedback coverage
4481/4481 when naively including all frames t>=63.
The **first** GRU GPS run worsened reference-label MAE from 0.121978
to 0.153665. Mean GPS feedback q=0.985697; only 3.2359% had q<1.
This shows that the original **two-sided** Beta soft-count update,
which pushes unreliable predictions toward 1 whenever q=1, is
**not a validated beneficial correction**. Do not use it in FGO or
claim positive calibration evidence. GPS speed/translation-distance
agreement can miss constant GPS bias and directional errors.

The later Mamba GPS run initially stopped with
\`ValueError: Beta evidence invalid (alpha/beta < 1)\`.
Source review identified a likely deterministic cause:
\`mamba_v6/train.py\` pre-fills all r/U with **ones**, and
\`mamba_v6/dataset.py\` trains/predicts samples only through
\`usable_end=n-horizon+1\` exclusive, so with n=4544, horizon=3
**frames 4542 and 4543 are still r=1,U=1 placeholders**.
These are impossible valid BetaV6 pairs but NOT necessarily a broken
checkpoint. V5 also leaves its last 2 frames unpredicted, with its
own placeholders (r=0.5,U=1). The prior audit incorrectly treated
all t>=63 as valid.

The audit now scores only frames **63..4541**, excludes these
unpredicted tail placeholders in BOTH models, and prints the excluded
tail values. It checks the exact structural Beta r/U constraints for
all remaining predictions and prints any invalid frame indices without
silently clipping/editing predictions. If additional invalid genuine
network outputs are found, the affected sensor/model calibration is
skipped, rather than hiding potential file provenance mismatch.
Original files and original evaluation outputs remain unchanged.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_beta_pair_integrity.py \
    tests/test_isj_evidence_feedback_calibration.py \
    tests/test_isj_cross_modal_consistency.py -v
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_cross_modal_feedback.py \
    > logs/isj_dev_feedback_after_tail_mask.log 2>&1
echo "AUDIT_EXIT=$?"
tail -n 100 logs/isj_dev_feedback_after_tail_mask.log
~~~

This is a **data/protocol repair and hypothesis falsification**,
not evidence that the main proposed method is now successful.
Before final manuscript experiments, freeze a new feedback mechanism
that treats measurement agreement as *lack of contradiction*, not as
proof of absolute GPS reliability. Existing V6 training input provenance
remains unconfirmed.


### P. Decision gate after failed two-sided Beta feedback on development 0027

The 2026-11-18 read-only server audit returned all 14 tests PASS, 4479
valid prediction samples per sensor and 0 invalid Beta pairs for both
GRU V5 and Mamba V6 (tails excluded). On 0027, the **two-sided**
\`r*=(2r+kUq)/(2+kU)\` with GPS/IMU-LiDAR consistency q worsened
reconstructed-label MAE:
GPS GRU 0.121810->0.153622 and Mamba 0.118842->0.146142;
IMU GRU 0.078406->0.079700 and Mamba 0.071694->0.080056.
GPS q<1 in only 3.2373% of 4479 valid frames; IMU q<1 in **0%**.
Therefore DO NOT integrate this calibration into FGO or publish it
as a positive innovation. Agreement without contradiction cannot
justify moving reliability toward 1. These are development-data
diagnostics; label reference is not independent truth.

The same audit found positive Pearson association of U and
absolute reliability-label error:
GPS GRU +0.67735, Mamba +0.61210; IMU GRU +0.54930,
Mamba +0.29067. These aggregate values may be driven by training
frames or time distribution shifts. Before investing in ANY further
U-dependent covariance mapping, audit its error-risk ranking
separately on the exact original chronological train/validation/test
index splits:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest \
    tests/test_isj_dev_uncertainty_risk_by_split.py -v
python -u experiments/isj_supplement/audit_dev_uncertainty_risk_by_split.py
~~~

The new audit reports Pearson/Spearman between U and
\`abs(predicted_r - V3_proxy_target)\` and rank-AUROC for
identifying the within-split highest 20% absolute errors. It uses
**original V5/V6 training-code split boundaries**, which are not
identical because V5 computes its split cutoffs from full n while V6
uses usable_end=n-horizon+1, and excludes the final two unpredicted
frames from both. It requires only numpy and saved results, writes
no files and does not rerun FGO.

Do not call any part of the same 0027 drive a new independent
external test. If U ranks prediction error only on training frames,
the evidential-U contribution should be dropped or substantially
redesigned. If U ranks error on the internal holdout and prospectively
on newly collected KITTI drives, it may support a more carefully
calibrated selective-confidence module, but this alone is not an
ISJ-worthy localization benefit. Freeze model provenance, signal
definitions, base baselines, and multiple unseen sequences before
making a final claim. No arbitrary threshold tuning on explored 0016.


### Q. New 0020 cross-sequence archive: preserve as prospective evaluation

2026-11-18 split-aware 0027 diagnostics (server tests 4/4 PASS)
confirm that GPS Beta vacuity U still ranks high-error reliability
predictions on the **internal** chronological TEST span:
GRU U-error AUC=0.8729 and Mamba AUC=0.8793. The Mamba IMU
internal-TEST AUC is weaker (0.6287); Camera Mamba=0.6685.
These results are promising GPS-specific **risk-ranking** evidence,
NOT proof U improves localization or transfers to an independent drive.
Previous **two-sided U+q Beta feedback degraded MAE on all four
GPS/IMU comparisons** and is NOT accepted as the paper method.

A newly uploaded \`2011_09_30_drive_0020_sync.zip\` is intended as a
**prospective held-out KITTI drive**. Before fitting a new method on
this drive, first inspect/install the raw data safely, WITHOUT looking
at its labels, prediction metrics, ATE or doing exploratory parameter
searches. Data preparation can be done independently of new method
selection as long as the original 0027 degradation plan and all
preprocessing settings remain frozen.

The helper \`install_kitti_sync_zip.py\` previews the ZIP layout by
default and **only writes** under a NEW drive folder with the explicit
\`--install\` flag. It handles either a \`<drive>/...\` or
\`<date>/<drive>/...\` ZIP root, rejects duplicate/traversal/symlink
entries, checks available disk, extracts into a temporary sibling,
and renames atomically. It never overwrites an existing drive folder
or results. Date-level KITTI calibration files are separate from most
sync ZIPs; reuse the official **same-date** 2011_09_30 files already
on disk with 0016, and NEVER borrow different-date calibrations.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_install_kitti_sync_zip.py -v

ZIP=$(find /root /home/vipuser -maxdepth 8 -type f \
    -name '2011_09_30_drive_0020_sync.zip' -print -quit 2>/dev/null)
echo "ARCHIVE=$ZIP"
if [ -n "$ZIP" ]; then
  # Read-only preview; no archive extraction.
  python -u experiments/isj_supplement/install_kitti_sync_zip.py \
    --archive "$ZIP"
fi
~~~

After verifying the preview and free space, opt in to installation:

~~~bash
python -u experiments/isj_supplement/install_kitti_sync_zip.py \
  --archive "$ZIP" --install
python -u experiments/isj_supplement/run_cross_date.py \
  --sequence 2011_09_30_drive_0020_sync --stage inspect
~~~

If \`--stage inspect\` reports \`READY_RAW\`, stop and retain the drive
for the FINAL frozen-method evaluation. Do **not** tune the new
GPS evidence gating or choose parameters based on 0020 results.
For broader final validity, obtain another genuinely separate KITTI
drive/date (e.g. 2011_09_26) rather than treating two drives from
2011_09_30 as independent acquisition conditions. All test claims
must disclose KITTI OXTS/GPS reference non-independence and synthetic
corruptions. Neither artifact upload nor this read-only preview
constitutes having run any held-out FGO experiment.


### R. 0020 READY_RAW; rejected GPS-negative-evidence pilot (archived)

The prospectively reserved KITTI `2011_09_30_drive_0020_sync` has
1104 matched OXTS/LiDAR/stereo frames and passed the raw-data installer
integrity check; no method-specific held-out localization result has
been read. Keep it untouched until a method and all hyperparameters
are frozen.

The GPS-only negative-evidence gate
`r'=2*r/(2+k*U*(1-q))` was evaluated on *development 0027* with
both frozen GRU V5 and Mamba V6 and **did not beat reliability-only**
(see measured results in section S). Its standalone implementation
and the dedicated old pilot runner/tests were deleted from the ACTIVE
branch to reduce dead experiment code. Full original sources remain
recoverable from the immutable Git commit
`87dd0d421e2feae0c8d802387bc2e8789fa3a9bc` and archival branch
`archive-neg-gps-gate-before-cleanup-20261118`.
The recorded development result files under
`results/isj_dev_gps_negative_evidence_v1/` on the user's server are
not deleted or overwritten. Original Stage6 FGO and the reusable raw
GPS/LiDAR discrepancy diagnostic are retained.

Do not restore the abandoned gate to claim positive ISJ results; its
no-U penalty strength was not matched to the U case, and initial-LM
ATE values were not uniformly convergence-controlled.

### S. E1: causal pre-update GPS 3D innovation diagnostic (next experiment)

The original GPS/LiDAR **5-step displacement magnitude** feedback
failed to improve development FGO with either frozen GRU V5 or
frozen Mamba V6. Verified Stage-6 initial-LM ATE3D on 0027:

| Backbone | r-only | No-U negative evidence | With-U negative evidence |
| --- | ---: | ---: | ---: |
| GRU | **2.184539 m** | 2.238903 m | 2.208585 m |
| Mamba | **2.350710 m** | 2.642782 m | 2.382172 m |

These are 0027 in-sample/development figures, without uniform
LM continuation. Do NOT keep modifying the old distance-only gate.
Never use raw-ready, not-yet-tested **2011_09_30_drive_0020_sync**
to select new-method parameters. The 0016 drive is already heavily
explored and also unsuitable as an untouched test.

New E1 module \`src/reliability/causal_gps_position_innovation.py\`
calculates **pre-update** 3D GPS innovations using previously updated
position and only LiDAR/Camera pair odometry \`t-1->t\`. First it aligns
the initial odometry heading against only the **past 63** degraded
GPS frames to bootstrap a common XY world frame, using NO clean
GPS, OXTS reference trajectory, future measurements or FGO optimized
pose. It records each innovation **before** the current GPS position
update; current GPS affects only *future* priors. The update uses fixed
5 m GPS observation scale, simple process-variance proxy and bounded
position correction, which are NOT a proper 6DoF EKF covariance or
probabilistically calibrated NIS. Persistent GPS bias can still be
absorbed by state updates, and historical GPS heading bootstrap may
be distorted by faults. LiDAR is first-choice relative odometry,
Camera is fallback. A separate LiDAR/Camera pair disagreement mask
helps identify cases in which a large GPS innovation could instead
be due to unreliable geometry.

New \`experiments/isj_supplement/audit_dev_causal_gps_innovation_e1.py\`
uses ONLY 0027. After computing the innovations, it reads
\`clean_gps_local\` from the existing degraded 0027 input **strictly
for diagnostics** to label injected GPS error >= **5m**. Original
OXTS GPS is NOT an independent reference or real-world fault label.
It compares same-frame AUROC, descriptive TPR and observed FPR
for (1) new 3D innovation norm, (2) new norm divided by a rough
scale proxy, and (3) old magnitude-difference NIS. It reports
geometric agreement/conflict subsets, valid support and simulated
GPS severity correlation. The negative-class 95th-percentile
threshold shown is **only** a descriptive developer-data operating
point; NOT a tuned deployable or externally validated threshold.
The evaluation supports making the next design decision but cannot
by itself justify an ISJ contribution.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest \
  tests/test_isj_causal_gps_position_innovation.py \
  tests/test_isj_cross_modal_consistency.py -v
echo "E1_TEST_EXIT=$?"
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_causal_gps_innovation_e1.py \
  > logs/isj_e1_dev_causal_gps_innovation.log 2>&1
echo "E1_AUDIT_EXIT=$?"
tail -n 100 logs/isj_e1_dev_causal_gps_innovation.log
~~~

No FGO, no checkpoint inference/training, no files modified by
E1 itself, no 0016/0020 access. Unit tests check specifically that
GPS[t] changes the pre-update **innovation** at t but cannot change
the **predicted position** at t (only later predictions may change).
If E1 does not beat the old norm-only score in fault detection,
or geometry conflict causes many false alarms, do not integrate
an E2 adaptive covariance algorithm; analyze/repair E1 on the 0027
development sequence. Parameter changes must be recorded and must
precede the prospective 0020 evaluation.


### T. Read-only E2A: GPS U + raw discrepancy complementarity decision gate

The read-only E1 diagnostic using 0027 showed that the new
**globally propagated** GPS innovation (mean 62.07 m) was far larger
than the average injected GPS perturbation (2.95 m), and its
synthetic-GPS-fault AUROC was **0.6006**, worse than the existing
5-step GPS/LiDAR *raw displacement magnitude mismatch* AUROC **0.8732**.
The global-pose prior is not validated; do not use it in FGO.
The old negative-feedback score q was capped at 1 for 96.76%
of valid frames, despite useful ranking in its raw discrepancy.
This is a candidate information-loss issue, **not** proof that
using the raw score will improve localization.

New \`audit_dev_gps_u_feedback_complementarity.py\` is a cheap,
read-only decision experiment. It computes **only on 0027**:
(1) \`R=1-r\` from original saved GPS GRU/Mamba reliability;
(2) \`D=log1p(gps_nis)\` from original 5-step LiDAR/GPS
measurement discrepancy, WITHOUT clipping to quality q;
(3) \`U\` saved evidential uncertainty. Note that gps_nis
is a rough heuristic score, **not** a statistically calibrated
normalised innovation squared.

Fault label is pre-existing synthetic GPS perturbation >=5m
against *original OXTS GPS*, NOT real GNSS fault or an
independent localization truth. All features use frozen sensor
measurements before reading this label. For each exact original
model-specific chronological train/val/test split, the script
reports (a) stand-alone fault-detection AUC for R,D,U and
(b) fixed low-capacity L2 logistic probes for R, R+D and R+D+U,
with each probe fitted ONLY on the TRAIN portion, including
normalization. It prints within-TEST r-risk quartile results.
The probe is a **diagnostic measuring predictive complementarity**,
not the proposed factor covariance mapping, and does not
establish causality or external generalization.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_gps_u_feedback_complementarity.py -v
echo "TEST_EXIT=$?"
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_gps_u_feedback_complementarity.py \
  > logs/isj_dev_gps_u_feedback_complementarity.log 2>&1
echo "E2A_EXIT=$?"
cat logs/isj_dev_gps_u_feedback_complementarity.log
~~~

**Decision rule**: if R+D+U does not improve over R+D on both
0027 model-chronological VAL and TEST (and across both backbones
or with a clear backbone-specific limitation), there is currently
no evidence that adding U helps *GPS fault detection conditional
on the other two signals*. Do not engineer an arbitrary U gate
solely to obtain a positive 0027 ATE. If it improves, the
next step is a physical/causal geometry and correct-covariance
diagnostic plus a predeclared U/no-U strength-matched FGO
ablation on 0027; this does not yet warrant reading 0020.
If neither model supports additive value, keep results as
negative findings and reconsider the novelty target. Both
the early 0027 segment and its later segments have been repeatedly
examined, so no result is an independent external test.


### U. E2A result and next geometry-only sanity audit (0027 development only)

Completed E2A \`audit_dev_gps_u_feedback_complementarity.py\` on the
user's 0027 server: pytest **4/4 PASS**, exit 0. All logistic
models were fitted on 0027's original model-specific TRAIN segment
and then evaluated on VAL/TEST without refitting.

| Backbone | Split | R+D AUC | R+D+U AUC | U addition |
|---|---|---:|---:|---:|
| GRU | VAL | .9365 | .9357 | -.0008 |
| GRU | TEST | .9357 | .9351 | -.0006 |
| Mamba | VAL | .9228 | .9236 | +.0008 |
| Mamba | TEST | .9464 | .9470 | +.0006 |

The added U signal was **not consistently helpful** at detecting
injected GPS faults beyond R+D on this in-drive experiment. In the
most risky 1-r TEST quartile, stand-alone conditional fault AUROC(U)
was **0.2160** GRU / **0.2127** Mamba. AUC(U) around .87-.88
from a DIFFERENT experiment measures *error in the learned r
prediction relative to a weak FGO-derived label*, not the physical
synthetic GPS fault event >=5m. Do not conflate these targets.
R-only GPS synthetic fault AUC itself was .9297 GRU/.9437 Mamba
on model-internal TEST; value remains in-sample/development and
does not imply good generalization.

**Stop rule triggered**: do not design an arbitrary new U
pseudo-count or threshold to force positive results, and do not
go to Stage FGO yet. First diagnose the old E1's mean 62m
GPS innovation. New read-only
\`audit_dev_odometry_geometry.py\` compares the existing physical
LiDAR and camera body-frame between measurements at k=1,5,20,63
frames against original OXTS/GPS-derived *clean* displacement
strictly as a post-hoc diagnostic reference, not a sensor-independent
ground truth or a method input. It prints mean/p95 length error
(rotation-invariant), mean/p95 vector error after the original
63-frame noisy-GPS yaw bootstrap, and median XY heading mismatch.
Large length error indicates odometry translation scale/magnitude
error; small length error with large vector error indicates rotation,
frame alignment, heading or accumulated angular drift. This
diagnostic **does not fix** the causal filter and does not prove a
new contribution.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_odometry_geometry.py -v
echo "GEOM_TEST_EXIT=$?"
# Run only if tests pass:
python -u experiments/isj_supplement/audit_dev_odometry_geometry.py \
  > logs/isj_dev_odometry_geometry.log 2>&1
echo "GEOM_AUDIT_EXIT=$?"
cat logs/isj_dev_odometry_geometry.log
~~~

Existing experimental outputs preserved, prospective 0020 not read.
Do not infer publication-ready method improvements from diagnostic
AUCs on already explored 0027.


### V. 0027 odometry geometry audit result and signed-heading follow-up

User reported \`audit_dev_odometry_geometry.py\` exited 0,
pytest 5/5 passed. Against original OXTS/GPS-derived (not independent)
reference, Stage-6 **physical LiDAR** gave:

| k | norm MAE (m) | vector MAE (m) | unsigned heading median |
|--:|--:|--:|--:|
| 1 | .044 | .464 | 28.8 deg |
| 5 | .152 | 2.302 | 28.6 deg |
| 20 | .543 | 9.151 | 28.5 deg |
| 63 | 1.578 | 28.039 | 28.2 deg |

The magnitude-only odometry is far better than its globally aligned
vector estimate, pointing to coordinate/yaw propagation or bootstrap
error, but **absolute median does NOT prove a constant 28-degree
offset**, because it discards the sign. The earlier Camera vector
comparison also **reused LiDAR's propagated world rotations**, so it
did **not** independently test Camera heading.

New \`audit_dev_signed_heading.py\`: source-specific separate
LiDAR and Camera historical (first 63 degraded GPS frames) XY yaw
bootstrap and source-specific SO(3) propagation; signed heading
circular mean and median, unsigned median, norm/vector errors
on k=1,5,20, stratified into four chronological blocks. It also
computes one *whole-drive original OXTS/GPS* best-fit static yaw
for diagnostic upper-bound vector error ONLY. This **uses future
evaluation references**, thus MUST NOT be deployed as an algorithm,
trained on, or described as a valid localization improvement.
Do not use the clean reference to infer real sensor bias with
independent accuracy. Never access 0020/0016 for this diagnosis.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_signed_heading.py -v
echo "SIGNED_HEADING_TEST_EXIT=$?"
# Only after tests pass:
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_signed_heading.py \
  > logs/isj_dev_signed_heading.log 2>&1
echo "SIGNED_HEADING_AUDIT_EXIT=$?"
cat logs/isj_dev_signed_heading.log
~~~

Decision: if signed angle remains near a consistent sign and
magnitude across time while oracle static yaw reduces errors,
investigate initial yaw/coordinate conversion using only available
historical (no future/clean-GT) information. If signed error
changes substantially by quarter, investigate SO(3) propagation
and odometry rotational drift. Do not immediately invent a new
U factor mapping or run FGO until geometry is understood.


### W. Confirmed 0027 signed heading: LiDAR accumulates yaw error; Camera stable

User completed \`audit_dev_signed_heading.py\`: pytest 4/4, exit 0.
Source-specific separate historical yaw and SO(3) propagation gave:
- **LiDAR** yaw signed circular mean by chronological Q1..Q4:
  **-6.02, -19.02, -39.96, -53.88 deg**;
  k=20 globally aligned vector MAE **8.566 m**.
- **Camera** signed circular mean:
  **+2.01, +0.49, +0.62, +2.27 deg**;
  k=20 vector MAE **0.961 m**.
- LiDAR norm errors remain much smaller than vector errors, and
  Camera's frame-specific yaw remains approximately stable.

This strongly indicates a cumulative LiDAR rotation/alignment
problem that corrupted the old E1 prior (62m mean GPS innovation,
synthetic-fault AUROC 0.6006). It does **not** establish which of
LiDAR ICP, body-frame calibration, time alignment or motion
convention caused drift, nor establish a real-world fault detector.
Any whole-drive OXTS 'oracle rotation' uses future reference and
cannot be used for method inputs.

New controlled **read-only E1 motion source ablation**:
\`src/reliability/causal_gps_position_innovation.py\` now accepts
\`CausalInnovationSettings(motion_policy=...)\`, preserving the
original default \`lidar_first\`. Alternative modes are
\`camera_first\` and
\`camera_rotation_lidar_translation\` (hybrid only when both
physically valid; else fallback). The code applies the source
policy also in the historical 63-frame bootstrap, **never** uses
current GPS in current prior, and preserves the original physical
LiDAR/Camera geometry disagreement score.

\`audit_dev_e1_motion_policy.py\` compares all three on identical
matched frames of 0027, same noisy GPS, same historical yaw protocol,
same input/mask, and common position process-noise proxy (both
sensor sources use 0.15 m per step for this *diagnostic*), holding
the simple causal GPS update fixed. Label is existing injected
GPS error >=5m against original OXTS/GPS, read only after prior
construction. Prints mean/p95 raw innovations, AUC and descriptive
development-only TPR at negative-class 95th-percentile, plus
chronological quartiles. The old 5-step GPS/LiDAR raw
displacement discrepancy is the strong comparison benchmark.
No learned model, FGO, independent truth or 0020/0016.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest \
  tests/test_isj_causal_gps_position_innovation.py \
  tests/test_isj_dev_e1_motion_policy.py -v
echo "E1_POLICY_TEST_EXIT=$?"
# Only if tests all pass:
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_e1_motion_policy.py \
  > logs/isj_dev_e1_motion_policy.log 2>&1
echo "E1_POLICY_EXIT=$?"
cat logs/isj_dev_e1_motion_policy.log
~~~

This is a controlled **root-cause diagnostic**, not a new paper
method. If Camera or hybrid reduces E1 innovation mean from 62m
and improves AUC, continue validating measurement/prior errors
and feedback false positives before designing FGO. If not,
investigate state-update/gain behavior and alternative sources
of absolute reference drift; do not arbitrarily tune thresholds.
Preserve all previous E1 and FGO outputs.


### X. E1 motion-source ablation outcome and SO3 reference-convention audit

User ran E1 source-policy read-only 0027 experiment successfully:
10/10 pytest pass; 4479 same matched frames, 19.469% injected
GPS error >=5m against original OXTS/GPS. Mean pre-update GPS
innovation / synthetic fault AUROC:

| motion prior | mean innovation | AUROC |
|---|---:|---:|
| LiDAR translation + LiDAR rotation | 62.0695 m | .6006 |
| Camera translation + Camera rotation | 8.2412 m | .7949 |
| LiDAR translation + Camera rotation | 7.9420 m | .7928 |
| original 5-step LiDAR/GPS norm-only measurement mismatch | n/a | **.8732** |

At development-normal p95 threshold (~5% observed FPR),
legacy norm-only TPR was **.5103** versus Camera **.0768**
and Hybrid **.0780**. Thus replacing LiDAR rotation fixes much
of the global-prior drift but the E1 fault detector is NOT
competitive with the old raw discrepancy and should **not** be
turned into a proposed covariance gate.

Even hybrid global GPS innovation rises in chronological Q1..Q4:
**1.489, 3.736, 7.322, 19.216 m**. Important causal drift and
GPS state-update issues remain. Existing fixed FGO uses tight
LiDAR rotation sigma **0.012857** versus camera sigma **0.04**
so whether that harms full FGO is plausible but UNPROVEN;
running FGO immediately is premature.

Next **read-only** diagnostic:
\`audit_dev_lidar_rotation_reference.py\`. Uses ORIGINAL 0027
KITTI OXTS [roll,pitch,yaw] only as post-hoc pose-increment
REFERENCE; these INS/GNSS-derived attitudes are not independent
high-precision truth and do NOT enter any method. For both
frozen Stage-6 physical LiDAR and Camera body-frame between
rotations, compare common-valid per-pair SO(3) geodesic residual
and signed yaw-increment residual. Report Q1..Q4 signed bias
and the error if the factor is inverted (to check
conversion convention), and LiDAR-vs-Camera relative rotation
disagreement. This helps distinguish:
1) systematic small yaw bias accumulating ~54deg in LiDAR,
2) rotation sign / inverse convention bug,
3) disagreements with reference due to IMU/body extrinsics
or OXTS timing. It is a diagnosis, not a demonstrated sensor
failure or ISJ method.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rotation_reference.py -v
echo "ROTATION_TEST_EXIT=$?"
# Only if all tests pass:
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_lidar_rotation_reference.py \
 > logs/isj_dev_lidar_rotation_reference.log 2>&1
echo "ROTATION_AUDIT_EXIT=$?"
cat logs/isj_dev_lidar_rotation_reference.log
~~~

Untouched 0020 stays for future prospective external test. 0016
is exploratory and may not be called a clean test. Do not use
post-hoc OXTS references to make model gating decisions, and
do not claim FGO gains without a fair convergence-controlled FGO.


### Y. 0027 OXTS raw-path resilience (2026-11-18)

After removing the rejected GPS-negative-evidence pilot from the
active branch (archived at
\`archive-neg-gps-gate-before-cleanup-20261118\`), the user
confirmed **23/23** current tests passed and there were no remaining
Python imports of the removed module. The existing source-specific
rotation audit originally exited 1 because the **default 0027
\`oxts/data/\` folder contained zero OXTS text files**.
This is a **missing reference path/data**, NOT a failed SO(3)
estimation experiment.

\`audit_dev_lidar_rotation_reference.py\` now accepts:
- \`--reference auto\` (DEFAULT): if raw original 0027 OXTS exists,
  compare both relative rotations to it; otherwise, **fall back to
  LiDAR-vs-Camera relative comparison ONLY**, emitting a prominent
  no-independent-truth warning. Does not construct fake OXTS data.
- \`--reference camera\`: run the pairwise LiDAR/Camera comparison
  only even if OXTS exists.
- \`--reference oxts\`: strict 0027 original OXTS required; fail
  if not available. Use \`--sequence-dir /absolute/path/
  2011_10_03_drive_0027_sync\` if the raw 0027 drive is stored
  somewhere other than the default repo dataset path. Identity
  must match 0027 and frame names must match 0..4543 exactly.

Commands:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rotation_reference.py -v
echo "ROT_TEST_EXIT=$?"
# Only if tests pass:
mkdir -p logs
python -u experiments/isj_supplement/audit_dev_lidar_rotation_reference.py \
  --reference auto > logs/isj_dev_lidar_rotation_reference.log 2>&1
echo "ROT_AUDIT_EXIT=$?"
cat logs/isj_dev_lidar_rotation_reference.log

# OPTIONAL read-only search for misplaced original 0027 raw drive:
find /root/projects/DFG /data /mnt /home/vipuser \
  -type d -name '2011_10_03_drive_0027_sync' -print 2>/dev/null
# If the actual OXTS raw folder exists elsewhere, explicitly pass its
# parent 0027 sequence directory via --sequence-dir with --reference oxts.
~~~

CAUTION: when raw OXTS is unavailable, pairwise disagreement alone
does not identify which sensor is wrong. Never call Camera ground
truth. Never delete untracked \`external/\`, \`logs/\` or \`results/\`
as a blanket cleanup operation, since these can contain unique raw
data and experimental evidence. Preserve the prospective 0020 drive.


### Z. 0027 physical-camera source check and first LiDAR-rotation sigma FGO sensitivity

The user ran the repaired SO(3) diagnostic on 0027 successfully:
pytest 6/6, audit exit 0, **4396 matched LiDAR/Camera pairs**.
No original 0027 OXTS attitude raw data were on the default server
path. Mean LiDAR-versus-Camera SO(3) angle **0.15361deg**, p95
**0.30137deg**; mean signed LiDAR-minus-Camera relative yaw step
**+0.009426deg**, arithmetic sum **+41.437deg**. Per-quarter
signed sums **+14.52,+8.78,+22.67,-4.53deg**. Inverting only
the LiDAR rotation worsens the LiDAR-vs-Camera mean discrepancy
from **0.15361deg** to **1.53031deg**; this does **not** prove
the absolute direction convention of either factor against an
independent attitude reference. The 41.437deg is arithmetic
signed step accumulation, not a validated global SO3 residual.
Without OXTS, neither sensor can be declared independently
"correct" from the pairwise comparison.

Source review:
\`experiments/degradation/build_degraded_four_sensor_measurements.py\`
calls \`StereoVisualOdometry.estimate(a,b,c)\` on degraded
stereo images to generate
\`degraded_camera_factor_data.npz\`; the routine uses ORB/stereo
3D-PnP-RANSAC, not GPS position as an input. The Stage-6 loader
reads this camera file, overrides only LiDAR factors with a
separate physical LiDAR ICP benchmark. This confirms the
**intended** code-generation path, but the exact provenance of
the server's saved NPZ artifacts should still be independently
reconciled with run manifests and dates when available.

Next predeclared diagnostic is
\`run_dev_lidar_rot_sigma_ablation.py\`: exact original Stage-6
fixed four-sensor FGO, frozen sensor measurements, initialization,
graph topology, solver and **all non-LiDAR-rotation sigmas**.
Only vary fixed \`LIDAR_R0\` on 0027 across:
\`0.012857142857\`, \`0.04\`, \`0.10\`, \`0.40\` rad. These
are **static sensitivity controls**, not proposed method or
hyperparameter selection from a prospective test drive. Results
go under a new output root
\`results/isj_dev_lidar_rot_sigma_v1\`; any partial/mismatched
existing folder causes a hard error. Original Stage-6 output is
preserved. The initial 100-iteration GTSAM LM may not converge
uniformly; compare pilot results cautiously before
a convergence-controlled rerun. No 0016 or 0020 data access.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rot_sigma_ablation.py -v
echo "ROT_SIGMA_TEST_EXIT=$?"
# First read-only preflight:
python -u experiments/isj_supplement/run_dev_lidar_rot_sigma_ablation.py \
  --stage audit
# Only after tests/preflight and explicit readiness for CPU FGO:
python -u experiments/isj_supplement/run_dev_lidar_rot_sigma_ablation.py \
  --stage fgo > logs/isj_dev_lidar_rot_sigma_fgo.log 2>&1
echo "ROT_SIGMA_FGO_EXIT=$?"
tail -n 100 logs/isj_dev_lidar_rot_sigma_fgo.log
~~~

Stop criteria:
- If original rotation confidence is not harming 0027 fixed-FGO
  under equal convergence, rotation decomposition is not yet
  empirically justified.
- If reduced LiDAR rotation weight improves metrics, that is
  preliminary *evidence of overconfident rotation constraints*,
  NOT proof that learning dynamic rotation uncertainty will help.
- Full new method requires sensor-available predictive rotation
  reliability, control of correlation with Camera, strength-matched
  static baselines, multi-sequence independent checks and honest
  OXTS-based reference limitations.


### AA. Completed fixed-LiDAR rotation sigma FGO pilot on 0027; uniform convergence next

User ran \`run_dev_lidar_rot_sigma_ablation.py\` on the **0027
development drive**, with five unit tests passing, read-only
preflight successful, FGO exit code zero. All 4 cases had
**18092 factors**, same frozen GPS/IMU/Camera and LiDAR translation
sigma, identical Stage-6 LiDAR initialization and initial 100-LM
stopping configuration. Original fixed FGO ATE3D was reproduced
at **2.456340220977 m**.

| Fixed LiDAR rotation σ (rad) | ATE3D (m) | ATE2D (m) | ZRMSE (m) |
|---:|---:|---:|---:|
| .012857142857 (original) | 2.456340 | 2.374576 | **.628487** |
| .040000 (Camera scale) | 2.490096 | **2.284889** | .989879 |
| .100000 | 2.480971 | 2.392186 | .657772 |
| .400000 | **2.429492** | 2.318908 | .724634 |

The best development 3D change is **-0.026849m / ~1.09%**
at 0.4rad, but ATE3D is non-monotonic with sigma; the
0.04rad case improves planar ATE while worsening z and
full 3D ATE. This is NOT evidence that LiDAR should always be
downweighted, nor a new algorithm. These numbers were produced
after only one 100-iteration LM solve with UNKNOWN comparative
stationarity; they are exploratory and may change on
uniform continuation.

New \`run_dev_lidar_rot_sigma_convergence.py\` **reuses the exact
Stage-6 factor graph construction** by invoking existing
\`s6.run_case\` with ONLY three scoped in-memory patches:
1. Stage-6 output path isolated to
   \`results/isj_dev_lidar_rot_sigma_convergence_v1\`;
2. corresponding fixed \`s6.LIDAR_R0\` matching its original
   pilot saved pose case;
3. LM solver factory to start from that case's **saved
   100-iteration solution**, not restart from the original
   pose initializer, and continue under the same predeclared
   stopping test for each case.

**Stop** when both relative weighted objective decrease
<=1e-7 AND interpass position shift RMSE <=1mm; maximum
4 extra passes of up to 100 LM iterations each. Stopping
criteria **do not use ground truth or ATE**. The script computes
ATE3D/ATE2D/ZRMSE and 1/10/20-frame translational RPE against
the OXTS/GPS-derived local reference **for post-hoc evaluation**,
with strict saved source manifest/shape verification and
new-output cache protection. Different covariance graphs have
different objective scales: compare stationarity separately,
NEVER compare the absolute graph objectives as a method score.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest \
  tests/test_isj_dev_lidar_rot_sigma_convergence.py \
  tests/test_isj_dev_lidar_rot_sigma_ablation.py -v
echo "ROT_CONVERGENCE_TEST_EXIT=$?"
# Read-only check of all 4 existing pilot manifests and saved poses:
python -u experiments/isj_supplement/run_dev_lidar_rot_sigma_convergence.py \
  --stage audit
echo "ROT_CONVERGENCE_AUDIT_EXIT=$?"
# Only after both previous commands succeed:
mkdir -p logs
python -u experiments/isj_supplement/run_dev_lidar_rot_sigma_convergence.py \
  --stage converge > logs/isj_dev_lidar_rot_sigma_convergence.log 2>&1
echo "ROT_CONVERGENCE_EXIT=$?"
tail -n 100 logs/isj_dev_lidar_rot_sigma_convergence.log
~~~

This is a *post-hoc numerical control* only. If all four
stationary, compare the resulting 3D, 2D, Z and RPE metrics
without changing their declared sigma values. If any case
is NOT_STABLE, don't make a ranking. No 0016/0020 access,
no checkpoint edits, no changes to the previous pilot outputs.


### AB. Observed uniform convergence and next read-only local trade-off audit

The user ran \`run_dev_lidar_rot_sigma_convergence.py\` on 0027:
10/10 related pytest passed; source-manifest audit and convergence exit
were both 0. All FOUR noise variants reported
\`stationary_under_rule=True\` under equal frozen stopping rules;
the 0.4rad configuration needed 3 additional LM passes, others 2.

| LiDAR static σrot(rad) | ATE3D (m) | ATE2D (m) | Z-RMSE (m) | translation RPE k1 (m) | RPE k10 (m) |
|---:|---:|---:|---:|---:|---:|
| .01285714 original | 2.459228 | 2.377495 | **.628743** | **.073686** | **.526835** |
| .040000 | 2.582553 | 2.366627 | 1.033758 | .096848 | .801273 |
| .100000 | 2.459968 | 2.375584 | .638781 | .082236 | .627618 |
| .400000 | **2.420614** | **2.319243** | .693170 | .084157 | .649254 |

Relative to original σrot:
- σrot=.4rad: development **ATE3D -1.57%**, **ATE2D -2.45%**,
  but **Z-RMSE +10.25%**, **RPE k1 +14.21%**, **RPE k10 +23.24%**.
- There is a clear full-trajectory *ATE-vs-RPE* trade-off. NO
  reliable overall superiority, no monotonic benefit of
  de-weighting rotation, and no current dynamic-gate evidence.
- Source 0027 is development/in-sample; original OXTS/GPS
  evaluation reference shares data provenance with the GPS input.
- Numerical stability under the same local LM stopping criterion
  does NOT establish global optimum or generalization.

Next **read-only** experiment:
\`audit_dev_lidar_rot_tradeoff.py\` uses ONLY already saved uniformly
converged 0027 pose results (rejects missing/nonstationary cases)
and Stage-6 physical LiDAR/Camera factors. It reports identical
matched-frame/trajectory evaluation metrics for:
1. whole development sequence (63..4541);
2. chronological Q1..Q4;
3. high vs low LiDAR/Camera pair SO(3) disagreement defined
   entirely from sensor observations as the **0027 q75
   discrepancy threshold** (no GT-derived gating); for these
   groups pairwise RPE is strictly k=1, so grouping is aligned
   with the immediate disagreement observation.
All subgroup comparisons use the same evaluation masks for every
static rotation weight. This is exploratory, not a trained online
dynamic rotation-confidence model and not independent validation.
No optimization runs, no saved files changed, no 0016/0020 accessed.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rot_tradeoff.py -v
echo "SUBGROUP_TEST_EXIT=$?"
# ONLY after pass: read-only diagnostic
python -u experiments/isj_supplement/audit_dev_lidar_rot_tradeoff.py \
  > logs/isj_dev_lidar_rot_tradeoff.log 2>&1
echo "SUBGROUP_AUDIT_EXIT=$?"
cat logs/isj_dev_lidar_rot_tradeoff.log
~~~

Stop criterion: if increased static sigma improves only long-term ATE
but worsens per-frame motion even in high angular-disagreement
regions, any future dynamic splitting method must demonstrate
both independent ATE and RPE benefit relative to the strongest
fixed control, not only beat the original 0.012857rad baseline.
Do not tune a gate from 0027 subgroup performance and call it
a prospective test on 0020.


### AC. 0027 static rotation noise quarter results: Q3-only benefit; existing discrepancy threshold not predictive

User ran \`audit_dev_lidar_rot_tradeoff.py\` on 0027 successfully,
4/4 tests passing, read-only run exit 0. All four static sigma
FGO pose sets had already met the same solver stopping criterion.

At 0.4rad vs unchanged 0.012857rad LiDAR rotational sigma,
quarterwise *ATE3D* and 10-frame translation RPE:

| quarter | ATE3D baseline -> .4rad (m) | RPE10 baseline -> .4rad (m) | interpretation |
|---|---|---|---|
| Q1 (63..1181) | 1.14444 -> 1.24590 | .21389 -> .36204 | WORSE |
| Q2 (1182..2301) | 3.10710 -> 3.18709 | .42757 -> .74805 | WORSE |
| Q3 (2302..3421) | 2.56517 -> 2.20360 | .65395 -> .54554 | BETTER |
| Q4 (3422..4541) | 2.44069 -> 2.58218 | .59294 -> .78363 | WORSE |

The largest localization benefit is **Q3 only**, with estimated
ATE3D -14.09%, RPE10 -16.58%; ATE and RPE worsen in Q1/Q2/Q4.
CAUTION: global factors are optimized together, so the
per-quarter metrics of FULL-TRAJECTORY static weighting are
NOT an oracle time-varying covariance experiment.

Sensor-only global 75th-percentile LiDAR-vs-Camera incremental
SO3 disagreement threshold was **0.182038deg**. On same-frame
subgroups:
- LOW: original ATE3D 2.28839 vs .4rad 2.27188m,
  RPE1 0.06830 vs 0.07880m.
- HIGH: original ATE3D 2.64115 vs .4rad 2.62458m,
  RPE1 0.07428 vs 0.08576m.

The ATE3D benefit is virtually IDENTICAL in low/high groups
(~.0165m); RPE worsens in BOTH. Therefore the
simple global 75th-percentile intermodal angular discrepancy is
**NOT justified as a dynamic LiDAR rotation gate**.
Neither Camera nor LiDAR can be deemed ground truth from this
comparison.

Next step uses **the existing audit script** rather than growing
a separate experiment package: print sensor-only per-quarter
statistics from frozen 0027 physical LiDAR and Camera factors:
angle mean/median/p95, fraction exceeding the descriptive global
0.182038deg angle threshold, signed LiDAR-vs-Camera yaw
difference, mean quality scores, average relative translation
magnitudes and common pair coverage. No post-hoc OXTS/GT or
optimized pose can enter these sensor-only summary features.
The purpose is simply to assess whether Q3 shows any distinctive
*available sensor* footprint that merits further causal testing.
The sensor-only quarter profiles were added to
\`audit_dev_lidar_rot_tradeoff.py\` and unit tests in-place.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rot_tradeoff.py -v
echo "PROFILE_TEST_EXIT=$?"
# Only when tests pass; read-only, no FGO:
python -u experiments/isj_supplement/audit_dev_lidar_rot_tradeoff.py \
 > logs/isj_dev_lidar_rot_tradeoff.log 2>&1
echo "PROFILE_AUDIT_EXIT=$?"
grep -A 6 "SENSOR-ONLY CHRONOLOGICAL QUARTER PROFILES" \
 logs/isj_dev_lidar_rot_tradeoff.log
~~~

**Decision**: If these sensor-only profiles do not clearly
distinguish the Q3-localized improvement from other times,
STOP designing disagreement-based LiDAR rotational gates.
Revisit a distinct research hypothesis and independent
validation instead of repeatedly mining 0027. Any apparently
distinct sensor indicator is only a hypothesis from
already-explored 0027; no performance/novelty claim until
proper pre-registered evaluation and stronger baselines.


### AD. Sensor-only Q1-Q4 profile outcome and final triangulation check

User ran the existing 0027 quarter-profile audit (5/5 pytest,
exit 0). The previous Q3-only 0.4-rad static rotation-weight FGO
ATE/RPE benefit did **not** align with the LIDAR/Camera SO(3)
disagreement magnitude or high-discrepancy fraction:
Q2 max angle mean **0.17994°** and high-angle fraction **34.26%**,
but Q2 localization deteriorated under 0.4 rad. Q3 had
angle mean **0.14721°** and high-angle fraction **24.12%**
yet was the only quarter to improve ATE and 10-frame RPE.
LiDAR-quality mean remained ~**0.94** in all quarters.
Q3 had the highest *signed incremental* LiDAR-minus-Camera yaw
mean **+0.02055°/pair**, versus Q1 +0.01298°, Q2 +0.00810°,
Q4 -0.00413°. The Q3 signed mean is an exploratory clue,
NOT a validated sensor-error measure: it does not tell us which
sensor is wrong or whether online sigma weighting helps.

As the LAST diagnostic before stopping 0027 threshold mining,
the **same** \`audit_dev_lidar_rot_tradeoff.py\` now prints
\`DEGRADED-IMU THIRD-SENSOR ROTATION CROSSCHECK\`. It
reconstructs GTSAM-compatible Exp(gyro[t]*actual_dt) SO3
relative rotations from the SAVED, DEGRADED 0027 IMU and
timestamp arrays (no raw OXTS attitude or reference poses).
Reports Q1..Q4 matched LiDAR-IMU and Camera-IMU geodesic
and signed yaw residual statistics. This can indicate which
sensor agrees with the *third noisy modality*, but IMU itself
has synthetic faults and OXTS provenance; do not treat it as
independent truth or tune from 0020.

~~~bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_dev_lidar_rot_tradeoff.py -v
echo "TRIPLE_TEST_EXIT=$?"
# Only after pass, read-only and no FGO:
python -u experiments/isj_supplement/audit_dev_lidar_rot_tradeoff.py \
  > logs/isj_dev_lidar_rot_tradeoff.log 2>&1
echo "TRIPLE_AUDIT_EXIT=$?"
grep -A 6 "DEGRADED-IMU THIRD-SENSOR ROTATION CROSSCHECK" \
  logs/isj_dev_lidar_rot_tradeoff.log
~~~

**Decision limit**: If Q3 residual pattern is absent/ambiguous,
STOP the LiDAR-rotation gate hypothesis and return to a broader
uncertainty-aware fusion research redesign. If Q3 is distinctive
with respect to the IMU as well, it is still a hypothesis, not
evidence of generalization or method superiority. Specify
a causal sensor-only rotation-confidence score BEFORE testing,
then compare against strong static/sensor-quality baselines
and check both ATE AND RPE. Keep blind 0020 untouched until
method/parameters are frozen; avoid more Q3 threshold tuning.

### AE. Third-sensor rotation diagnostics: STOP the LiDAR/Camera disagreement gate

User ran the last previously predeclared 0027 READ-ONLY gyro
cross-check successfully: all **7 tests passed** and audit
exit=0. Existing Stage-6 DEGRADED gyro[t]*actual_dt served as
the third noisy relative SO(3) source, **not ground truth**.
On matching triply valid physical factors:

| Quarter | LiDAR–IMU SO3 mean (deg) | Camera–IMU SO3 mean (deg) | LiDAR–IMU signed yaw mean (deg) | Camera–IMU signed yaw mean (deg) |
|---|---:|---:|---:|---:|
| Q1 | 0.27402 | 0.27391 | +0.01206 | -0.00092 |
| Q2 | 0.35184 | 0.37095 | +0.00264 | -0.00546 |
| Q3 | **0.49301** | **0.48814** | **+0.02440** | +0.00386 |
| Q4 | 0.36088 | 0.37352 | -0.01042 | -0.00629 |

Q3 LiDAR–IMU SO3 p95=2.24236deg, Camera–IMU
p95=2.15675deg. Both show increased magnitude
relative to previous quarters; signed yaw difference
between LiDAR–IMU and Camera–IMU in Q3 is
+0.02054deg/pair, in agreement with the existing
LiDAR–Camera signed relative yaw observation. This
**does not** establish which sensor is wrong. IMU has
synthetic degradation; all three modalities can
share motion-condition dependencies. There is no
independent OXTS attitude reference in the current
0027 raw folder.

**Decision as of 2026-11-18: STOP exploratory 0027
intermodal rotation-disagreement threshold mining.**
The q75 magnitude discrepancy did not predict useful
local FGO weight adaptation; Q3-only static-noise
improvements are not evidence of a working temporal
dynamic gate. Uniformly continued fixed FGO showed
ATE3D 2.459228m -> 2.420614m at sigma_r .4rad
(-1.57%) but RPE k1 0.073686m -> 0.084157m and
RPE k10 0.526835m -> 0.649254m, both worse.
This cannot be sold as a broadly superior
method, especially on a repeatedly explored
development sequence.

Do NOT invent further ad hoc q/IMU yaw thresholds or
use prospective untouched 0020 for tuning.
Treat these as valuable **negative controls and
diagnostic evidence**, not the new paper algorithm.
Next research step should be a **method-first
design** with explicit falsifiable reliability
hypotheses, GT-free input features, frozen
evaluation protocol (ATE3D/ATE2D/ZRMSE/RPE,
diverse degradation and stronger static/robust
baselines), and test after method and parameters
freeze. The existing GPS reliability, Mamba
and factor graph work remains available; this
decision does not establish Mamba/U gains.
Retain raw data, outputs, model checkpoints and
the cleaned repository's reproducibility.
