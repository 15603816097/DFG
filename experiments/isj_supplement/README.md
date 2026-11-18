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
