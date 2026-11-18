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
