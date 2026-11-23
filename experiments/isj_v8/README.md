# V8.1 — Development-only forecast-skill audit

V8 is a NEW RESEARCH STAGE. Frozen V7 code, model weights, 0016/0020
measurement archives, and final scores stay immutable and in the
original DFG branch `isj-supplementary-experiments-20261115`.

V8.1 answers one prerequisite question **before** implementing predictive
factor-graph weighting: are the already-trained h1/h2 heads more
accurate than the simple deployment-feasible assumption that the
predicted *current* h0 reliability remains unchanged one or two
frames in the future?

## Frozen read-only input

- Exactly the frozen 0027 GPS train/validation candidate (not 0027
  internal TEST), 680 validation anchors `t=3180,...,3859`.
- Exactly the existing seed43 saved per-model `val_predictions.npz`
  for GRU96, GRU106 (matched capacity), and Mamba.
- Confirm original model and validation prediction provenance.
- The checkpoint SHA256 is validated, but no checkpoint is loaded
  or network inference performed. No new training takes place.
- Only synthetic oracle reliability targets are used for
  retrospective scoring; no clean GPS positions or held-out
  0016/0020 measurement files are loaded.

## Metrics and reference

- For each head h1 and h2, evaluate MAE and RMSE against the
  saved same-horizon validation target.
- Compare against the causal **predicted-h0-persistence** baseline:
  use h0(t) to predict the target at t+1 / t+2. No future input.
- A positive `rmse_change_vs_persistence_magnitude` means h1/h2
  had smaller RMSE; a negative value means persistence was better.
- Recompute the same metrics on retrospectively identified
  reliability changes where `abs(y(t+h)-y(t))>=0.15`, a locked
  descriptive threshold. These oracle-selected events are NOT
  detectable online and NOT sufficient to claim a deployed gain.
- Validate target alignment exactly: `y_h1(t)==y_h0(t+1)`
  and `y_h2(t)==y_h0(t+2)` wherever both are in VAL.
- The Beta evidence U is reported only as a concentration proxy.

## Run on server

```bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
git status --short
git fetch origin
git switch -c isj-v8-forecast-audit-20261121 --track origin/isj-v8-forecast-audit-20261121
python -m pytest tests/test_isj_v7_*.py tests/test_isj_v8_*.py -q
python -m experiments.isj_v8.audit_forecast_dev --preflight
```

The entrypoint has no output path / write-report flag and does NOT
create results or train/optimize anything. It prints JSON and
`V8_DEV_HORIZON_PREFLIGHT_OK` on full success.

## Decision gate before V8.2

If h1/h2 give no reproducible incremental forecast skill over
predicted-h0 persistence on this DEV interval, do not pretend they
already improve the graph. Diagnose the feature/target/architecture
limitations. If they appear promising, next implement a causal
**offline replay** prototype with pre-registered temporal selection:
at time t, cache h1(t)/h2(t) for t+1/t+2; at frame t+1
an incoming observation can only use predictions made at or before
t+1, and any innovation-feedback must be measured from a
pre-update state (never from its own optimized posterior).

Development-only predictive skill does NOT imply cross-sequence
generalization. V7 0016/0020 cannot be reused to select V8
architecture, parameters, thresholds or stopping rules. A future
confirmatory protocol must specify NEW unseen data and, ideally,
an independent trajectory reference before evaluating it.

## Status

Code and unit tests committed; real Ubuntu server tests/DEV report
are required before any conclusion about h1/h2 forecasting.
## V8.1 real Ubuntu dev-forecast result (2026-11-21)

User-provided server transcript confirmed `138 passed`,
`V8_TEST_EXIT=0`, `V8_FORECAST_EXIT=0`, and
`V8_DEV_HORIZON_PREFLIGHT_OK`. All three seed43 frozen
checkpoints matched their expected SHA256 and only 0027
validation archives were read. No FGO or 0016/0020 access.

V8.1 horizon RMSE on the 680 development anchors:

| Backend | h1 forecast | h1 h0(t) persistence | h2 forecast | h2 h0(t) persistence |
|---|---:|---:|---:|---:|
| GRU96 | 0.22003659 | 0.22549680 | 0.22750424 | 0.23466183 |
| GRU106 | 0.22205254 | 0.22176731 | 0.22798983 | 0.23286991 |
| Mamba | **0.21795851** | 0.22529657 | **0.22693520** | 0.23823028 |

Mamba's future-head improvement versus its predicted current-head
persistence is 0.007338 (h1) and 0.011295 (h2) RMSE units.
These are *small DEV-only* gains with no measured FGO effect.
On oracle-defined large-change subsets, Mamba's change-direction
accuracy is 51.08% (h1) and 58.16% (h2), not strong evidence
of online change-point prediction. H1/H2 predict synthetic
reliability, not independently verified physical GPS covariance.

## V8.2 read-only pre-observation causal-prior audit

Before adding forecasting to a factor graph, evaluate how
predictions MADE BEFORE the current GPS(t) observation perform:

- `h1(t-1)`: one-step-ahead reliability prior for GPS(t).
- `h2(t-2)`: two-step-ahead reliability prior for GPS(t).
- Fixed, non-tuned 50/50 average of the two previous forecasts.
- Fair past-only baselines `h0(t-1)` / `h0(t-2)`.
- `h0(t)` is shown ONLY as a post-observation diagnostic and
  is never permitted into a pre-observation prior.

The script uses existing frozen prediction archives and targets
on 0027 val, t=3182,...,3859 (678 anchors). Starting at t=3182
is necessary because previous saved val predictions for t-1/t-2
are unavailable at the first two anchors. This is not retraining,
an optimization run, or a held-out test. Any retrospective
oracle-defined change subset is descriptive only.

Entry point:

```bash
python -m pytest tests/test_isj_v8_*.py -q
python -u -m experiments.isj_v8.causal_prior_dev --preflight
```

**Caution:** the original server worktree may still be on
`isj-supplementary-experiments-20261115` with unrelated modified
and untracked files. Do NOT use `git reset --hard`, `git clean`
or `git switch --force`. V8.2 files can be safely fetched and
extracted from the explicit V8 branch Git blobs without changing
the current branch, provided any existing destination content
is identical (otherwise stop). Do not overwrite local 0027/0016
or 0020 results or any server-generated artifacts.

V8.2 success will **not** establish model performance in FGO.
If the previous-step forecast is not better than past h0
persistence, do not deploy an unsubstantiated predictive gate.
## V8.2 actual Ubuntu 0027 validation results (2026-11-21)

Server transcript: `147 passed` and `V8_2_PREFLIGHT_EXIT=0`.
All scores are retrospective, existing frozen seed43 validation predictions,
only 0027 frames 3182..3859 (678 anchors); no FGO or withheld data.

| Model | h0(t-1) persistence | h1(t-1) | h0(t-2) persistence | h2(t-2) | 0.5 h1(t-1) + 0.5 h2(t-2) |
|---|---:|---:|---:|---:|---:|
| GRU96 | 0.225792 | 0.220314 | 0.234972 | 0.227793 | 0.223276 |
| GRU106 | 0.222063 | 0.222343 | 0.233183 | 0.228289 | 0.224601 |
| Mamba | 0.225615 | 0.218263 | 0.238568 | 0.227252 | **0.215450** |

The Mamba fixed blend beats its own one-frame past-h0-persistence
by 0.010165 (4.51% RMSE); direct h1 wins 0.007352 and direct
h2 wins 0.011316 against the *respective* one/two-frame baselines.
These synthetic-label RMSE improvements are small and cannot
be reported as state-estimation gains.

## V8.3 frozen development-only FGO design (pre-registered in code)

One **exploratory** 0027 validation replay is now authorized.
Every method uses identical development four-sensor factors,
the same 680-frame interval [3180,3860), the same LiDAR
initialization, the V7 train-only 5-bin XYZ noise mapping,
the V7 Cap4 precision bound and the same GTSAM LM stop rules.

Methods:

- Shared `axis_fixed` (no network).
- For each of the three seed43 frozen models, five alternatives:
  `h0_current_observation_available` — uses h0(t) after GPS(t) arrives;
  `h0_tminus1_persistence` — past only;
  `h1_tminus1_causal_prior` — past only;
  `h2_tminus2_causal_prior` — past only;
  `h1h2_equal_causal_prior` — past only, fixed [0.5,0.5] weights.
- To compare whole-graph results with identical factor counts,
  every case uses train-fixed XYZ noise on the initial two
  frames of this 680-frame validation graph; after that
  the chosen predictions are mapped to per-axis GPS sigma.
  This startup policy is fixed BEFORE viewing any V8.3 FGO results.

**Primary evaluation**: origin-relative OXTS/GPS-correlated
3D position RMSE of the same 680 frames. Secondary diagnostics:
2D RMSE, 1-frame and 10-frame translation displacement error.
Primary scientific comparison is the difference between
`h1_tminus1_causal_prior` and `h0_tminus1_persistence`
for each model; the h2 comparison uses `h0(t-2)`.
Predeclared additional comparisons: h1h2 50/50 vs h0(t-1)
and h0_current (diagnostic current-observation reference only).
All 16 graphs are solved under the same 12-pass LM budget
and required stabilization before any reference position error is
computed. Only one seed and one initialization; no significance claim.

**Important distinction**: time-aligned forecast weights do NOT
make this an online estimator. The actual graph is still OFFLINE
full-batch optimization using future pose factors. No feedback
residual weighting, evidence calibration or iSAM2 is added.

V8.3 writes exactly one *new*, non-overwriting development folder:
`results/isj_v8/dev_0027_feedforward_fgo_v1/`. The old
`results/isj_v7` and 0016/0020 archives remain untouched.

Preflight (metadata / covariance preparation only; no solver, writes, test):

```bash
python -m pytest tests/test_isj_v7_*.py tests/test_isj_v8_*.py -q
python -m experiments.isj_v8.run_dev_feedforward_fgo --preflight
```

Explicit DEV solver (only AFTER all tests+preflight pass):

```bash
python -u -m experiments.isj_v8.run_dev_feedforward_fgo \
  --run-graph --token V8_DEV_ONLY_CAUSAL_FGO_0027
```

Do not alter the candidate methods, equal-blend weight, noise map
or optimizer based on existing 0016/0020 scores. This is an
exploratory test on the previously inspected 0027 development
interval, NOT new independent confirmatory evidence for ISJ.

**Future V8.4:** introduce feedback using a pre-update innovation
and validate its conditional independence. Do not add posterior
same-observation residual reweighting inside the current offline
graph without a statistically justified robust estimation model.
## V8.3.1 — numerical local-solution diagnosis (post-hoc DEV)

V8.3 result JSON has been audited with **164 passed** and
`V8_3_REPORT_READONLY_AUDIT_PASSED`, SHA256:

`4a716af6bbdd1f7edfcb9e3282191a781a06d013d8b574e9956c97a8446979ca`

This new stage is explicitly **post-hoc** (cases selected after inspecting
V8.3), and can diagnose but **not** independently validate an algorithm.

Representative 0027 DEV objectives with strongly different reference errors:

- `mamba_r__h0_current_observation_available`: 2.073255 m.
- `mamba_r__h1h2_equal_causal_prior`: 6.644273 m.

The new `diagnose_dev_local_solutions.py` runs exactly **four** GTSAM
optimizations across these two fixed target objectives: first each from
the identical LiDAR initialization, then each again warm-started with
the **other target's** converged pose sequence. For a given target,
GPS covariance is **unchanged** across both starts; it only tests
initialization dependence. We may compare graph objectives **within**
each target; objective values between different covariance targets are
not comparable as method-quality scores.

Preflight audits the original JSON's SHA256, exact training-only XYZ
covariance map for both cases, and frozen 0027 measurement hashes. The
original frozen JSON is never rewritten or used to select solver
hyperparameters; no 0016/0020 data is accessed.

The diagnostic reproduces original 3D RMSE within 0.05 m or fails
closed. Clean OXTS-correlated GPS reference is opened only after
all four solves, and never influences weights, initial values or LM.

```bash
python -m pytest tests/test_isj_v7_*.py tests/test_isj_v8_*.py -q
python -m experiments.isj_v8.diagnose_dev_local_solutions --preflight
# Only after all tests and preflight pass:
python -u -m experiments.isj_v8.diagnose_dev_local_solutions \
  --run-probe --token V8_3_1_0027_DIAGNOSE_LOCAL_SOLUTIONS
```

Output (new folder only):
`results/isj_v8/dev_0027_local_solution_probe_v1/cross_initialization_diagnostics.json`

**Interpretation:** if the same graph objective converges to
statistically distinct stationary objective values with different starts,
initialization sensitivity is directly demonstrated for that objective.
If both starts converge to similar objectives, this **does not rule
out** other local minima or covariance sensitivity, and does not
validate the future reliability mechanism. A lower OXTS-relative RMSE
under a *different graph objective* is not itself local-minimum
evidence. The original V8.3 result remains a mixed/negative result
for predictive fusion; V8.3.1 is a numerical diagnosis only.

## V8.3.1 actual server result (2026-11-23)

On 0027 DEV, `173 passed`, four solves completed, and
`V8_3_1_RUN_EXIT=0`. SHA256 of the new probe archive:

`c684238ad7b97c23add7e116129ab4b2d9be8ad79fe7017d2c2e3140dca3a854`

For exactly the same GPS covariance graph:

| Target case | Start | Objective | 3D RMSE [m] |
|---|---|---:|---:|
| Mamba h0(t) | LiDAR default | 2044.7042 | 2.0733 |
| Mamba h0(t) | Mamba 50/50 result | 11090.5444 | 6.6443 |
| Mamba h1/h2 50/50 | LiDAR default | 10967.3191 | 6.6443 |
| Mamba h1/h2 50/50 | Mamba h0(t) result | 2236.6191 | 2.1552 |

The first target's cross-start LM solution did NOT decrease objective
at all: 11090.54442322 both before and after one reported pass, while
the wrapper marked `stable=True`. The second target can achieve a
same-graph objective about 79.6% lower from a different start.

**Correct interpretation:** this establishes large
initialization/solver-outcome dependence for the same fixed objectives.
It does **not** prove both returned estimates are stationary local
minima: the current outer loop calls a graph `stable` whenever
relative objective reduction and pose displacement are small, including
when there is literally NO progress. Therefore the word "convergence"
does not yet establish first-order stationarity.

## V8.3.2 stationarity diagnostic (pre-registered before gradient results)

`diagnose_dev_stationarity.py` reads and verifies BOTH frozen JSON
archives (V8.3 SHA256 and V8.3.1 SHA256), re-creates the same
two 0027 graphs and re-executes the original four solves solely
because the prior JSON retained no optimized Pose3 matrices.

For each initial and final pose estimate, query
`graph.linearize(values).gradientAtZero().norm()` as documented in
GTSAM. The L2 norm of `J^T r` is a first-order diagnostic in mixed
Pose3 tangent coordinates; there is no pre-selected universal
absolute threshold for "stationary". Report absolute norm and
per-sqrt-4080 tangent-dimension descriptive normalization, within
the SAME graph. Do not compare objectives between differently
weighted graphs, and do not call a stationary point a global optimum.

The stage does **not** open the OXTS clean reference, evaluate
position RMSE, retrain models, adjust solver configurations,
read held-out sequences 0016/0020 or modify V8.3/V8.3.1 results.

```bash
python -m pytest tests/test_isj_v7_*.py tests/test_isj_v8_*.py -q
python -m experiments.isj_v8.diagnose_dev_stationarity --preflight
# Only once tests+preflight pass:
python -u -m experiments.isj_v8.diagnose_dev_stationarity \
  --run-diagnostic --token V8_3_2_0027_STATIONARITY_DIAGNOSTIC
```

Exactly one new directory is generated, never overwritten:
`results/isj_v8/dev_0027_stationarity_audit_v1`.

Interpretation: a large tangent gradient at a no-progress "stable"
LM output directly invalidates the claim that the outer stopping
flag verifies first-order convergence. A small gradient after two
initializations would support genuine distinct stationary points,
but would not by itself show local minimum (second-order test
would be needed). In either case, the V8.3 FGO results are
development-only, non-independent, and insufficient for claiming
universal Mamba prediction-gated robustness.
