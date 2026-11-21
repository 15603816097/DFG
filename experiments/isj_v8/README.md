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