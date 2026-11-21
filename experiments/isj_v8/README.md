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