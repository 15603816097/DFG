from __future__ import annotations

"""V7 GPS-only GRU/Mamba development comparison, NEVER final-paper training.

Only exported 0027 GPS train+validation synthetic-injection dataset is
accepted. No 0027 internal test segment, no independent drives (0020,
0016), no FGO, no clean GPS input and no old V5/V6 checkpoint loading.
The saved labels are synthetic supervision, NOT physical GPS truth.

Usage:
  python -m experiments.isj_v7.train_gps_dev --backend gru --smoke-only
  python -m experiments.isj_v7.train_gps_dev --backend gru
  python -m experiments.isj_v7.train_gps_dev --backend mamba

Mamba requires working actual mamba_ssm and NVIDIA CUDA. No fallback.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.isj_v7.build_dev_gps_candidate import DEFAULT_DEST
from experiments.isj_v7.audit_gps_canonical_dryrun import _array_sha
from experiments.isj_v7.model_dev_gps import (
    ReliabilityForecastV7, evidential_loss, parameter_count,
)
from experiments.isj_v7.supervision_gate import read_policy, validate_policy


EXPECTED_KEYS = {
    "gps_features", "train_anchors", "val_anchors",
    "train_targets", "val_targets", "train_median", "train_iqr",
}
DEFAULT_OUTPUT = DEFAULT_DEST.parent / "dev_gps_baseline_v1"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dev_data(root: Path = DEFAULT_DEST) -> tuple[dict, dict]:
    root = Path(root).resolve()
    # No legacy files or held-out drives may be substituted.
    if root != DEFAULT_DEST.resolve():
        raise ValueError("only the frozen V7 GPS 0027 candidate is accepted")
    npz = root / "gps_train_val.npz"
    metadata_path = root / "provenance.json"
    if not npz.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("missing GPS V7 development archive or provenance")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema") != "isj_v7_dev_gps_same_source_v1"
            or metadata.get("development_only") is not True
            or metadata.get("approved_for_final_training") is not False
            or metadata.get("approved_for_independent_test") is not False
            or metadata.get("do_not_treat_as_independent_truth") is not True
            or metadata.get("source_drive") != "2011_10_03_drive_0027_sync"
            or metadata.get("source_frames") != 4544
            or metadata.get("exported_feature_frames") != 3862
            or metadata.get("omitted_internal_test_range") != [3862, 4544]
            or metadata.get("window") != 64
            or metadata.get("forecast_heads") != [0, 1, 2]
            or metadata.get("train_end_exclusive") != 3180
            or metadata.get("val_end_exclusive") != 3862):
        raise ValueError("frozen GPS development provenance is invalid")
    policy = read_policy()
    validate_policy(policy)  # requires FINAL_TRAINING_BLOCKED still true
    src_hash = metadata.get("source_archive_sha256", {})
    for name in ("v3", "degraded_sensor"):
        if src_hash.get(name) != policy["raw_sources"][name]["sha256"]:
            raise ValueError("GPS source SHA256 does not match frozen policy")

    with np.load(npz, allow_pickle=False) as src:
        if set(src.files) != EXPECTED_KEYS:
            raise ValueError("unrecognized data keys (possible oracle/test leakage)")
        data = {name: np.asarray(src[name]).copy() for name in src.files}

    if (data["gps_features"].shape != (3862, 21)
            or data["train_anchors"].shape != (3115,)
            or data["train_targets"].shape != (3115, 3)
            or data["val_anchors"].shape != (680,)
            or data["val_targets"].shape != (680, 3)
            or data["train_median"].shape != (21,)
            or data["train_iqr"].shape != (21,)):
        raise ValueError("expected 0027 GPS development shapes were changed")
    if not np.array_equal(data["train_anchors"],
                          np.arange(63, 3178, dtype=np.int64)):
        raise ValueError("train anchor contamination")
    if not np.array_equal(data["val_anchors"],
                          np.arange(3180, 3860, dtype=np.int64)):
        raise ValueError("validation anchor contamination")
    for name, a in data.items():
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite values in {name}")
        declared_dtype = metadata.get("array_dtypes", {}).get(name)
        if str(a.dtype) != declared_dtype:
            raise ValueError(f"{name} changed dtype")
        digest = metadata.get("array_sha256_float32", {}).get(name)
        if _array_sha(a) != digest:
            raise ValueError(f"{name} SHA256 mismatch")
    if np.any((data["train_targets"] < 0) | (data["train_targets"] > 1)):
        raise ValueError("invalid train target range")
    if np.any((data["val_targets"] < 0) | (data["val_targets"] > 1)):
        raise ValueError("invalid val target range")
    if np.any(data["train_iqr"] <= 0):
        raise ValueError("invalid train IQR")
    # Validate saved scaler against actual train prefix, not VAL.
    from experiments.isj_v7.protocol import robust_train_scaler
    center, scale = robust_train_scaler(data["gps_features"], 3180)
    if not np.array_equal(data["train_median"], center) or not np.array_equal(
            data["train_iqr"], scale):
        raise ValueError("normalization does not come from train prefix")

    metadata["archive_sha256"] = file_sha(npz)
    metadata["manifest_sha256"] = file_sha(metadata_path)
    return data, metadata


class WindowDataset(Dataset):
    def __init__(
        self, x: np.ndarray, anchors: np.ndarray, targets: np.ndarray,
        median: np.ndarray, iqr: np.ndarray, *, window: int = 64,
        clip: float = 20.0,
    ):
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        if (window < 1 or clip <= 0 or not np.isfinite(clip)
                or self.targets.shape != (len(self.anchors), 3)
                or x.ndim != 2 or len(self.anchors) == 0
                or self.anchors.min() < window - 1
                or self.anchors.max() >= len(x)
                or median.shape != (x.shape[1],)
                or iqr.shape != (x.shape[1],)):
            raise ValueError("invalid window dataset inputs")
        xx = (x.astype(np.float32) - median) / iqr
        self.x = np.clip(xx, -clip, clip).astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, item):
        t = int(self.anchors[item])
        # t is prediction time, so no future features are ever returned.
        window_x = self.x[t - self.window + 1:t + 1]
        return (
            torch.from_numpy(window_x.copy()),
            torch.from_numpy(self.targets[item].copy()),
            t,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    data: dict, *, batch_size: int,
    window: int, clip: float, device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    args = (data["gps_features"], data["train_median"], data["train_iqr"])
    train_ds = WindowDataset(
        args[0], data["train_anchors"], data["train_targets"],
        args[1], args[2], window=window, clip=clip)
    val_ds = WindowDataset(
        args[0], data["val_anchors"], data["val_targets"],
        args[1], args[2], window=window, clip=clip)
    loader_args = dict(
        batch_size=batch_size, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return (
        DataLoader(
            train_ds, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **loader_args,
        ),
        DataLoader(val_ds, shuffle=False, **loader_args),
    )


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    totals = 0.0
    losses = 0.0
    prediction, uncertainty, target, anchors = [], [], [], []
    for xb, yb, t in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        loss = evidential_loss(out, yb)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite validation loss")
        totals += int(len(xb))
        losses += float(loss.item()) * len(xb)
        prediction.append(out["reliability"].detach().float().cpu().numpy())
        uncertainty.append(out["uncertainty"].detach().float().cpu().numpy())
        target.append(yb.detach().float().cpu().numpy())
        anchors.append(np.asarray(t, dtype=np.int64))
    if totals == 0:
        raise ValueError("empty validation loader")
    return (
        losses / totals,
        np.concatenate(prediction),
        np.concatenate(uncertainty),
        np.concatenate(target),
        np.concatenate(anchors),
    )


def score_heads(pred, truth, uncertainty, train_truth) -> dict:
    if (pred.shape != truth.shape or pred.shape != uncertainty.shape
            or pred.ndim != 2 or train_truth.ndim != 2
            or train_truth.shape[1] != pred.shape[1]
            or not np.isfinite(pred).all()
            or not np.isfinite(truth).all()
            or not np.isfinite(uncertainty).all()):
        raise ValueError("invalid validation predictions")
    out = {}
    train_mean = np.mean(train_truth, axis=0)
    for k in range(pred.shape[1]):
        err = pred[:, k] - truth[:, k]
        baseline = train_mean[k] - truth[:, k]
        out[f"h{k}"] = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "train_mean_baseline_MAE": float(np.mean(np.abs(baseline))),
            "train_mean_baseline_RMSE": float(
                np.sqrt(np.mean(baseline ** 2))),
            "always_one_baseline_MAE": float(
                np.mean(np.abs(1.0 - truth[:, k]))),
            "uncertainty_mean": float(np.mean(uncertainty[:, k])),
            "uncertainty_std": float(np.std(uncertainty[:, k])),
        }
    return out


def choose_device(request: str, backend: str) -> torch.device:
    if request not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto/cpu/cuda")
    if request == "cuda" or request == "auto" and torch.cuda.is_available():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but unavailable")
        target = torch.device("cuda")
    else:
        target = torch.device("cpu")
    if backend == "mamba" and target.type != "cuda":
        raise RuntimeError(
            "V7 backend=mamba requires NVIDIA CUDA and working mamba_ssm; "
            "CPU cannot silently run a GRU instead."
        )
    return target


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("ascii").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run(args: argparse.Namespace) -> dict:
    if (args.epochs < 1 or args.patience < 1 or args.batch_size < 1
            or args.hidden < 1 or args.layers < 1 or args.seed < 0
            or args.lr <= 0 or args.weight_decay < 0
            or args.min_delta < 0):
        raise ValueError("invalid training arguments")
    if args.backend not in {"gru", "mamba"}:
        raise ValueError("choose either gru or mamba, no backend fallback")
    # Pass the live frozen development path explicitly. Python binds
    # default argument objects at function definition, so omitting this
    # argument would bypass a controlled test-time path injection.
    data, metadata = load_dev_data(DEFAULT_DEST)
    device = choose_device(args.device, args.backend)
    seed_everything(args.seed)
    model = ReliabilityForecastV7(
        backend=args.backend, input_dim=21, hidden=args.hidden,
        layers=args.layers, horizon=3, dropout=args.dropout,
    ).to(device)
    train_loader, val_loader = make_loaders(
        data, batch_size=args.batch_size,
        window=64, clip=args.input_clip, device=device, seed=args.seed,
    )
    parameters = parameter_count(model)
    print(
        f"V7 GPS DEVELOPMENT backend={args.backend} device={device} "
        f"params={parameters} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"input_clip={args.input_clip}", flush=True,
    )
    print("WARNING: validation is within already explored 0027; "
          "no independent drive score is computed.", flush=True)

    x, y, _ = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    output = model(x)
    smoke_loss = evidential_loss(output, y)
    if not torch.isfinite(smoke_loss):
        raise RuntimeError("nonfinite first-batch loss")
    smoke_loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters()):
        raise RuntimeError("nonfinite gradients")
    model.zero_grad(set_to_none=True)
    print(f"MODEL_FORWARD_BACKWARD_OK loss={float(smoke_loss):.6f}",
          flush=True)
    if args.smoke_only:
        print("SMOKE_ONLY: no optimizer steps, checkpoint or files written.",
              flush=True)
        return {
            "backend": args.backend, "device": str(device),
            "parameters": parameters, "smoke_loss": float(smoke_loss),
            "development_only": True,
        }

    if args.output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("training outputs must stay in frozen V7 dev root")
    dest = args.output / args.backend / f"seed_{args.seed}"
    if dest.exists():
        raise FileExistsError(
            f"existing run: {dest}; refuse to overwrite. Use a new seed.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=False)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = evidential_loss(out, y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sum += float(loss.item()) * len(x)
            train_count += len(x)
        val_loss, _, _, _, _ = evaluate(model, val_loader, device)
        train_loss = train_sum / train_count
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(f"epoch={epoch:03d} train={train_loss:.6f} "
              f"val={val_loss:.6f}", flush=True)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                print(f"EARLY_STOP at epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("no valid best model")
    model.load_state_dict(best_state, strict=True)
    final_loss, predictions, uncertainties, targets, anchors = evaluate(
        model, val_loader, device)
    if not np.array_equal(anchors, data["val_anchors"]):
        raise AssertionError("validation anchors changed during training")
    metrics = {
        "schema": "isj_v7_dev_gps_trainval_baseline_v1",
        "backend": args.backend,
        "device": str(device),
        "parameter_count": parameters,
        "minibatch_order": "fixed independent torch Generator(seed)",
        "development_only": True,
        "approved_for_final_training": False,
        "independent_test_run": False,
        "validation_is_from_0027_development": True,
        "uncertainty_meaning": (
            "Beta evidence/concentration proxy, not calibrated covariance "
            "or established epistemic uncertainty."
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "recomputed_best_val_loss": final_loss,
        "val_heads": score_heads(
            predictions, targets, uncertainties, data["train_targets"]),
        "epoch_count": len(history),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "git_commit": current_git_commit(),
        "dataset_sha256": metadata["archive_sha256"],
        "dataset_manifest_sha256": metadata["manifest_sha256"],
        "config": {
            "seed": args.seed,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "input_clip": args.input_clip,
            "window": 64,
            "heads": [0, 1, 2],
        },
        "torch": torch.__version__,
    }
    # Output save never contains clean GPS or held-out test.
    torch.save({
        "model_state_dict": best_state,
        "backend": args.backend,
        "model_kwargs": {
            "backend": args.backend, "input_dim": 21,
            "hidden": args.hidden, "layers": args.layers,
            "horizon": 3, "dropout": args.dropout,
        },
        "train_median": data["train_median"],
        "train_iqr": data["train_iqr"],
        "development_only": True,
        "approved_for_final_training": False,
        "dataset_sha256": metadata["archive_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "git_commit": metrics["git_commit"],
    }, dest / "best.pt")
    np.savez_compressed(
        dest / "val_predictions.npz",
        anchors=anchors, reliability=predictions,
        evidence_uncertainty=uncertainties,
        synthetic_targets=targets,
    )
    (dest / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (dest / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("V7_GPS_DEV_COMPLETE output=", dest, flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("gru", "mamba"), required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--input-clip", type=float, default=20.0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--smoke-only", action="store_true",
                   help="one forward/backward pass without optimizer or writes")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
