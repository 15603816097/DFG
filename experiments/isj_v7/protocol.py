from __future__ import annotations

"""Read-only V7 temporal-label and provenance preflight for development 0027.

IMPORTANT: horizon boundaries are controlled here, but V3 'current'
labels are offline oracle/full-trajectory FGO proxies. Thus the legacy
V3 archive is NOT an approved independent supervisory target source.
No weights are trained, no model checkpoints are loaded, and no files
are written by this module.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (ROOT / "results/predictive_factor_reliability_v3_same_source"
                  / "factor_reliability_training_data.npz")
SENSOR_DIMS = {"gps": 21, "imu": 20, "camera": 16}
LABEL_PROVENANCE = (
    "V3 GPS target uses clean-vs-degraded GPS oracle; V3 IMU/LiDAR/Camera "
    "targets use fixed FGO over the complete 0027 trajectory. Neither "
    "is independently validated online ground truth."
)


@dataclass(frozen=True)
class SplitPlan:
    frames: int
    window: int
    horizon: int
    train_end: int
    val_end: int

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(self.horizon))

    def indices(self, part: str) -> np.ndarray:
        """Anchor t; all targets t..t+horizon-1 stay IN the given split.

        Evaluation windows may read past input features from the prior
        partition, never future features. This does not prove that the
        upstream feature engineering or offline targets are causal.
        """
        if part == "train":
            low, high = self.window - 1, self.train_end
        elif part == "val":
            low, high = self.train_end, self.val_end
        elif part == "test":
            low, high = self.val_end, self.frames
        else:
            raise ValueError(f"unknown split: {part}")
        last_exclusive = high - (self.horizon - 1)
        return np.arange(low, last_exclusive, dtype=np.int64)

    def boundary(self, part: str) -> tuple[int, int]:
        if part == "train":
            return 0, self.train_end
        if part == "val":
            return self.train_end, self.val_end
        if part == "test":
            return self.val_end, self.frames
        raise ValueError(f"unknown split: {part}")


def plan_splits(n: int, *, window: int = 64, horizon: int = 3,
                train_fraction: float = 0.70,
                val_fraction: float = 0.85) -> SplitPlan:
    if (not isinstance(n, (int, np.integer)) or
            not isinstance(window, (int, np.integer)) or
            not isinstance(horizon, (int, np.integer)) or
            window < 1 or horizon < 1 or
            not 0 < train_fraction < val_fraction < 1):
        raise ValueError("invalid split or horizon configuration")
    train_end = int(n * train_fraction)
    val_end = int(n * val_fraction)
    plan = SplitPlan(int(n), int(window), int(horizon), train_end, val_end)
    if any(len(plan.indices(p)) == 0 for p in ("train", "val", "test")):
        raise ValueError("empty train/val/test after horizon purge")
    return plan


def aligned_targets(current: np.ndarray, anchors: np.ndarray,
                    horizon: int = 3) -> np.ndarray:
    """Heads [current[t],current[t+1],...] WITHOUT V3 padded future."""
    y = np.asarray(current)
    idx = np.asarray(anchors)
    if (y.ndim != 1 or not np.isfinite(y).all() or
            np.any((y < 0) | (y > 1)) or
            idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or
            horizon < 1):
        raise ValueError("invalid reliability labels or anchor indices")
    if len(idx) and (idx.min() < 0 or idx.max() + horizon > len(y)):
        raise ValueError("targets exceed available current labels")
    return np.asarray(y[idx[:, None] + np.arange(horizon)[None, :]],
                      dtype=np.float32)


def robust_train_scaler(features: np.ndarray,
                        train_end: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    if (x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all() or
            not isinstance(train_end, (int, np.integer)) or
            not 1 <= train_end <= len(x)):
        raise ValueError("invalid train-only scaler inputs")
    train_x = x[:train_end]
    center = np.median(train_x, axis=0).astype(np.float32)
    p25, p75 = np.percentile(train_x, [25, 75], axis=0)
    scale = np.asarray(p75 - p25, dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    return center, scale


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    h = hashlib.sha256()
    h.update(str(x.shape).encode("ascii"))
    h.update(x.tobytes())
    return h.hexdigest()


def audit_archive(source: Path = DEFAULT_SOURCE, *, window: int = 64,
                  horizon: int = 3) -> dict:
    """Validate V3 archive and describe V7 preflight; never approve V3 labels.

    This intentionally ignores *_future_factor_reliability, whose
    single column is t+3, not a sequence of t+1,t+2 heads.
    """
    source = Path(source)
    if any(part in {"0016", "0020"} for part in source.parts):
        raise ValueError("0016/0020 must remain untouched in this V7 preflight")
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        inspected: dict[str, dict] = {}
        n: int | None = None
        plan: SplitPlan | None = None
        for sensor, dim in SENSOR_DIMS.items():
            fk = f"{sensor}_features"
            ck = f"{sensor}_current_factor_reliability"
            if fk not in archive.files or ck not in archive.files:
                raise KeyError(f"missing {fk} or {ck}")
            x = np.asarray(archive[fk], dtype=np.float32)
            current = np.asarray(archive[ck], dtype=np.float32)
            if (x.ndim != 2 or x.shape[1] != dim or
                    current.ndim != 1 or len(x) != len(current) or
                    not np.isfinite(x).all()):
                raise ValueError(f"{sensor}: shape, length or feature validity error")
            if n is None:
                n = len(x)
                plan = plan_splits(n, window=window, horizon=horizon)
            elif len(x) != n:
                raise ValueError(f"{sensor}: inconsistent frame count")
            assert plan is not None
            center, scale = robust_train_scaler(x, plan.train_end)
            targets = {}
            for part in ("train", "val", "test"):
                anchors = plan.indices(part)
                lo, hi = plan.boundary(part)
                if (len(anchors) == 0 or anchors.min() < lo and part != "train" or
                        anchors.max() + horizon - 1 >= hi):
                    raise AssertionError("internal future target boundary breach")
                aligned = aligned_targets(current, anchors, horizon)
                targets[part] = {"count": int(len(anchors)),
                                 "first_anchor": int(anchors[0]),
                                 "last_anchor": int(anchors[-1]),
                                 "first_target": int(anchors[0]),
                                 "last_target": int(anchors[-1] + horizon - 1),
                                 "aligned_target_sha256": _array_digest(aligned)}
            inspected[sensor] = {
                "feature_dim": dim,
                "features_sha256_float32": _array_digest(x),
                "current_labels_sha256_float32": _array_digest(current),
                "train_only_median_sha256": _array_digest(center),
                "train_only_iqr_sha256": _array_digest(scale),
                "samples": targets,
            }
    assert n is not None and plan is not None
    return {
        "schema": "isj_v7_preflight_v1",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "frames": n,
        "window": window,
        "horizon": horizon,
        "head_offsets": list(plan.offsets),
        "train_end_exclusive": plan.train_end,
        "val_end_exclusive": plan.val_end,
        "split_rule": "70/15/15 frames; purge last H-1 anchors per split",
        "normalizer_rule": "same train-only median/IQR for future GRU and Mamba",
        "legacy_future_column_ignored": True,
        "label_provenance_warning": LABEL_PROVENANCE,
        "upstream_feature_causality_reviewed": False,
        "independent_supervision_verified": False,
        "approved_for_final_training": False,
        "sensors": inspected,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args(argv)
    report = audit_archive(args.source, window=args.window,
                           horizon=args.horizon)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("BLOCKED: V3 proxy labels are not approved final-training targets.")


if __name__ == "__main__":
    main()
