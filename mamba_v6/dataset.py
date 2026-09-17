
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_DIMS = {"gps": 21, "imu": 20, "camera": 16}


def _key(data, *candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of these keys exist: {candidates}. Available={list(data.keys())}")


def load_sensor_arrays(npz_path: str | Path, sensor: str):
    data = np.load(npz_path, allow_pickle=False)
    fkey = _key(data, f"{sensor}_features")
    ckey = _key(data, f"{sensor}_current_factor_reliability")
    futkey = _key(data, f"{sensor}_future_factor_reliability")

    x = np.asarray(data[fkey], dtype=np.float32)
    current = np.asarray(data[ckey], dtype=np.float32).reshape(-1)
    future = np.asarray(data[futkey], dtype=np.float32)

    if future.ndim == 1:
        future = future[:, None]

    if not (len(x) == len(current) == len(future)):
        raise ValueError(
            f"{sensor} length mismatch: features={len(x)}, "
            f"current={len(current)}, future={len(future)}"
        )
    return x, np.clip(current, 0, 1), np.clip(future, 0, 1)


def build_targets(current, future, horizon=3):
    # Preserve the same V5 semantics used by the current project:
    # h0=current[t], h1=stored future[t], remaining horizons fall back
    # to shifted current labels when only one stored future column exists.
    n = len(current)
    y = np.full((n, horizon), np.nan, dtype=np.float32)
    y[:, 0] = current

    for h in range(1, horizon):
        if future.shape[1] >= h:
            y[:, h] = future[:, h - 1]
        else:
            valid = np.arange(n - h)
            y[valid, h] = current[valid + h]
    return y


@dataclass
class Split:
    train_end: int
    val_end: int
    usable_end: int


def chronological_split(n, window=64, horizon=3):
    usable_end = n - horizon + 1
    train_end = int(usable_end * 0.70)
    val_end = int(usable_end * 0.85)
    return Split(train_end, val_end, usable_end)


class SequenceDataset(Dataset):
    def __init__(self, x, y, indices, window, mean, scale):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = window
        self.mean = mean
        self.scale = scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        t = int(self.indices[j])
        seq = self.x[t-self.window+1:t+1]
        seq = (seq - self.mean) / self.scale
        return (
            torch.from_numpy(seq.astype(np.float32)),
            torch.from_numpy(self.y[t].astype(np.float32)),
            t,
        )


def make_datasets(npz_path, sensor, window=64, horizon=3):
    x, current, future = load_sensor_arrays(npz_path, sensor)
    y = build_targets(current, future, horizon=horizon)
    split = chronological_split(len(x), window, horizon)

    first = window - 1
    train_idx = np.arange(first, split.train_end)
    val_idx = np.arange(max(first, split.train_end), split.val_end)
    test_idx = np.arange(max(first, split.val_end), split.usable_end)

    # Strict training-only normalization.
    norm_x = x[:split.train_end]
    mean = np.median(norm_x, axis=0).astype(np.float32)
    q25 = np.percentile(norm_x, 25, axis=0)
    q75 = np.percentile(norm_x, 75, axis=0)
    scale = (q75 - q25).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    return (
        SequenceDataset(x, y, train_idx, window, mean, scale),
        SequenceDataset(x, y, val_idx, window, mean, scale),
        SequenceDataset(x, y, test_idx, window, mean, scale),
        dict(
            x=x, y=y, current=current, future=future,
            mean=mean, scale=scale, split=split,
        ),
    )
