from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence.stage7_config import (
    V6_DIR,
    TEST_SEQUENCES,
    sequence_out,
)
from mamba_v6.model import MambaReliabilityUncertaintyV6


SENSORS = ("gps", "imu", "camera")


def _load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"V6 checkpoint not found: {path}")

    # These are our own trusted project checkpoints.
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Expected dict checkpoint, got {type(ckpt)}: {path}"
        )

    required = (
        "model",
        "sensor",
        "input_dim",
        "hidden",
        "horizon",
        "layers",
        "window",
        "mean",
        "scale",
    )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"V6 checkpoint missing keys {missing}: {path}"
        )

    return ckpt


def _load_features(sequence: str, sensor: str):
    feature_path = sequence_out(sequence) / "features.npz"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Stage-7 features not found: {feature_path}"
        )

    with np.load(feature_path, allow_pickle=False) as data:
        key = f"{sensor}_features"
        if key not in data.files:
            raise KeyError(
                f"{key} not found in {feature_path}. "
                f"Available keys: {data.files}"
            )

        x = np.asarray(data[key], dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(
            f"{sequence}/{sensor}: expected 2-D features, got {x.shape}"
        )

    if len(x) == 0:
        raise ValueError(
            f"{sequence}/{sensor}: empty feature array"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{sequence}/{sensor}: features contain NaN/Inf"
        )

    return x


def _validate_checkpoint(
    ckpt: dict,
    sensor: str,
    x: np.ndarray,
):
    ckpt_sensor = str(ckpt["sensor"]).lower()

    if ckpt_sensor != sensor:
        raise ValueError(
            f"Checkpoint sensor mismatch: "
            f"requested={sensor}, checkpoint={ckpt_sensor}"
        )

    input_dim = int(ckpt["input_dim"])
    if x.shape[1] != input_dim:
        raise ValueError(
            f"{sensor}: feature dimension mismatch: "
            f"target={x.shape[1]}, checkpoint={input_dim}"
        )

    mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1)
    scale = np.asarray(ckpt["scale"], dtype=np.float32).reshape(-1)

    if len(mean) != input_dim or len(scale) != input_dim:
        raise ValueError(
            f"{sensor}: normalization dimension mismatch: "
            f"input_dim={input_dim}, "
            f"mean={mean.shape}, scale={scale.shape}"
        )

    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{sensor}: checkpoint mean contains NaN/Inf")

    if not np.all(np.isfinite(scale)):
        raise ValueError(f"{sensor}: checkpoint scale contains NaN/Inf")

    # Preserve the frozen training normalization.
    # Do NOT estimate normalization statistics from Stage-7 sequences.
    scale = scale.copy()
    scale[scale < 1e-6] = 1.0

    return mean, scale


def _build_model(
    ckpt: dict,
    device: torch.device,
):
    model = MambaReliabilityUncertaintyV6(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden"]),
        horizon=int(ckpt["horizon"]),
        num_layers=int(ckpt["layers"]),
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def infer(
    sequence: str,
    sensor: str,
    device: torch.device,
    batch_size: int = 256,
):
    if sequence not in TEST_SEQUENCES:
        raise ValueError(
            f"Stage-7 inference is restricted to held-out sequences: "
            f"{TEST_SEQUENCES}"
        )

    if sensor not in SENSORS:
        raise ValueError(f"Unsupported sensor: {sensor}")

    x = _load_features(sequence, sensor)
    n = len(x)

    checkpoint_path = V6_DIR / sensor / "best.pt"
    ckpt = _load_checkpoint(checkpoint_path)

    mean, scale = _validate_checkpoint(
        ckpt,
        sensor,
        x,
    )

    window = int(ckpt["window"])
    horizon = int(ckpt["horizon"])

    if window <= 0:
        raise ValueError(f"Invalid V6 window: {window}")

    if horizon <= 0:
        raise ValueError(f"Invalid V6 horizon: {horizon}")

    if n < window:
        raise ValueError(
            f"{sequence}/{sensor}: sequence length {n} "
            f"is shorter than window {window}"
        )

    model = _build_model(
        ckpt,
        device,
    )

    # EXACT V6 training semantics:
    # train.py initializes target-aligned arrays with
    # reliability=1 and uncertainty=1 before the first valid window.
    reliability = np.ones(n, dtype=np.float32)
    uncertainty = np.ones(n, dtype=np.float32)

    valid_indices = np.arange(
        window - 1,
        n,
        dtype=np.int64,
    )

    for start in range(0, len(valid_indices), batch_size):
        ids = valid_indices[start:start + batch_size]

        batch = np.stack(
            [
                (x[t - window + 1:t + 1] - mean) / scale
                for t in ids
            ],
            axis=0,
        ).astype(np.float32)

        batch_tensor = torch.from_numpy(batch).to(
            device,
            non_blocking=device.type == "cuda",
        )

        output = model(batch_tensor)

        r = output["reliability"][:, 0]
        u = output["uncertainty"][:, 0]

        reliability[ids] = r.detach().cpu().numpy().astype(np.float32)
        uncertainty[ids] = u.detach().cpu().numpy().astype(np.float32)

    if not np.all(np.isfinite(reliability)):
        raise RuntimeError(
            f"{sequence}/{sensor}: reliability contains NaN/Inf"
        )

    if not np.all(np.isfinite(uncertainty)):
        raise RuntimeError(
            f"{sequence}/{sensor}: uncertainty contains NaN/Inf"
        )

    reliability = np.clip(reliability, 0.0, 1.0)
    uncertainty = np.clip(uncertainty, 0.0, 1.0)

    out_dir = (
        sequence_out(sequence)
        / "predictions"
        / "v6"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reliability_path = (
        out_dir
        / f"{sensor}_reliability.txt"
    )
    uncertainty_path = (
        out_dir
        / f"{sensor}_uncertainty.txt"
    )

    np.savetxt(
        reliability_path,
        reliability,
        fmt="%.9f",
    )
    np.savetxt(
        uncertainty_path,
        uncertainty,
        fmt="%.9f",
    )

    print(
        f"{sequence} {sensor} V6 "
        f"n={n} "
        f"window={window} "
        f"rmean={reliability.mean():.6f} "
        f"umean={uncertainty.mean():.6f}"
    )

    return reliability, uncertainty


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        required=True,
        choices=TEST_SEQUENCES,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    print("=" * 80)
    print("STAGE-7 OFFICIAL V6 MAMBA FROZEN INFERENCE")
    print("=" * 80)
    print("sequence :", args.sequence)
    print("device   :", device)
    print("training : DISABLED")
    print("checkpoint normalization : FROZEN")
    print()

    for sensor in SENSORS:
        infer(
            args.sequence,
            sensor,
            device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
