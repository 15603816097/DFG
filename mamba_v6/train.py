
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mamba_v6.dataset import make_datasets
from mamba_v6.model import MambaReliabilityUncertaintyV6


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_nll(alpha, beta, target):
    target = target.clamp(1e-4, 1.0 - 1e-4)
    dist = torch.distributions.Beta(alpha, beta)
    return -dist.log_prob(target)


def loss_fn(out, target):
    nll = beta_nll(out["alpha"], out["beta"], target)
    weights = torch.tensor(
        [1.0, 0.7, 0.5],
        device=target.device,
        dtype=target.dtype,
    )[:target.shape[1]]
    return (nll * weights[None, :]).mean()


def corrcoef(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, uncs, tgts, idxs = [], [], [], []
    losses = []
    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(float(loss_fn(out, y).item()))
        preds.append(out["reliability"].cpu().numpy())
        uncs.append(out["uncertainty"].cpu().numpy())
        tgts.append(y.cpu().numpy())
        idxs.append(np.asarray(idx))
    return (
        float(np.mean(losses)) if losses else math.inf,
        np.concatenate(preds),
        np.concatenate(uncs),
        np.concatenate(tgts),
        np.concatenate(idxs),
    )


def train_sensor(args, sensor):
    train_ds, val_ds, test_ds, meta = make_datasets(
        args.data, sensor, args.window, args.horizon
    )
    device = torch.device("cuda")
    model = MambaReliabilityUncertaintyV6(
        input_dim=meta["x"].shape[1],
        hidden_dim=args.hidden,
        horizon=args.horizon,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    out_dir = Path(args.output) / sensor
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_val = math.inf
    bad = 0
    history = []

    print(f"\n{'='*80}\n{sensor.upper()} V6-MAMBA\n{'='*80}")
    print("backend=mamba")
    print("device=", device)
    print(
        f"input_dim={meta['x'].shape[1]}, window={args.window}, "
        f"H={args.horizon}, hidden={args.hidden}, layers={args.layers}"
    )
    print(
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss, _, _, _, _ = evaluate(model, val_loader, device)
        history.append((epoch, train_loss, val_loss))
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "sensor": sensor,
                    "input_dim": meta["x"].shape[1],
                    "hidden": args.hidden,
                    "horizon": args.horizon,
                    "layers": args.layers,
                    "window": args.window,
                    "mean": meta["mean"],
                    "scale": meta["scale"],
                    "best_val": best_val,
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, pred, unc, target, idx = evaluate(model, test_loader, device)

    p0, t0 = pred[:, 0], target[:, 0]
    mae = float(np.mean(np.abs(p0 - t0)))
    rmse = float(np.sqrt(np.mean((p0 - t0) ** 2)))
    corr = corrcoef(p0, t0)

    print(
        f"TEST {sensor}: loss={test_loss:.6f}, MAE={mae:.6f}, "
        f"RMSE={rmse:.6f}, Corr={corr:.6f}"
    )
    print(
        "uncertainty test min/max/mean/std "
        f"{unc[:,0].min():.6f}/{unc[:,0].max():.6f}/"
        f"{unc[:,0].mean():.6f}/{unc[:,0].std():.6f}"
    )

    n = len(meta["x"])
    aligned_r = np.ones(n, dtype=np.float32)
    aligned_u = np.ones(n, dtype=np.float32)

    # Fill all available train/val/test target-aligned predictions.
    full_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])
    full_loader = DataLoader(
        full_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    _, all_pred, all_unc, _, all_idx = evaluate(model, full_loader, device)
    aligned_r[all_idx] = all_pred[:, 0]
    aligned_u[all_idx] = all_unc[:, 0]

    np.savetxt(
        Path(args.output) / f"{sensor}_reliability_target_aligned.txt",
        aligned_r, fmt="%.9f"
    )
    np.savetxt(
        Path(args.output) / f"{sensor}_uncertainty_target_aligned.txt",
        aligned_u, fmt="%.9f"
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sensor": sensor,
                "backend": "mamba",
                "best_val": best_val,
                "test_loss": test_loss,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_corr": corr,
                "uncertainty_min": float(unc[:,0].min()),
                "uncertainty_max": float(unc[:,0].max()),
                "uncertainty_mean": float(unc[:,0].mean()),
                "uncertainty_std": float(unc[:,0].std()),
            },
            f, indent=2
        )

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        w.writerows(history)

    return {"sensor": sensor, "mae": mae, "rmse": rmse, "corr": corr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/factor_reliability_training_data.npz")
    p.add_argument("--output", default="results/v6_mamba_uncertainty")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. V6-Mamba cloud experiment must run on NVIDIA GPU."
        )

    seed_all(42)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    rows = [train_sensor(args, s) for s in ("gps", "imu", "camera")]

    print("\n" + "="*80)
    print("V6 MAMBA SUMMARY")
    print("="*80)
    for r in rows:
        print(
            f"{r['sensor']:8s} MAE={r['mae']:.6f} "
            f"RMSE={r['rmse']:.6f} Corr={r['corr']:.6f}"
        )


if __name__ == "__main__":
    main()
