"""
Train / validate / test an MLP age regressor from face morphometric features.

Features (--features):
  gpa   — GPA PC scores only  (≤50 dims)
  edma  — EDMA pairwise distances  (190 dims)
  both  — concatenated  (default, ≤240 dims)

Outputs (all under data/models/<run_name>/):
  model.pt             PyTorch MLP weights
  scaler.joblib        fitted StandardScaler  (train only)
  config.json          architecture + training hyperparameters
  train_predictions.csv / val_predictions.csv / test_predictions.csv

Usage:
  python scripts/train_age_regressor.py
  python scripts/train_age_regressor.py --features gpa --hidden 128 64 32
  python scripts/train_age_regressor.py --features both --lr 3e-4 --dropout 0.3
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ROOT     = Path(__file__).resolve().parents[1]
_FEAT_DIR = _ROOT / "data" / "features"
_MDL_DIR  = _ROOT / "data" / "models"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str, feature_set: str) -> dict:
    """Load one .npz split and return X, y and metadata."""
    path = _FEAT_DIR / f"{split}.npz"
    if not path.exists():
        log.error("Feature file not found: %s  (run extract_features.py first)", path)
        sys.exit(1)

    d = np.load(str(path), allow_pickle=True)

    if feature_set == "gpa":
        X = d["gpa_scores"].astype(np.float32)
    elif feature_set == "edma":
        X = d["edma_distances"].astype(np.float32)
    else:  # both
        X = np.concatenate([d["gpa_scores"], d["edma_distances"]], axis=1).astype(np.float32)

    return {
        "X":           X,
        "y":           d["ages"].astype(np.float32),
        "subject_ids": d["subject_ids"],
        "sexes":       d["sexes"],
        "sites":       d["sites"],
    }


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    r, _ = pearsonr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r": float(r), "r2": r2}


def report_metrics(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sexes: np.ndarray,
    sites: np.ndarray,
) -> dict:
    m = compute_metrics(y_true, y_pred)
    log.info(
        "%s — MAE %.2f yr  RMSE %.2f yr  R %.3f  R² %.3f",
        split.upper(), m["mae"], m["rmse"], m["r"], m["r2"],
    )

    # Per-site
    for site in np.unique(sites):
        mask = sites == site
        sm = compute_metrics(y_true[mask], y_pred[mask])
        log.info("  %-6s  n=%3d  MAE %.2f  RMSE %.2f", site, mask.sum(), sm["mae"], sm["rmse"])

    # Per-sex
    for sex in np.unique(sexes):
        mask = sexes == sex
        sm = compute_metrics(y_true[mask], y_pred[mask])
        log.info("  sex=%s    n=%3d  MAE %.2f  RMSE %.2f", sex, mask.sum(), sm["mae"], sm["rmse"])

    return m


def save_predictions(
    path: Path,
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sexes: np.ndarray,
    sites: np.ndarray,
) -> None:
    pd.DataFrame({
        "subject_id": subject_ids,
        "age_true":   y_true,
        "age_pred":   y_pred,
        "error":      y_pred - y_true,
        "abs_error":  np.abs(y_pred - y_true),
        "sex":        sexes,
        "site":       sites,
    }).to_csv(path, index=False)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> list[dict]:
    model.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    Xt  = torch.tensor(X_train, device=device)
    yt  = torch.tensor(y_train, device=device)
    Xv  = torch.tensor(X_val,   device=device)
    yv  = torch.tensor(y_val,   device=device)

    n = len(Xt)
    history = []
    best_val_mae = float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss = 0.0
        for start in range(0, n, batch_size):
            idx  = perm[start: start + batch_size]
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(idx)
        train_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred    = model(Xv).cpu().numpy()
            val_mae     = float(np.abs(val_pred - y_val).mean())
            val_loss    = float(loss_fn(torch.tensor(val_pred), yv.cpu()).item())

        scheduler.step(val_mae)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_mae": val_mae,
        })

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "epoch %3d  train_mse %.3f  val_mse %.3f  val_mae %.2f yr  lr %.2e",
                epoch, train_loss, val_loss, val_mae,
                opt.param_groups[0]["lr"],
            )

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info("Early stopping at epoch %d (best val MAE %.2f yr)", epoch, best_val_mae)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   default="both", choices=["gpa", "edma", "both"])
    ap.add_argument("--hidden",     nargs="+", type=int, default=[256, 128, 64],
                    help="Hidden layer sizes (e.g. --hidden 256 128 64)")
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--epochs",     type=int,   default=300)
    ap.add_argument("--patience",   type=int,   default=30,
                    help="Early-stopping patience (epochs with no val MAE improvement)")
    ap.add_argument("--device",     default="auto",
                    help="cpu | mps | cuda | auto (auto picks mps > cuda > cpu)")
    ap.add_argument("--run-name",   default=None,
                    help="Output subdirectory name. Defaults to features_hiddenH1-H2-…")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("Using device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    run_name = args.run_name or (
        f"{args.features}_" + "-".join(str(h) for h in args.hidden)
    )
    out_dir = _MDL_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_d = load_split("train", args.features)
    val_d   = load_split("val",   args.features)
    test_d  = load_split("test",  args.features)

    log.info(
        "Features: %s  |  dims: %d  |  train %d  val %d  test %d",
        args.features, train_d["X"].shape[1],
        len(train_d["y"]), len(val_d["y"]), len(test_d["y"]),
    )
    log.info(
        "Age — train %.1f±%.1f  val %.1f±%.1f  test %.1f±%.1f",
        train_d["y"].mean(), train_d["y"].std(),
        val_d["y"].mean(),   val_d["y"].std(),
        test_d["y"].mean(),  test_d["y"].std(),
    )

    # ── Normalise features (fit on train only) ────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_d["X"])
    X_val   = scaler.transform(val_d["X"])
    X_test  = scaler.transform(test_d["X"])

    joblib.dump(scaler, str(out_dir / "scaler.joblib"))
    log.info("Scaler saved.")

    # ── Build model ───────────────────────────────────────────────────────────
    in_dim = X_train.shape[1]
    model  = MLP(in_dim, args.hidden, args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("MLP: %d → %s → 1  (%d params)", in_dim, args.hidden, n_params)

    # Save config
    cfg = {
        "features":   args.features,
        "in_dim":     in_dim,
        "hidden":     args.hidden,
        "dropout":    args.dropout,
        "lr":         args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience":   args.patience,
        "seed":       args.seed,
        "n_params":   n_params,
        "n_train":    len(train_d["y"]),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    history = train(
        model, X_train, train_d["y"], X_val, val_d["y"],
        lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.epochs, patience=args.patience,
        device=device,
    )
    log.info("Training finished in %.1f s", time.time() - t0)

    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    log.info("Model saved → %s", out_dir / "model.pt")

    pd.DataFrame(history).to_csv(str(out_dir / "history.csv"), index=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    model.to("cpu")

    results = {}
    for split, X, d in [
        ("train", X_train, train_d),
        ("val",   X_val,   val_d),
        ("test",  X_test,  test_d),
    ]:
        with torch.no_grad():
            y_pred = model(torch.tensor(X)).numpy()
        y_true = d["y"]

        log.info("")
        m = report_metrics(split, y_true, y_pred, d["sexes"], d["sites"])
        results[split] = m

        save_predictions(
            out_dir / f"{split}_predictions.csv",
            d["subject_ids"], y_true, y_pred, d["sexes"], d["sites"],
        )

    # Final summary
    log.info("")
    log.info("=" * 55)
    log.info("FINAL RESULTS  (%s features, run: %s)", args.features, run_name)
    log.info("=" * 55)
    for split in ("train", "val", "test"):
        m = results[split]
        log.info("  %-5s  MAE %.2f yr  RMSE %.2f yr  R² %.3f",
                 split, m["mae"], m["rmse"], m["r2"])
    log.info("=" * 55)
    log.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
