"""
Benchmark multiple age regressors on face morphometric features.

Models
------
  ridge    — Ridge regression            (RidgeCV, alphas log-spaced)
  lasso    — Lasso regression            (LassoCV, 5-fold)
  elastic  — ElasticNet                  (ElasticNetCV, 5-fold)
  svr      — Support Vector Regression   (RBF kernel, GridSearchCV)
  gbm      — Gradient Boosting           (sklearn, GridSearchCV)
  mlp      — Small MLP                   (64→32, early stopping)

Hyperparameters are tuned by cross-validation on the TRAIN set only.
Val and test sets are never seen during tuning.

Outputs (data/models/benchmark/)
---------------------------------
  {model}_train/val/test_predictions.csv
  benchmark_results.csv      — comparison table (MAE / RMSE / R²)

Usage
-----
  python scripts/benchmark_regressors.py
  python scripts/benchmark_regressors.py --features edma --models ridge svr gbm
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parents[1]
_FEAT    = _ROOT / "data" / "features"
_OUT     = _ROOT / "data" / "models" / "benchmark"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str, feature_set: str) -> dict:
    d = np.load(str(_FEAT / f"{split}.npz"), allow_pickle=True)
    if feature_set == "gpa":
        X = d["gpa_scores"].astype(np.float64)
    elif feature_set == "edma":
        X = d["edma_distances"].astype(np.float64)
    else:
        X = np.concatenate([d["gpa_scores"], d["edma_distances"]], axis=1).astype(np.float64)
    return {
        "X": X, "y": d["ages"].astype(np.float64),
        "subject_ids": d["subject_ids"], "sexes": d["sexes"], "sites": d["sites"],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err   = y_pred - y_true
    mae   = float(np.abs(err).mean())
    rmse  = float(np.sqrt((err ** 2).mean()))
    ss_r  = (err ** 2).sum()
    ss_t  = ((y_true - y_true.mean()) ** 2).sum()
    r2    = float(1 - ss_r / ss_t)
    r, _  = pearsonr(y_true, y_pred)
    bias  = float(err.mean())
    return {"mae": mae, "rmse": rmse, "r2": r2, "r": float(r), "bias": bias}


def save_predictions(path: Path, d: dict, y_pred: np.ndarray) -> None:
    pd.DataFrame({
        "subject_id": d["subject_ids"],
        "age_true":   d["y"],
        "age_pred":   y_pred,
        "error":      y_pred - d["y"],
        "abs_error":  np.abs(y_pred - d["y"]),
        "sex":        d["sexes"],
        "site":       d["sites"],
    }).to_csv(path, index=False)


# ── Model definitions ─────────────────────────────────────────────────────────

def make_models() -> dict:
    return {
        "ridge": RidgeCV(
            alphas=np.logspace(-3, 4, 20),
            cv=5,
        ),
        "lasso": LassoCV(
            alphas=np.logspace(-3, 2, 30),
            cv=5, max_iter=10_000, random_state=42,
        ),
        "elastic": ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            alphas=np.logspace(-3, 2, 20),
            cv=5, max_iter=10_000, random_state=42,
        ),
        "svr": GridSearchCV(
            SVR(kernel="rbf"),
            param_grid={
                "C":       [0.1, 1, 10, 50, 100],
                "epsilon": [0.5, 1.0, 2.0, 5.0],
                "gamma":   ["scale"],
            },
            cv=5, scoring="neg_mean_absolute_error",
            # n_jobs=1: Accelerate framework (Apple Silicon BLAS) is already
            # multi-threaded internally; spawning parallel workers multiplies
            # memory arenas without real speedup on n=335.
            n_jobs=1, refit=True,
        ),
        "gbm": RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions={
                "n_estimators":  [100, 200, 300],
                "max_depth":     [2, 3, 4],
                "learning_rate": [0.05, 0.1],
                "subsample":     [0.8, 1.0],
            },
            n_iter=20, cv=5, scoring="neg_mean_absolute_error",
            # Same reasoning as SVR: sequential is safe and fast enough for
            # n=335. RandomizedSearchCV (n_iter=20) covers the space well
            # without the 180-fit overhead of the full grid.
            n_jobs=1, refit=True, random_state=42,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,                 # L2 regularisation
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="both", choices=["gpa", "edma", "both"])
    ap.add_argument("--models",   nargs="+",
                    default=["ridge", "lasso", "elastic", "svr", "gbm", "mlp"],
                    choices=["ridge", "lasso", "elastic", "svr", "gbm", "mlp"])
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)

    train_d = load_split("train", args.features)
    val_d   = load_split("val",   args.features)
    test_d  = load_split("test",  args.features)

    log.info("Features: %s  |  dims: %d  |  train=%d  val=%d  test=%d",
             args.features, train_d["X"].shape[1],
             len(train_d["y"]), len(val_d["y"]), len(test_d["y"]))

    # Scale features: fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_d["X"])
    X_val   = scaler.transform(val_d["X"])
    X_test  = scaler.transform(test_d["X"])
    joblib.dump(scaler, _OUT / "scaler.joblib")

    all_models = make_models()
    rows = []

    for name in args.models:
        model_path = _OUT / f"{name}_model.joblib"
        if model_path.exists():
            log.info("─" * 55)
            log.info("Skipping %s — already fitted (%s)", name, model_path)
            model = joblib.load(model_path)
            yp_train = model.predict(X_train)
            yp_val   = model.predict(X_val)
            yp_test  = model.predict(X_test)
            train_time = 0.0
        else:
            model = all_models[name]
            log.info("─" * 55)
            log.info("Training: %s", name)

            t0 = time.time()
            model.fit(X_train, train_d["y"])
            train_time = time.time() - t0
            joblib.dump(model, model_path)

            yp_train = model.predict(X_train)
            yp_val   = model.predict(X_val)
            yp_test  = model.predict(X_test)

        # Best hyperparameters (for grid-searched models)
        if hasattr(model, "best_params_"):
            log.info("  Best params: %s", model.best_params_)
        elif hasattr(model, "alpha_"):
            log.info("  Best alpha: %.4g", model.alpha_)
        if hasattr(model, "l1_ratio_"):
            log.info("  Best l1_ratio: %.2f", model.l1_ratio_)

        # Metrics
        m_tr = metrics(train_d["y"], yp_train)
        m_va = metrics(val_d["y"],   yp_val)
        m_te = metrics(test_d["y"],  yp_test)

        log.info("  Train  MAE=%.2f  RMSE=%.2f  R²=%.3f",
                 m_tr["mae"], m_tr["rmse"], m_tr["r2"])
        log.info("  Val    MAE=%.2f  RMSE=%.2f  R²=%.3f",
                 m_va["mae"], m_va["rmse"], m_va["r2"])
        log.info("  Test   MAE=%.2f  RMSE=%.2f  R²=%.3f  r=%.3f  bias=%+.2f",
                 m_te["mae"], m_te["rmse"], m_te["r2"], m_te["r"], m_te["bias"])

        # Per-site on test
        for site in np.unique(test_d["sites"]):
            mask = test_d["sites"] == site
            sm = metrics(test_d["y"][mask], yp_test[mask])
            log.info("    %-6s  n=%2d  MAE=%.2f  RMSE=%.2f",
                     site, mask.sum(), sm["mae"], sm["rmse"])

        # Per-sex on test
        for sex in np.unique(test_d["sexes"]):
            mask = test_d["sexes"] == sex
            sm = metrics(test_d["y"][mask], yp_test[mask])
            log.info("    sex=%s  n=%2d  MAE=%.2f  RMSE=%.2f",
                     sex, mask.sum(), sm["mae"], sm["rmse"])

        # Save predictions
        save_predictions(_OUT / f"{name}_train_predictions.csv", train_d, yp_train)
        save_predictions(_OUT / f"{name}_val_predictions.csv",   val_d,   yp_val)
        save_predictions(_OUT / f"{name}_test_predictions.csv",  test_d,  yp_test)

        rows.append({
            "model":      name,
            "features":   args.features,
            "train_mae":  round(m_tr["mae"],  2),
            "val_mae":    round(m_va["mae"],  2),
            "test_mae":   round(m_te["mae"],  2),
            "train_rmse": round(m_tr["rmse"], 2),
            "val_rmse":   round(m_va["rmse"], 2),
            "test_rmse":  round(m_te["rmse"], 2),
            "test_r2":    round(m_te["r2"],   3),
            "test_r":     round(m_te["r"],    3),
            "test_bias":  round(m_te["bias"], 2),
            "train_time_s": round(train_time, 1),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    # Merge with any previously computed results so a partial run still
    # produces a complete benchmark_results.csv for all finished models.
    results_path = _OUT / "benchmark_results.csv"
    if results_path.exists():
        prior = pd.read_csv(results_path)
        current_models = {r["model"] for r in rows}
        prior = prior[~prior["model"].isin(current_models)]
        results = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)
    else:
        results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)

    log.info("\n%s", "=" * 65)
    log.info("BENCHMARK SUMMARY  (features: %s)", args.features)
    log.info("=" * 65)
    log.info("%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s",
             "Model", "Tr-MAE", "Va-MAE", "Te-MAE",
             "Tr-RMSE", "Va-RMSE", "Te-RMSE", "Te-R²", "Time(s)")
    log.info("-" * 65)
    for _, r in results.sort_values("test_mae").iterrows():
        log.info("%-8s  %6.2f  %6.2f  %6.2f  %6.2f  %6.2f  %6.2f  %6.3f  %6.1f",
                 r.model, r.train_mae, r.val_mae, r.test_mae,
                 r.train_rmse, r.val_rmse, r.test_rmse, r.test_r2, r.train_time_s)
    log.info("=" * 65)
    log.info("Saved: %s", results_path)


if __name__ == "__main__":
    main()
