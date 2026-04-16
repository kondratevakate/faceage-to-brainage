"""
Extract head mesh → face mesh → landmarks → GPA/EDMA features for all IXI splits.

Stages
------
1. Per subject (checkpointed, resumable):
     T1 NIfTI → head PLY → face PLY (LSA) → landmarks .npy
2. GPA:
     Fit Procrustes + PCA on **train** landmarks only.
     Transform train / val / test using train mean shape + PCA (no leakage).
     Save gpa_mean_shape.npy and gpa_pca.joblib.
3. EDMA:
     Compute 190-distance vector per subject (no fitting required).
4. Assemble per-split .npz:
     subject_ids, ages, sexes, sites, gpa_scores, edma_distances,
     centroid_sizes, procrustes_distances

Outputs
-------
results/ixi_meshes/{IXI_ID}_head.ply          intermediate (large, ~5 MB each)
results/ixi_meshes/{IXI_ID}_face.ply          intermediate (~2 MB each)
results/ixi_meshes/{IXI_ID}_face.txt          raw MVCNN output
results/ixi_meshes/{IXI_ID}_landmarks.npy     (20, 3) float64

data/features/gpa_mean_shape.npy              (20, 3) train Procrustes mean
data/features/gpa_pca.joblib                  fitted sklearn PCA (train only)
data/features/train.npz
data/features/val.npz
data/features/test.npz

Usage
-----
  python scripts/extract_features.py
  python scripts/extract_features.py --device mps --predict-num 5 --splits train val test
  python scripts/extract_features.py --rerun-failed   # retry subjects that previously failed
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.spatial.distance
from sklearn.decomposition import PCA

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data"
_FEAT_DIR = _DATA_DIR / "features"
_MESH_DIR = _ROOT / "results" / "ixi_meshes"

sys.path.insert(0, str(_ROOT))
from src.head_extraction import extract_head_mesh, center_and_extract_face
from src.landmarks import detect_landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_N_LM = 20
_TRIU_I, _TRIU_J = np.triu_indices(_N_LM, k=1)   # 190 upper-triangle indices


# ── GPA helpers ───────────────────────────────────────────────────────────────

def _centroid_size(c: np.ndarray) -> float:
    return float(np.sqrt(np.sum((c - c.mean(axis=0)) ** 2)))


def _procrustes_rotate(c: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
    """Rotate c to minimise squared distance to mean_shape (SVD step)."""
    U, _, Vt = scipy.linalg.svd(mean_shape.T @ c)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = (U @ Vt).T
    return c @ R.T


def fit_gpa_pca(
    landmark_arrays: list[np.ndarray],
    n_components: int = 9,
) -> dict:
    """
    Generalised Procrustes Analysis + PCA on a set of landmark configurations.

    Iteratively removes translation, unit-centroid scale, and rotation
    (Rohlf & Slice 1990), then fits sklearn PCA on the flattened coordinates.

    Fit on **train** landmarks only to avoid data leakage.

    Returns dict with: mean_shape (L,3), pca (fitted PCA), pc_scores (N, n_comp),
    centroid_sizes (N,), procrustes_distances (N,).
    """
    configs = [np.array(a, dtype=np.float64) for a in landmark_arrays]

    centroid_sizes = np.array([_centroid_size(c) for c in configs])
    for i, c in enumerate(configs):
        c -= c.mean(axis=0)
        if centroid_sizes[i] > 0:
            c /= centroid_sizes[i]

    mean_shape = configs[0].copy()
    cs0 = _centroid_size(mean_shape)
    if cs0 > 0:
        mean_shape /= cs0

    for _ in range(100):
        for i, c in enumerate(configs):
            configs[i] = _procrustes_rotate(c, mean_shape)
        new_mean = np.mean(configs, axis=0)
        cs = _centroid_size(new_mean)
        if cs > 0:
            new_mean /= cs
        if np.sum((new_mean - mean_shape) ** 2) < 1e-12:
            break
        mean_shape = new_mean

    proc_dists = np.array([
        float(np.sqrt(np.sum((c - mean_shape) ** 2))) for c in configs
    ])

    flat = np.stack([c.ravel() for c in configs])          # (N, 3*L)
    n_comp = min(n_components, flat.shape[0] - 1, flat.shape[1])
    pca_model = PCA(n_components=max(n_comp, 1))
    pc_scores = pca_model.fit_transform(flat)

    return {
        "mean_shape":            mean_shape,
        "pca":                   pca_model,
        "pc_scores":             pc_scores,
        "centroid_sizes":        centroid_sizes,
        "procrustes_distances":  proc_dists,
    }


def transform_gpa_pca(
    landmark_arrays: list[np.ndarray],
    mean_shape: np.ndarray,
    pca_model: PCA,
) -> dict:
    """
    Align new landmark configurations to a pre-fitted GPA mean shape and
    project through a pre-fitted PCA.  No refitting — prevents data leakage.
    """
    configs = [np.array(a, dtype=np.float64) for a in landmark_arrays]
    centroid_sizes = np.array([_centroid_size(c) for c in configs])

    for i, c in enumerate(configs):
        c -= c.mean(axis=0)
        if centroid_sizes[i] > 0:
            c /= centroid_sizes[i]
        configs[i] = _procrustes_rotate(c, mean_shape)

    proc_dists = np.array([
        float(np.sqrt(np.sum((c - mean_shape) ** 2))) for c in configs
    ])

    flat = np.stack([c.ravel() for c in configs])
    pc_scores = pca_model.transform(flat)

    return {
        "pc_scores":             pc_scores,
        "centroid_sizes":        centroid_sizes,
        "procrustes_distances":  proc_dists,
    }


# ── EDMA helper ───────────────────────────────────────────────────────────────

def _edma_vec(lm: np.ndarray) -> np.ndarray:
    """190-element pairwise distance vector (upper triangle, L=20)."""
    dists = scipy.spatial.distance.pdist(lm, metric="euclidean")
    fm = scipy.spatial.distance.squareform(dists)
    return fm[_TRIU_I, _TRIU_J].astype(np.float64)


# ── Stage 1: per-subject mesh + landmark extraction ───────────────────────────

def _landmark_path(subject_id: int) -> Path:
    return _MESH_DIR / f"IXI{subject_id:03d}_landmarks.npy"


def process_subject(
    subject_id: int,
    t1_path: str,
    device: str,
    predict_num: int,
    rerun_failed: bool,
) -> np.ndarray | None:
    """
    Run the full pipeline for one subject.
    Returns (20, 3) landmark array or None on failure.
    Checkpointed: skips all steps whose output already exists.
    """
    sid    = f"IXI{subject_id:03d}"
    t1     = Path(t1_path)
    head   = _MESH_DIR / f"{sid}_head.ply"
    face   = _MESH_DIR / f"{sid}_face.ply"
    lm_npy = _landmark_path(subject_id)
    failed = _MESH_DIR / f"{sid}.failed"

    if lm_npy.exists():
        return np.load(str(lm_npy))

    if failed.exists() and not rerun_failed:
        log.debug("%s: skipping (previously failed)", sid)
        return None

    if failed.exists():
        failed.unlink()

    def _fail(reason: str) -> None:
        log.warning("%s: FAILED — %s", sid, reason)
        failed.touch()

    if not head.exists():
        log.info("%s: extracting head mesh …", sid)
        try:
            ok = extract_head_mesh(t1, head)
        except Exception as exc:
            _fail(f"head extraction: {exc}")
            return None
        if not ok:
            _fail("head extraction returned False")
            return None

    if not face.exists():
        log.info("%s: extracting face mesh …", sid)
        try:
            ok = center_and_extract_face(head, face)
        except Exception as exc:
            _fail(f"face extraction: {exc}")
            return None
        if not ok:
            _fail("face extraction returned False (empty mesh)")
            return None

    log.info("%s: detecting landmarks (device=%s, pn=%d) …", sid, device, predict_num)
    try:
        lm = detect_landmarks(face, _MESH_DIR, predict_num=predict_num, device=device)
    except Exception as exc:
        _fail(f"landmark detection: {exc}")
        return None

    if lm is None:
        _fail("landmark detection returned None")
        return None

    np.save(str(lm_npy), lm)
    log.info("%s: landmarks saved → %s", sid, lm_npy.name)
    return lm


# ── Stages 2-4: GPA / EDMA / assemble ────────────────────────────────────────

def assemble_features(
    split_dfs: dict[str, pd.DataFrame],
    n_gpa_components: int,
) -> None:
    """Fit GPA on train, transform all splits, compute EDMA, save .npz files."""
    _FEAT_DIR.mkdir(parents=True, exist_ok=True)

    split_landmarks: dict[str, list] = {}
    split_meta:      dict[str, list] = {}

    for split, df in split_dfs.items():
        lms, ids, ages, sexes, sites = [], [], [], [], []
        n_missing = 0
        for _, row in df.iterrows():
            p = _landmark_path(int(row["subject_id"]))
            if not p.exists():
                n_missing += 1
                continue
            lms.append(np.load(str(p)))
            ids.append(int(row["subject_id"]))
            ages.append(float(row["age"]))
            sexes.append(str(row["sex"]))
            sites.append(str(row["site"]))

        if n_missing:
            log.warning("%s: %d subjects have no landmarks — excluded.", split, n_missing)

        split_landmarks[split] = lms
        split_meta[split]      = {"ids": ids, "ages": ages, "sexes": sexes, "sites": sites}
        log.info("%s: %d subjects with landmarks.", split, len(lms))

    train_lms = split_landmarks["train"]
    if len(train_lms) == 0:
        log.error("No train landmarks found — cannot fit GPA.")
        return

    log.info("Fitting GPA+PCA on %d train subjects (n_components=%d) …",
             len(train_lms), n_gpa_components)
    train_result = fit_gpa_pca(train_lms, n_components=n_gpa_components)

    mean_shape = train_result["mean_shape"]
    pca_model  = train_result["pca"]

    np.save(str(_FEAT_DIR / "gpa_mean_shape.npy"), mean_shape)
    joblib.dump(pca_model, str(_FEAT_DIR / "gpa_pca.joblib"))
    log.info("Saved GPA mean shape → gpa_mean_shape.npy")
    log.info("Saved GPA PCA        → gpa_pca.joblib  (%d components, %.1f%% variance)",
             pca_model.n_components_,
             100 * pca_model.explained_variance_ratio_.sum())

    for split, lms in split_landmarks.items():
        if not lms:
            continue

        meta = split_meta[split]

        if split == "train":
            gpa_scores = train_result["pc_scores"]
            cs         = train_result["centroid_sizes"]
            proc_dists = train_result["procrustes_distances"]
        else:
            res        = transform_gpa_pca(lms, mean_shape, pca_model)
            gpa_scores = res["pc_scores"]
            cs         = res["centroid_sizes"]
            proc_dists = res["procrustes_distances"]

        edma_matrix = np.stack([_edma_vec(lm) for lm in lms])   # (N, 190)

        out_path = _FEAT_DIR / f"{split}.npz"
        np.savez_compressed(
            str(out_path),
            subject_ids          = np.array(meta["ids"]),
            ages                 = np.array(meta["ages"]),
            sexes                = np.array(meta["sexes"]),
            sites                = np.array(meta["sites"]),
            gpa_scores           = gpa_scores,
            edma_distances       = edma_matrix,
            centroid_sizes       = cs,
            procrustes_distances = proc_dists,
        )
        log.info("%s: saved %s  (gpa %s, edma %s)",
                 split, out_path.name, gpa_scores.shape, edma_matrix.shape)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits",       nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--device",       default="cpu", choices=["cpu", "mps", "cuda"],
                    help="Device for MVCNN landmark detection.")
    ap.add_argument("--predict-num",  type=int, default=5,
                    help="Number of MVCNN predictions to average per subject.")
    ap.add_argument("--n-gpa",        type=int, default=9,
                    help="Number of GPA PCA components to retain (default 9 ≈ 99%% variance).")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="Retry subjects that previously failed.")
    ap.add_argument("--features-only", action="store_true",
                    help="Skip mesh/landmark extraction; only assemble features "
                         "from existing landmark .npy files.")
    args = ap.parse_args()

    _MESH_DIR.mkdir(parents=True, exist_ok=True)
    _FEAT_DIR.mkdir(parents=True, exist_ok=True)

    split_dfs: dict[str, pd.DataFrame] = {}
    for split in args.splits:
        csv = _DATA_DIR / f"ixi_{split}.csv"
        if not csv.exists():
            log.error("Split CSV not found: %s  (run prepare_metadata.py first)", csv)
            sys.exit(1)
        split_dfs[split] = pd.read_csv(csv)
        log.info("Loaded %s: %d subjects", csv.name, len(split_dfs[split]))

    if not args.features_only:
        all_rows = pd.concat(split_dfs.values(), ignore_index=True)
        n_total  = len(all_rows)

        for i, (_, row) in enumerate(all_rows.iterrows(), 1):
            t0 = time.time()
            sid = int(row["subject_id"])
            log.info("[%d/%d] subject IXI%03d", i, n_total, sid)
            process_subject(
                subject_id   = sid,
                t1_path      = row["filepath"],
                device       = args.device,
                predict_num  = args.predict_num,
                rerun_failed = args.rerun_failed,
            )
            log.info("[%d/%d] done in %.1f s", i, n_total, time.time() - t0)

        n_done   = sum(1 for _, r in all_rows.iterrows()
                       if _landmark_path(int(r["subject_id"])).exists())
        n_failed = sum(1 for _, r in all_rows.iterrows()
                       if (_MESH_DIR / f"IXI{int(r['subject_id']):03d}.failed").exists())
        log.info("Extraction complete: %d succeeded, %d failed, %d total.",
                 n_done, n_failed, n_total)

    if "train" not in split_dfs:
        log.warning("'train' not in requested splits — skipping GPA fit and feature assembly.")
        return

    assemble_features(split_dfs, n_gpa_components=args.n_gpa)
    log.info("All done. Features saved to %s", _FEAT_DIR)


if __name__ == "__main__":
    main()
