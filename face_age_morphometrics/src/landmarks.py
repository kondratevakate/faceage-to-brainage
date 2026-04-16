"""
Face PLY → 20 3D facial landmark coordinates (BioFace3D-20 MVCNN).

Uses the BioFace3D Multi-view Consensus CNN (Deep-MVLM, Paulsen et al.) with the
BioFace3D-20 model trained on 339 MRI-derived facial meshes.  Achieves MRE ≈ 1.61 mm
on IXI scans (Heredia-Lidón et al., CMPB 2025, Table 2).

Input:  face-only PLY from center_and_extract_face() in src/head_extraction.py
Output: (20, 3) numpy array of landmark XYZ coordinates in mesh space (RAS+ mm)

On first run the model weights (model_best.pth, ~276 MB) must be present at
  bioface3d/mvcnn/__configs/20Landmarks_25v_depth_geom/model_best.pth
They are NOT downloaded automatically (unlike the DTU-3D model); obtain them from
the BioFace3D Bitbucket repository.
"""
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Resource paths ────────────────────────────────────────────────────────────
_BIOFACE3D_ROOT = Path(__file__).resolve().parents[1] / "bioface3d"
_MVCNN_DIR      = _BIOFACE3D_ROOT / "mvcnn"
_DEFAULT_CONFIG = _MVCNN_DIR / "__configs" / "20Landmarks_25v_depth_geom" / "config.json"
_MVCNN_DRIVER   = Path(__file__).parent / "_mvcnn_driver.py"
_N_LANDMARKS    = 20


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_landmark_txt(txt_path: Path) -> np.ndarray | None:
    """
    Parse a BioFace3D landmark .txt file into a (20, 3) float array.
    Format: one landmark per line, space-separated x y z.
    Returns None if the file has fewer than _N_LANDMARKS lines.
    """
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    if len(lines) < _N_LANDMARKS:
        log.warning("Landmark file %s has only %d lines (expected %d).",
                    txt_path, len(lines), _N_LANDMARKS)
        return None
    coords = []
    for ln in lines[:_N_LANDMARKS]:
        parts = ln.split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(coords, dtype=np.float64)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_landmarks(
    face_mesh_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    predict_num: int = 5,
    device: str = "cpu",
) -> np.ndarray | None:
    """
    Detect 20 facial landmarks on a cropped face PLY mesh.

    Calls BioFace3D's Deep-MVLM via subprocess (src/_mvcnn_driver.py), which
    handles Apple Silicon MPS support and fixes the hardcoded Windows model path.

    Parameters
    ----------
    face_mesh_path : path to face-only PLY from center_and_extract_face()
                     (NOT the full head mesh)
    out_dir        : directory where <basename>.txt landmark file is written
    config_path    : MVCNN config JSON; defaults to 20Landmarks_25v_depth_geom/config.json
    predict_num    : number of independent predictions to average (paper default 10;
                     5 is faster and still accurate for single-subject use)
    device         : 'cpu' | 'mps' (Apple Silicon) | 'cuda'

    Returns
    -------
    (20, 3) float64 array of XYZ landmark coordinates in the face mesh's coordinate
    space (RAS+ mm after center_and_extract_face()), or None on failure.
    """
    face_mesh_path = Path(face_mesh_path)
    out_dir        = Path(out_dir)
    config_path    = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    out_dir.mkdir(parents=True, exist_ok=True)

    if not face_mesh_path.exists():
        raise FileNotFoundError(f"Face mesh not found: {face_mesh_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"MVCNN config not found: {config_path}")

    basename   = face_mesh_path.stem
    out_txt    = out_dir / (basename + ".txt")

    # Idempotent: skip if already done
    if out_txt.exists():
        log.info("Landmark file already exists, loading: %s", out_txt)
        return _read_landmark_txt(out_txt)

    cmd = [
        "python", str(_MVCNN_DRIVER),
        "-c", str(config_path),
        "-n", str(face_mesh_path),
        "-o", str(out_dir),
        "-of", "txt",
        "-pn", str(predict_num),
        "--device", device,
    ]

    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    log.info("Running BioFace3D-20 MVCNN (%d predictions, device=%s) on %s",
             predict_num, device, face_mesh_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        log.error("MVCNN subprocess failed for %s:\n%s", face_mesh_path, result.stderr)
        return None

    if not out_txt.exists():
        log.error("MVCNN finished but output file not found: %s", out_txt)
        return None

    return _read_landmark_txt(out_txt)
