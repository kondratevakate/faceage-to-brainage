"""
Extract head mesh → face mesh → landmarks for all SIMON sessions.

Reads from data/simon/manifest.csv, processes each scan through the same
head_extraction → landmark detection pipeline used for IXI.

Outputs (results/simon_meshes/)
--------------------------------
  ses-{NNN}_run-{M}_head.ply
  ses-{NNN}_run-{M}_face.ply
  ses-{NNN}_run-{M}_landmarks.npy    (20, 3) float64
  ses-{NNN}_run-{M}.failed           created on failure, skipped on re-run

Usage
-----
  python scripts/extract_simon_landmarks.py
  python scripts/extract_simon_landmarks.py --device mps --predict-num 5
  python scripts/extract_simon_landmarks.py --rerun-failed
  python scripts/extract_simon_landmarks.py --session 3       # single session
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT     = Path(__file__).resolve().parents[1]
_MESH_DIR = _ROOT / "results" / "simon_meshes"
_DATA_DIR = _ROOT.parent / "data" / "simon"

sys.path.insert(0, str(_ROOT))
from src.head_extraction import extract_head_mesh, center_and_extract_face
from src.landmarks import detect_landmarks
from src.utils import load_simon_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _sid(session_id: int, run: int) -> str:
    return f"ses-{session_id:03d}_run-{run}"


def _lm_path(session_id: int, run: int) -> Path:
    return _MESH_DIR / f"{_sid(session_id, run)}_landmarks.npy"


def process_session(
    session_id: int,
    run: int,
    t1_path: str,
    device: str,
    predict_num: int,
    rerun_failed: bool,
) -> np.ndarray | None:
    """
    Full pipeline for one scan. Returns (20, 3) landmarks or None on failure.
    Checkpointed: skips steps whose outputs already exist.
    """
    sid    = _sid(session_id, run)
    t1     = Path(t1_path)
    head   = _MESH_DIR / f"{sid}_head.ply"
    face   = _MESH_DIR / f"{sid}_face.ply"
    lm_npy = _lm_path(session_id, run)
    failed = _MESH_DIR / f"{sid}.failed"

    if lm_npy.exists():
        log.info("%s: landmarks already exist — skipping", sid)
        return np.load(str(lm_npy))

    if failed.exists() and not rerun_failed:
        log.info("%s: previously failed — skipping (use --rerun-failed to retry)", sid)
        return None

    if failed.exists():
        failed.unlink()

    def _fail(reason: str) -> None:
        log.warning("%s: FAILED — %s", sid, reason)
        failed.touch()

    # Head mesh
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

    # Face mesh
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

    # Landmarks
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simon-dir",    type=Path, default=_DATA_DIR,
                    help="Path to SIMON dataset directory (contains manifest.csv + images/).")
    ap.add_argument("--device",       default="cpu", choices=["cpu", "mps", "cuda"],
                    help="Device for MVCNN landmark detection.")
    ap.add_argument("--predict-num",  type=int, default=5,
                    help="Number of MVCNN predictions to average per scan.")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="Retry scans that previously failed.")
    ap.add_argument("--session",      type=int, default=None,
                    help="Process only this session_id (all runs). Useful for debugging.")
    args = ap.parse_args()

    _MESH_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_simon_metadata(args.simon_dir)
    if args.session is not None:
        sessions = sessions[sessions["session_id"] == args.session].reset_index(drop=True)
        if sessions.empty:
            log.error("No sessions found with session_id=%d", args.session)
            sys.exit(1)

    log.info("Processing %d scans from %s", len(sessions), args.simon_dir)

    n_done, n_fail, n_skip = 0, 0, 0

    for i, (_, row) in enumerate(sessions.iterrows(), 1):
        t0 = time.time()
        log.info("[%d/%d] session %d run %d", i, len(sessions),
                 int(row["session_id"]), int(row["run"]))
        lm = process_session(
            session_id   = int(row["session_id"]),
            run          = int(row["run"]),
            t1_path      = row["path"],
            device       = args.device,
            predict_num  = args.predict_num,
            rerun_failed = args.rerun_failed,
        )
        elapsed = time.time() - t0
        if lm is not None:
            n_done += 1
            log.info("[%d/%d] done in %.1f s", i, len(sessions), elapsed)
        elif (_MESH_DIR / f"{_sid(int(row['session_id']), int(row['run']))}.failed").exists():
            n_fail += 1
        else:
            n_skip += 1

    log.info("=" * 55)
    log.info("Done: %d succeeded | %d failed | %d skipped | %d total",
             n_done, n_fail, n_skip, len(sessions))
    log.info("Landmarks saved to: %s", _MESH_DIR)


if __name__ == "__main__":
    main()
