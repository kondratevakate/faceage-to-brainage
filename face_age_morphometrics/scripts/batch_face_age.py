#!/usr/bin/env python3
"""
Batch FaceAge inference on rendered face PNGs.

Usage:
    python scripts/batch_face_age.py <render_dir> <out_csv> --faceage vendor/FaceAge [--bypass-mtcnn]

Output: CSV with columns: filename, face_age, mtcnn_found, bypass_used
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch FaceAge inference on renders")
    parser.add_argument("render_dir", help="Directory with rendered face PNGs")
    parser.add_argument("out_csv", help="Output CSV path")
    parser.add_argument(
        "--faceage", default="vendor/FaceAge",
        help="Path to cloned FaceAge repository"
    )
    parser.add_argument(
        "--bypass-mtcnn", action="store_true",
        help="Skip MTCNN face detection (recommended for MRI renders)"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.face_age import predict_age_batch

    render_dir = Path(args.render_dir)
    pngs = sorted(render_dir.glob("*.png"))

    if not pngs:
        log.error(f"No PNG files found in {render_dir}")
        return

    log.info(f"Running FaceAge on {len(pngs)} images (bypass_mtcnn={args.bypass_mtcnn})")

    results = predict_age_batch(
        img_paths=pngs,
        faceage_root=args.faceage,
        bypass_mtcnn=args.bypass_mtcnn,
        device=args.device,
    )

    rows = [
        {
            "filename": Path(r["path"]).stem,
            "face_age": r["age"],
            "mtcnn_found": r["mtcnn_found"],
            "bypass_used": r["bypass_used"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)

    n_found = df["mtcnn_found"].sum()
    n_nan = df["face_age"].isna().sum()
    log.info(f"Done. MTCNN found face in {n_found}/{len(df)}. NaN ages: {n_nan}")
    log.info(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()
