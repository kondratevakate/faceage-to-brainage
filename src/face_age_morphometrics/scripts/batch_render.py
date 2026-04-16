#!/usr/bin/env python3
"""
Batch-render frontal face PNGs from a directory of MRI scans.

Usage:
    python scripts/batch_render.py <scan_dir> <out_dir> [--level 40] [--workers 4]

Example:
    python scripts/batch_render.py data/simon/ results/simon_renders/
    python scripts/batch_render.py data/ixi/T1/ results/ixi_renders/
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _render_one(args):
    scan_path, out_dir, level = args
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.render import render_face

    stem = Path(scan_path).name.replace(".nii.gz", "").replace(".nii", "").replace(".mgz", "")
    out_png = Path(out_dir) / f"{stem}.png"
    if out_png.exists():
        return str(scan_path), "skipped"
    ok = render_face(scan_path, out_png, level=level)
    return str(scan_path), "ok" if ok else "failed"


def main():
    parser = argparse.ArgumentParser(description="Batch MRI → face render")
    parser.add_argument("scan_dir", help="Directory with .nii/.nii.gz/.mgz files")
    parser.add_argument("out_dir", help="Output directory for PNGs")
    parser.add_argument("--level", type=float, default=40.0, help="Isosurface threshold")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--glob", default="**/*.mgz,**/*.nii.gz,**/*.nii",
        help="Comma-separated glob patterns"
    )
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.glob.split(",")]
    scans = []
    for pat in patterns:
        scans.extend(scan_dir.glob(pat))
    scans = sorted(set(scans))

    if not scans:
        log.error(f"No scans found in {scan_dir} matching {args.glob}")
        return

    log.info(f"Found {len(scans)} scans → rendering to {out_dir}")

    tasks = [(str(s), str(out_dir), args.level) for s in scans]
    ok_count = failed_count = skipped_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_render_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="scan"):
            path, status = fut.result()
            if status == "ok":
                ok_count += 1
            elif status == "failed":
                failed_count += 1
                log.warning(f"Render failed: {path}")
            else:
                skipped_count += 1

    log.info(f"Done. OK={ok_count}  Failed={failed_count}  Skipped={skipped_count}")


if __name__ == "__main__":
    main()
