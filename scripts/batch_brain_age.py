#!/usr/bin/env python3
"""
Batch brain age inference using SynthSeg (FreeSurfer 7.4+).

Usage:
    python scripts/batch_brain_age.py <scan_dir> <out_dir> [--model models/synthseg_age.joblib]

Outputs one CSV per scan in out_dir/, plus a summary CSV: out_dir/brain_ages.csv
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch SynthSeg brain age inference")
    parser.add_argument("scan_dir", help="Directory with .nii/.nii.gz/.mgz files")
    parser.add_argument("out_dir", help="Output directory for SynthSeg CSVs")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained sklearn age regressor (.joblib). "
             "If omitted, volumetric CSVs are produced but age is not predicted.",
    )
    parser.add_argument(
        "--glob", default="**/*.nii.gz,**/*.nii,**/*.mgz",
        help="Glob patterns (comma-separated)"
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.brain_age import run_synthseg, synthseg_to_brain_age

    scan_dir = Path(args.scan_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.glob.split(",")]
    scans = sorted({s for p in patterns for s in scan_dir.glob(p)})

    if not scans:
        log.error(f"No scans found in {scan_dir}")
        return

    log.info(f"Found {len(scans)} scans")

    records = []
    for scan in tqdm(scans, unit="scan"):
        stem = scan.name.replace(".nii.gz", "").replace(".nii", "").replace(".mgz", "")
        try:
            vol_csv = run_synthseg(scan, out_dir)
            brain_age = synthseg_to_brain_age(vol_csv, args.model)
            records.append({"subject": stem, "brain_age": brain_age, "vol_csv": str(vol_csv)})
        except Exception as e:
            log.warning(f"Failed {scan.name}: {e}")
            records.append({"subject": stem, "brain_age": float("nan"), "vol_csv": ""})

    summary = pd.DataFrame(records)
    summary_path = out_dir / "brain_ages.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved summary to {summary_path}")
    log.info(f"Subjects processed: {len(records)}  NaN: {summary['brain_age'].isna().sum()}")


if __name__ == "__main__":
    main()
