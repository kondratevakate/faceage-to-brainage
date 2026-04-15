"""
Compute face-age gap vs brain-age gap correlation on paired IXI subjects.

Inputs (gitignored, must be present locally):
  papers/tables/age_predictions_multiview_complete.csv  — FaceAge predictions
  papers/tables/synthba_predictions.csv                 — SynthBA predictions

Output:
  papers/tables/gap_correlation.csv                     — per-subject merged table
  stdout                                                — summary statistics
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


FACE_CSV = Path("papers/tables/age_predictions_multiview_complete.csv")
BRAIN_CSV = Path("papers/tables/synthba_predictions.csv")
OUT_CSV = Path("papers/tables/gap_correlation.csv")


def load_face(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # subject_id like IXI013-HH-1212-T1 → integer 13
    df["ixi_num"] = df["subject_id"].str.extract(r"IXI(\d+)").astype(int)
    df["face_age_gap"] = df["predicted_age"] - df["true_age"]
    return df[["ixi_num", "subject_id", "true_age", "predicted_age", "face_age_gap"]]


def load_brain(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # subject_id column is the IXI integer directly
    df = df.rename(columns={"subject_id": "ixi_num", "predicted_age": "brain_predicted_age"})
    return df[["ixi_num", "chron_age", "brain_predicted_age", "brain_age_gap"]]


def main(face_csv: Path, brain_csv: Path, out_csv: Path) -> None:
    face = load_face(face_csv)
    brain = load_brain(brain_csv)

    merged = face.merge(brain, on="ixi_num", how="inner")
    n = len(merged)
    print(f"Overlapping subjects: {n}")

    r, p_r = stats.pearsonr(merged["face_age_gap"], merged["brain_age_gap"])
    rho, p_rho = stats.spearmanr(merged["face_age_gap"], merged["brain_age_gap"])

    print(f"\nFace-age gap vs Brain-age gap correlation (N={n})")
    print(f"  Pearson  r   = {r:.3f}  (p = {p_r:.4f})")
    print(f"  Spearman rho = {rho:.3f}  (p = {p_rho:.4f})")

    # Save per-subject table
    out_cols = [
        "ixi_num", "subject_id", "true_age",
        "predicted_age", "face_age_gap",
        "brain_predicted_age", "brain_age_gap",
    ]
    merged[out_cols].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Append summary row
    summary = pd.DataFrame([{
        "ixi_num": "SUMMARY",
        "subject_id": f"N={n}",
        "true_age": np.nan,
        "predicted_age": np.nan,
        "face_age_gap": np.nan,
        "brain_predicted_age": np.nan,
        "brain_age_gap": np.nan,
        "pearson_r": round(r, 4),
        "pearson_p": round(p_r, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_rho, 4),
    }])
    summary.to_csv(out_csv, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, default=FACE_CSV)
    parser.add_argument("--brain", type=Path, default=BRAIN_CSV)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    main(args.face, args.brain, args.out)
