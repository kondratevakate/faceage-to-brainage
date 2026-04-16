"""
Build the IXI T1 metadata table and split into train/val/test.

Outputs (written to data/):
  ixi_metadata.csv   — full table: subject_id, site, filepath, age, sex
  ixi_train.csv      — 70% split
  ixi_val.csv        — 15% split
  ixi_test.csv       — 15% split

Usage:
  python scripts/prepare_metadata.py
  python scripts/prepare_metadata.py --t1-dir /path/to/T1s --xls /path/to/IXI.xls
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Defaults ──────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[1]
_T1_DIR  = _ROOT.parent / "data" / "IXI"
_XLS     = _T1_DIR / "IXI.xls"
_OUT_DIR = _ROOT / "data"

_FILENAME_RE = re.compile(
    r"^IXI(?P<id>\d+)-(?P<site>[^-]+)-\d+-T1\.nii\.gz$"
)


def parse_t1_files(t1_dir: Path) -> pd.DataFrame:
    """Glob all T1 NIfTI files and extract subject_id + site from filename."""
    rows = []
    for p in sorted(t1_dir.glob("IXI*-T1.nii.gz")):
        m = _FILENAME_RE.match(p.name)
        if m:
            rows.append({
                "subject_id": int(m.group("id")),
                "site":       m.group("site"),
                "filepath":   str(p),
            })
    return pd.DataFrame(rows)


def load_xls(xls_path: Path) -> pd.DataFrame:
    """Load IXI.xls, keep IXI_ID / AGE / SEX, rename."""
    df = pd.read_excel(str(xls_path), engine="xlrd")
    df = df.rename(columns={
        "IXI_ID":            "subject_id",
        "SEX_ID (1=m, 2=f)": "sex",   # 1=male, 2=female
        "AGE":                "age",
    })[["subject_id", "age", "sex"]]
    return df


def build_metadata(t1_dir: Path, xls_path: Path) -> pd.DataFrame:
    files = parse_t1_files(t1_dir)
    meta  = load_xls(xls_path)

    df = files.merge(meta, on="subject_id", how="left")

    # ── Quality checks ────────────────────────────────────────────────────────
    n_total      = len(df)
    n_missing_xls = df["age"].isnull().sum()   # T1 exists but not in spreadsheet
    n_missing_age = df["age"].isnull().sum()

    df = df.dropna(subset=["age"])
    df = df.drop_duplicates(subset=["subject_id"])

    n_usable = len(df)

    df["sex"] = df["sex"].map({1: "M", 2: "F"})
    df = df.sort_values("subject_id").reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"  IXI T1 metadata summary")
    print(f"{'='*55}")
    print(f"  T1 files found:          {n_total}")
    print(f"  Missing in spreadsheet:  {n_missing_xls}")
    print(f"  Dropped (no age):        {n_missing_age}")
    print(f"  Usable subjects:         {n_usable}")
    print()

    # Age distribution
    print(f"  Age (mean ± SD):   {df['age'].mean():.1f} ± {df['age'].std():.1f} yr")
    print(f"  Age range:         {df['age'].min():.1f} – {df['age'].max():.1f} yr")
    q = df["age"].quantile([0.25, 0.5, 0.75])
    print(f"  Age Q1/Median/Q3:  {q[0.25]:.1f} / {q[0.5]:.1f} / {q[0.75]:.1f} yr")
    print()

    # Sex balance
    sex_counts = df["sex"].value_counts()
    for s, n in sex_counts.items():
        print(f"  Sex {s}:  {n}  ({100*n/n_usable:.1f}%)")
    print()

    # Site balance
    site_counts = df["site"].value_counts()
    for site, n in site_counts.items():
        print(f"  Site {site:6s}: {n}  ({100*n/n_usable:.1f}%)")
    print()

    # Missing values in final table
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if mv.empty:
        print("  Missing values in final table: none")
    else:
        print(f"  Missing values:\n{mv}")
    print()

    # Duplicates
    dup = df.duplicated(subset=["subject_id"]).sum()
    print(f"  Duplicate subject_ids: {dup}")
    print(f"{'='*55}\n")

    return df


def split_data(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, ...]:
    """
    Stratified 70/15/15 split stratified by site × sex × age_decade.
    Falls back to unstratified if any stratum is too small.
    """
    df = df.copy()
    df["_stratum"] = (
        df["site"] + "_" + df["sex"] + "_" + (df["age"] // 10).astype(int).astype(str)
    )

    # Drop strata with fewer than 3 members (can't split into train/val/test)
    counts = df["_stratum"].value_counts()
    small  = counts[counts < 3].index
    mask_small = df["_stratum"].isin(small)
    df_main  = df[~mask_small]
    df_small = df[mask_small]

    if len(df_small):
        print(f"  Note: {len(df_small)} subjects in small strata assigned to train.")

    train, temp = train_test_split(
        df_main, test_size=0.30, stratify=df_main["_stratum"], random_state=random_state
    )

    # Second split: strata with only 1 member in temp can't be stratified → fold into train
    temp_counts = temp["_stratum"].value_counts()
    temp_small_strata = temp_counts[temp_counts < 2].index
    mask_temp_small = temp["_stratum"].isin(temp_small_strata)
    temp_main  = temp[~mask_temp_small]
    temp_small = temp[mask_temp_small]

    val, test = train_test_split(
        temp_main, test_size=0.50, stratify=temp_main["_stratum"], random_state=random_state
    )
    train = pd.concat([train, df_small, temp_small], ignore_index=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n = len(split)
        print(f"  {name:5s}: {n:4d}  age {split['age'].mean():.1f}±{split['age'].std():.1f}"
              f"  M/F {(split['sex']=='M').sum()}/{(split['sex']=='F').sum()}"
              f"  sites {dict(split['site'].value_counts())}")

    train = train.drop(columns=["_stratum"]).sort_values("subject_id").reset_index(drop=True)
    val   = val.drop(columns=["_stratum"]).sort_values("subject_id").reset_index(drop=True)
    test  = test.drop(columns=["_stratum"]).sort_values("subject_id").reset_index(drop=True)

    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir",  default=str(_T1_DIR))
    ap.add_argument("--xls",     default=str(_XLS))
    ap.add_argument("--out-dir", default=str(_OUT_DIR))
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--force",   action="store_true",
                    help="Overwrite existing split files. Without this flag the "
                         "script refuses to overwrite to prevent accidental reshuffling.")
    args = ap.parse_args()

    t1_dir  = Path(args.t1_dir)
    xls     = Path(args.xls)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Guard: refuse to overwrite existing splits unless --force is passed ───
    split_files = [out_dir / f"ixi_{s}.csv" for s in ("train", "val", "test")]
    existing    = [p for p in split_files if p.exists()]
    if existing and not args.force:
        print("Split files already exist and will NOT be overwritten:")
        for p in existing:
            print(f"  {p}")
        print("\nRe-run with --force to regenerate (this will reshuffle the split).")
        return

    df = build_metadata(t1_dir, xls)

    meta_path = out_dir / "ixi_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path}")

    print("\nSplit (70 / 15 / 15, stratified by site × sex × age-decade):")
    train, val, test = split_data(df, random_state=args.seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        p = out_dir / f"ixi_{name}.csv"
        split.to_csv(p, index=False)
        print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
