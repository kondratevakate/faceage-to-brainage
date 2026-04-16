"""
Shared utilities: volume loading, orientation, metadata parsing.
"""
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import pandas as pd


def load_vol(path: str | Path) -> nib.Nifti1Image:
    """Load .nii, .nii.gz, or .mgz and return a Nifti1Image."""
    path = Path(path)
    img = nib.load(str(path))
    # MGZ → Nifti1 transparently via nibabel
    if not isinstance(img, nib.Nifti1Image):
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
    return img


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient volume to RAS+ canonical orientation."""
    return nib.as_closest_canonical(img)


def conform_1mm(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Resample to 1 mm isotropic voxels (matches FreeSurfer convention)."""
    return nib_proc.conform(img, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0))


def load_ixi_metadata(xls_path: str | Path) -> pd.DataFrame:
    """
    Parse the IXI demographic spreadsheet (IXI.xls / IXI.xlsx).

    Returns DataFrame with columns:
        IXI_ID, SEX_ID (1=male, 2=female), AGE, ETHNIC_ID,
        MARITAL_ID, OCCUPATION_ID, QUALIFICATION_ID,
        IQ, VISIT_DATE, DATE_OF_BIRTH, SITE (Guys / HH / IOP)
    """
    df = pd.read_excel(xls_path)
    df.columns = [c.strip().upper() for c in df.columns]

    # IXI_ID is zero-padded 3-digit in filenames, e.g. IXI002
    df["IXI_ID"] = df["IXI_ID"].astype(int)

    # Derive site from IXI_ID ranges (documented in IXI dataset)
    # Guys: 002–314, HH: 316–571, IOP: 573–600 (approximate)
    def _site(ixi_id: int) -> str:
        if ixi_id <= 314:
            return "Guys"
        elif ixi_id <= 571:
            return "HH"
        else:
            return "IOP"

    df["SITE"] = df["IXI_ID"].apply(_site)
    return df


def load_simon_metadata(sessions_dir: str | Path) -> pd.DataFrame:
    """
    Build a session table from the SIMON manifest.csv.

    SIMON dataset layout:
        <sessions_dir>/manifest.csv   — metadata (session_id, age, scanner, …)
        <sessions_dir>/images/        — BIDS NIfTI files (sub-032633_ses-NNN_run-M_T1w.nii.gz)

    Returns DataFrame with columns:
        session_id, age, run, manufacturer, scanner_model, acquisition_date,
        institution, filename, path
    Only rows whose NIfTI file exists locally are included.
    """
    sessions_dir = Path(sessions_dir)
    manifest = pd.read_csv(sessions_dir / "manifest.csv")
    images_dir = sessions_dir / "images"

    records = []
    for _, row in manifest.iterrows():
        local_path = images_dir / row["t1_filename"]
        if not local_path.exists():
            continue
        records.append({
            "session_id":    int(row["session_id"]),
            "age":           float(row["age"]),
            "run":           int(row["run"]),
            "manufacturer":  str(row["manufacturer"]),
            "scanner_model": str(row["man_model_name"]),
            "acquisition_date": str(row["acquisition_date"]),
            "institution":   str(row["institution_name"]),
            "filename":      row["t1_filename"],
            "path":          str(local_path),
        })

    return pd.DataFrame(records).sort_values("session_id").reset_index(drop=True)
