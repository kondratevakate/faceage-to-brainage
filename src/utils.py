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
    Build a session table from SIMON filenames.

    SIMON filename convention (FreeSurfer outputs):
        simon_freesurfer<version>_ses-<NNN>_mri_orig.mgz

    Returns DataFrame with columns:
        filename, session_id, fs_version, path
    """
    sessions_dir = Path(sessions_dir)
    records = []
    for f in sorted(sessions_dir.glob("simon_freesurfer*_ses-*_mri_orig.mgz")):
        parts = f.stem.split("_")
        # parts: ['simon', 'freesurferX', 'ses-NNN', 'mri', 'orig']
        fs_version = parts[1].replace("freesurfer", "")
        ses_id = parts[2].replace("ses-", "")
        records.append(
            {
                "filename": f.name,
                "session_id": int(ses_id),
                "fs_version": fs_version,
                "path": str(f),
            }
        )
    return pd.DataFrame(records).sort_values("session_id").reset_index(drop=True)
