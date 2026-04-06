"""
Brain age inference wrappers.

Supported models:
  1. SynthSeg  — FreeSurfer 7.4+ built-in segmentation → volumetric features
                 → ridge regression age predictor (trained on IXI here)
  2. SFCN      — Peng et al. 2021, pretrained on UK Biobank (PAC 2019 winner)
                 Pretrained weights: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Both functions return a predicted brain age (years, float).
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SynthSeg wrapper
# ---------------------------------------------------------------------------

def run_synthseg(
    nifti_path: str | Path,
    out_dir: str | Path,
    parc: bool = False,
    robust: bool = True,
) -> Path:
    """
    Run FreeSurfer mri_synthseg on a single scan.

    Parameters
    ----------
    nifti_path : input T1 (.nii / .nii.gz / .mgz)
    out_dir    : directory where output CSV and segmentation are written
    parc       : if True, also produce cortical parcellation
    robust     : use robust mode (slower but better cross-scanner)

    Returns
    -------
    Path to the output volumes CSV produced by SynthSeg.

    Raises
    ------
    RuntimeError if mri_synthseg is not on PATH (FreeSurfer not installed).
    """
    nifti_path = Path(nifti_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_out = out_dir / (nifti_path.stem.replace(".nii", "") + "_synthseg.nii.gz")
    vol_csv = out_dir / (nifti_path.stem.replace(".nii", "") + "_volumes.csv")

    cmd = [
        "mri_synthseg",
        "--i", str(nifti_path),
        "--o", str(seg_out),
        "--vol", str(vol_csv),
        "--threads", "4",
    ]
    if parc:
        cmd.append("--parc")
    if robust:
        cmd.append("--robust")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mri_synthseg failed for {nifti_path}:\n{result.stderr}"
        )
    return vol_csv


def synthseg_to_brain_age(
    vol_csv: str | Path,
    model_path: str | Path | None = None,
) -> float:
    """
    Predict brain age from SynthSeg volumetric output.

    If a trained ridge-regression model (sklearn joblib) is provided via
    model_path, it is used. Otherwise returns NaN — you need to train the
    regressor first using notebooks/03_ixi_main_experiment.ipynb.

    Parameters
    ----------
    vol_csv    : path to _volumes.csv written by mri_synthseg
    model_path : path to saved sklearn regressor (.joblib)

    Returns
    -------
    Predicted brain age (float), or NaN if model not available.
    """
    if model_path is None or not Path(model_path).exists():
        return float("nan")

    import joblib

    df = pd.read_csv(vol_csv)
    feature_cols = [c for c in df.columns if c not in ("subject", "Dice")]
    X = df[feature_cols].values.reshape(1, -1)

    model = joblib.load(str(model_path))
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# SFCN wrapper
# ---------------------------------------------------------------------------

def predict_sfcn(
    nifti_path: str | Path,
    model_dir: str | Path,
    device: str = "cpu",
) -> float:
    """
    Predict brain age using the SFCN pretrained model (Peng et al. 2021).

    Pretrained weights must be downloaded separately:
      https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

    Parameters
    ----------
    nifti_path : T1 MRI (.nii / .nii.gz / .mgz), should be skull-stripped
    model_dir  : directory containing sfcn.py and the .pth weight file
    device     : 'cpu' or 'cuda'

    Returns
    -------
    Predicted brain age (years, float).
    """
    import sys

    import torch

    from .utils import conform_1mm, load_vol, reorient_to_ras

    model_dir = Path(model_dir)
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    # Locate weight file
    pth_files = list(model_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth weight file found in {model_dir}")
    weight_path = pth_files[0]

    from sfcn import SFCN  # noqa: PLC0415

    model = SFCN()
    state = torch.load(str(weight_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    img = conform_1mm(reorient_to_ras(load_vol(nifti_path)))
    data = img.get_fdata(dtype=np.float32)

    # SFCN expects (1, 1, 160, 192, 160) — crop to standard size
    data = _crop_center(data, (160, 192, 160))
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    # SFCN output is a probability distribution over age bins (40 bins, 3-year width)
    # Age bins: 42, 45, 48, ..., 159 — decode as weighted sum
    age_bins = np.arange(42, 42 + 40 * 1, 1, dtype=np.float32)  # adjust to actual bins
    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    # Typical SFCN: bins from 42 to 82 years in 1-year steps (40 bins)
    predicted_age = float(np.sum(probs * age_bins[: len(probs)]))
    return predicted_age


def _crop_center(data: np.ndarray, target: tuple) -> np.ndarray:
    """Crop (or pad) 3D array to target shape, centered."""
    result = np.zeros(target, dtype=data.dtype)
    slices_in = []
    slices_out = []
    for d_in, d_out in zip(data.shape, target):
        start_out = max(0, (d_out - d_in) // 2)
        start_in = max(0, (d_in - d_out) // 2)
        length = min(d_in - start_in, d_out - start_out)
        slices_in.append(slice(start_in, start_in + length))
        slices_out.append(slice(start_out, start_out + length))
    result[tuple(slices_out)] = data[tuple(slices_in)]
    return result
