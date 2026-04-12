"""
Brain age inference wrappers.

Supported models:
  1. SynthSeg  — FreeSurfer 7.4+ built-in segmentation → volumetric features
                 → ridge regression age predictor (trained on IXI here)
  2. SFCN      — Peng et al. 2021, pretrained on UK Biobank (PAC 2019 winner)
                 Pretrained weights: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Both functions return a predicted brain age (years, float).
"""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
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

def run_synthstrip(
    nifti_path: str | Path,
    out_path: str | Path,
    mask_path: str | Path | None = None,
    command: str = "mri_synthstrip",
) -> Path:
    """
    Run FreeSurfer's SynthStrip brain extraction on a single scan.

    Parameters
    ----------
    nifti_path : input T1 (.nii / .nii.gz / .mgz)
    out_path   : path to the skull-stripped output NIfTI
    mask_path  : optional path to save the brain mask
    command    : SynthStrip executable name on PATH
    """
    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which(command) is None:
        raise RuntimeError(
            f"{command} is not on PATH. Install FreeSurfer/SynthStrip or set the command in local config."
        )

    cmd = [command, "-i", str(nifti_path), "-o", str(out_path)]
    if mask_path is not None:
        mask_path = Path(mask_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-m", str(mask_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{command} failed for {nifti_path}:\n{result.stderr}")

    return out_path


def run_deepbet(
    nifti_path: str | Path,
    out_path: str | Path,
    mask_path: str | Path | None = None,
) -> Path:
    """Brain extraction via deepbet (Fisch et al. 2024) — pure pip, no FreeSurfer."""
    from deepbet import run_bet

    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_p = (
        Path(mask_path)
        if mask_path
        else out_path.parent / out_path.name.replace(".nii.gz", "_mask.nii.gz")
    )
    run_bet(
        input_path=str(nifti_path),
        brain_path=str(out_path),
        mask_path=str(mask_p),
        tiv_path=None,
        threshold=0.5,
        n_dilate=0,
    )
    return out_path


def prepare_sfcn_input(
    nifti_path: str | Path,
    out_path: str | Path,
    skullstrip: bool = True,
    skullstrip_command: str = "mri_synthstrip",
    keep_skullstripped: bool = False,
) -> Path:
    """
    Prepare a T1 volume for SFCN inference.

    Pipeline:
      1. optional skull stripping via SynthStrip
      2. RAS reorientation
      3. 1 mm isotropic conforming
      4. centered crop/pad to [160, 192, 160]
    """
    from .utils import conform_1mm, load_vol, reorient_to_ras

    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_for_geometry = nifti_path
    if "_sfcn_input" in out_path.name:
        skull_name = out_path.name.replace("_sfcn_input", "_brain")
    else:
        skull_name = f"{out_path.stem}_brain.nii.gz"
    skull_path = out_path.with_name(skull_name)

    if skullstrip:
        if skullstrip_command == "deepbet":
            input_for_geometry = run_deepbet(
                nifti_path=nifti_path,
                out_path=skull_path,
            )
        else:
            input_for_geometry = run_synthstrip(
                nifti_path=nifti_path,
                out_path=skull_path,
                command=skullstrip_command,
            )

    img = conform_1mm(reorient_to_ras(load_vol(input_for_geometry)))
    data = img.get_fdata(dtype=np.float32)
    cropped = _crop_center(data, (160, 192, 160))
    prepared = nib.Nifti1Image(cropped.astype(np.float32), img.affine, img.header.copy())
    nib.save(prepared, str(out_path))

    if skullstrip and not keep_skullstripped and skull_path.exists():
        skull_path.unlink()

    return out_path


def build_sfcn_age_bins(
    start: float = 42.0,
    step: float = 1.0,
    count: int = 40,
) -> np.ndarray:
    """Build the age-bin centers used to decode SFCN outputs."""
    return np.arange(start, start + count * step, step, dtype=np.float32)


def decode_sfcn_output(
    model_output,
    age_bins: np.ndarray | None = None,
) -> float:
    """Decode SFCN log-probabilities into an expected age value."""
    import torch

    if isinstance(model_output, (list, tuple)):
        if not model_output:
            raise ValueError("SFCN model returned an empty output container.")
        model_output = model_output[0]

    if model_output.ndim > 2:
        model_output = model_output.reshape(model_output.shape[0], model_output.shape[1], -1).mean(dim=-1)
    if model_output.ndim == 1:
        model_output = model_output.unsqueeze(0)

    probs = torch.softmax(model_output, dim=1).squeeze(0).detach().cpu().numpy()
    bins = build_sfcn_age_bins() if age_bins is None else np.asarray(age_bins, dtype=np.float32)
    if probs.shape[0] != bins.shape[0]:
        raise ValueError(
            f"SFCN output dimension {probs.shape[0]} does not match configured age bins {bins.shape[0]}."
        )

    return float(np.sum(probs * bins))


def _load_sfcn_class(model_dir: str | Path):
    model_dir = Path(model_dir)
    candidate_paths = [
        model_dir / "sfcn.py",
        model_dir / "dp_model" / "model_files" / "sfcn.py",
        model_dir / "brain_age" / "sfcn.py",
    ]

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_sfcn_repo_module", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "SFCN"):
            return module.SFCN

    searched = ", ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(f"Could not locate sfcn.py under {model_dir}. Searched: {searched}")


def _resolve_sfcn_weight_path(
    model_dir: str | Path,
    weight_path: str | Path | None = None,
) -> Path:
    if weight_path is not None:
        resolved = Path(weight_path)
        if not resolved.exists():
            raise FileNotFoundError(f"SFCN weight file not found: {resolved}")
        return resolved

    model_dir = Path(model_dir)
    candidates = sorted(model_dir.rglob("*.p")) + sorted(model_dir.rglob("*.pth")) + sorted(model_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No SFCN weight file (*.p / *.pth / *.pt) found under {model_dir}")

    preferred = [p for p in candidates if "best_mae" in p.name.lower()]
    if preferred:
        return preferred[0]
    return candidates[0]


def _normalize_sfcn_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Unexpected SFCN checkpoint format; expected a state_dict-like mapping.")

    normalized = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        normalized[clean_key] = value
    return normalized


def predict_sfcn(
    nifti_path: str | Path,
    model_dir: str | Path,
    device: str = "cpu",
    weight_path: str | Path | None = None,
    age_bin_start: float = 42.0,
    age_bin_step: float = 1.0,
    age_bin_count: int = 40,
) -> float:
    """
    Predict brain age using the SFCN pretrained model (Peng et al. 2021).

    Pretrained weights must be downloaded separately:
      https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

    Parameters
    ----------
    nifti_path : preprocessed T1 MRI (.nii / .nii.gz / .mgz), ideally skull-stripped
    model_dir  : directory containing the official SFCN repo
    device     : 'cpu' or 'cuda'
    weight_path: optional explicit path to the official pretrained weight file
    age_bin_*  : age-bin decoding parameters; keep configurable until verified in runtime

    Returns
    -------
    Predicted brain age (years, float).
    """
    import torch

    model_dir = Path(model_dir)
    resolved_weight_path = _resolve_sfcn_weight_path(model_dir, weight_path)
    SFCN = _load_sfcn_class(model_dir)
    model = SFCN()
    state = torch.load(str(resolved_weight_path), map_location="cpu")
    model.load_state_dict(_normalize_sfcn_state_dict(state), strict=True)
    model.eval()
    model = model.to(device)

    from .utils import load_vol

    img = load_vol(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SFCN expects a 3D input volume, got shape {data.shape} for {nifti_path}")
    data = _crop_center(data, (160, 192, 160))
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    age_bins = build_sfcn_age_bins(
        start=age_bin_start,
        step=age_bin_step,
        count=age_bin_count,
    )
    return decode_sfcn_output(output, age_bins=age_bins)


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
