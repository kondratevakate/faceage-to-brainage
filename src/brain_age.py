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
        input_paths=[str(nifti_path)],
        brain_paths=[str(out_path)],
        mask_paths=[str(mask_p)],
        tiv_paths=None,
        threshold=0.5,
        n_dilate=0,
    )
    return out_path


def n4_bias_correction(
    nifti_path: str | Path,
    out_path: str | Path,
) -> Path:
    """
    N4ITK bias field correction via SimpleITK.

    Removes slow intensity inhomogeneity artefacts before registration.
    Runs on CPU in ~10-30 s per scan.
    """
    import SimpleITK as sitk

    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrected = corrector.Execute(img)
    sitk.WriteImage(corrected, str(out_path))
    return out_path


def register_to_mni(
    nifti_path: str | Path,
    out_path: str | Path,
) -> Path:
    """
    Affine registration to MNI152 1mm template via SimpleITK.

    SFCN and SynthBA were both trained on MNI-registered T1 volumes.
    Uses mutual information metric + multi-scale optimisation.
    Runs on CPU in ~60-120 s per scan.
    """
    import tempfile

    import SimpleITK as sitk
    from nilearn.datasets import load_mni152_template

    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write MNI template to a temp file; clean up after registration
    mni_nib = load_mni152_template(resolution=1)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        mni_tmp = Path(f.name)
    try:
        nib.save(mni_nib, str(mni_tmp))
        fixed = sitk.ReadImage(str(mni_tmp), sitk.sitkFloat32)
    finally:
        mni_tmp.unlink(missing_ok=True)

    moving = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)

    # moment-based initial alignment
    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_tx, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)

    final_tx = reg.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving, fixed, final_tx,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    sitk.WriteImage(resampled, str(out_path))
    return out_path


def prepare_sfcn_input(
    nifti_path: str | Path,
    out_path: str | Path,
    skullstrip: bool = True,
    skullstrip_command: str = "deepbet",
    keep_skullstripped: bool = False,
    n4_correct: bool = True,
    register_mni: bool = True,
) -> Path:
    """
    Prepare a T1 volume for SFCN inference.

    Pipeline (Peng et al. 2021 / UK Biobank protocol):
      1. N4 bias field correction (optional, recommended)
      2. skull stripping via deepbet or synthstrip
      3. affine registration to MNI152 1mm (optional, recommended)
      4. RAS reorientation + 1 mm isotropic conforming
      5. centered crop/pad to [160, 192, 160]
    """
    from .utils import conform_1mm, load_vol, reorient_to_ras

    nifti_path = Path(nifti_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stem = out_path.name.replace("_sfcn_input.nii.gz", "").replace(".nii.gz", "")
    n4_path = out_path.with_name(f"{stem}_n4.nii.gz")
    skull_path = out_path.with_name(f"{stem}_brain.nii.gz")
    mni_path = out_path.with_name(f"{stem}_mni.nii.gz")

    intermediates = []

    # 1. N4 bias correction
    current = nifti_path
    if n4_correct:
        n4_bias_correction(current, n4_path)
        current = n4_path
        intermediates.append(n4_path)

    # 2. skull stripping
    if skullstrip:
        if skullstrip_command == "deepbet":
            run_deepbet(nifti_path=current, out_path=skull_path)
        else:
            run_synthstrip(nifti_path=current, out_path=skull_path, command=skullstrip_command)
        current = skull_path
        if not keep_skullstripped:
            intermediates.append(skull_path)

    # 3. MNI registration
    if register_mni:
        register_to_mni(current, mni_path)
        current = mni_path
        intermediates.append(mni_path)

    # 4-5. reorient + conform + crop
    img = conform_1mm(reorient_to_ras(load_vol(current)))
    data = img.get_fdata(dtype=np.float32)
    cropped = _crop_center(data, (160, 192, 160))
    prepared = nib.Nifti1Image(cropped.astype(np.float32), img.affine, img.header.copy())
    nib.save(prepared, str(out_path))

    for p in intermediates:
        if p.exists():
            p.unlink()

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


def _load_sfcn_model(
    model_dir: str | Path,
    device: str = "cpu",
    weight_path: str | Path | None = None,
):
    """Load SFCN model and weights once. Returns (model, device)."""
    import torch

    model_dir = Path(model_dir)
    resolved_weight_path = _resolve_sfcn_weight_path(model_dir, weight_path)
    SFCN = _load_sfcn_class(model_dir)
    model = SFCN()
    state = torch.load(str(resolved_weight_path), map_location="cpu")
    model.load_state_dict(_normalize_sfcn_state_dict(state), strict=True)
    model.eval()
    model = model.to(device)
    return model


def _predict_sfcn_with_model(
    model,
    nifti_path: str | Path,
    device: str,
    age_bin_start: float,
    age_bin_step: float,
    age_bin_count: int,
) -> float:
    """Run one SFCN forward pass with a pre-loaded model."""
    import torch

    from .utils import load_vol

    img = load_vol(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SFCN expects a 3D input volume, got shape {data.shape} for {nifti_path}")
    data = _crop_center(data, (160, 192, 160))
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    age_bins = build_sfcn_age_bins(start=age_bin_start, step=age_bin_step, count=age_bin_count)
    return decode_sfcn_output(output, age_bins=age_bins)


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
    age_bin_*  : age-bin decoding parameters

    Returns
    -------
    Predicted brain age (years, float).
    """
    model = _load_sfcn_model(model_dir, device=device, weight_path=weight_path)
    return _predict_sfcn_with_model(
        model, nifti_path, device, age_bin_start, age_bin_step, age_bin_count
    )


def predict_synthba(
    nifti_path: str | Path,
    device: str = "cpu",
    mr_weighting: str = "t1",
) -> float:
    """
    Predict brain age using SynthBA (Puglisi et al. 2024).

    SynthBA handles its own preprocessing (SynthSeg + MNI alignment).
    Pass raw T1 — do NOT use prepare_sfcn_input beforehand.

    Parameters
    ----------
    nifti_path   : raw T1 NIfTI (.nii / .nii.gz)
    device       : 'cpu' or 'cuda'
    mr_weighting : 't1', 't2', or 'flair'
    """
    from synthba import SynthBA

    sba = SynthBA(device=device)
    img = nib.load(str(nifti_path))
    result = sba.run(img, preprocess=True, mr_weighting=mr_weighting)
    return float(result)


def predict_synthba_tta(
    nifti_path: str | Path,
    device: str = "cpu",
    mr_weighting: str = "t1",
) -> dict:
    """
    SynthBA with minimal TTA: original + L-R flip.

    SynthBA is trained with domain randomization so it is already robust.
    Only L-R flip is applied (2 forward passes).
    """
    import tempfile

    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    flipped = nib.Nifti1Image(np.flip(data, axis=0).copy(), img.affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        flip_path = f.name
    nib.save(flipped, flip_path)

    p_orig = predict_synthba(nifti_path, device=device, mr_weighting=mr_weighting)
    p_flip = predict_synthba(flip_path, device=device, mr_weighting=mr_weighting)
    Path(flip_path).unlink(missing_ok=True)

    preds = [p_orig, p_flip]
    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "n_aug": 2,
    }


def predict_midi_brainage(
    nifti_path: str | Path,
    midi_dir: str | Path,
    device: str = "cpu",
    sequence: str = "t1",
    skull_strip: bool = True,
) -> float:
    """
    Predict brain age using MIDIconsortium BrainAge (Wood et al. 2024).

    Calls the official run_inference.py via subprocess.
    Pass raw T1 — MIDI handles its own preprocessing (reorient + 1.4mm + 130³).

    For T1, skull_strip=True is required (MIDI has no raw-T1 model).
    When skull_strip=True, MIDI runs HD-BET + ANTs MNI registration internally
    (requires antspyx, works on Colab Linux but not Windows WDAC).

    Parameters
    ----------
    nifti_path  : raw T1 NIfTI
    midi_dir    : path to cloned MIDIconsortium/BrainAge repo
    device      : 'cpu' or 'cuda' (maps to --gpu flag)
    sequence    : 't1' or 't2'
    skull_strip : enable MIDI's built-in skull stripping + MNI registration
    """
    import csv
    import re
    import uuid
    import tempfile

    nifti_path = Path(nifti_path)
    midi_dir = Path(midi_dir)

    # Patch pre_process.py for MONAI >= 1.0 (AddChannel removed)
    pp = midi_dir / "pre_process.py"
    if pp.exists():
        raw = pp.read_text(encoding="utf-8")
        if "AddChannel" in raw and "class AddChannel" not in raw:
            raw = re.sub(r"[ \t]*AddChannel,?[ \t]*\r?\n", "", raw)
            raw = re.sub(r"from monai\.transforms import AddChannel\r?\n", "", raw)
            shim = (
                "try:\n"
                "    from monai.transforms import AddChannel\n"
                "except ImportError:\n"
                "    class AddChannel:\n"
                "        def __call__(self, x):\n"
                "            import numpy as np\n"
                "            return np.expand_dims(x, 0)\n"
            )
            raw = shim + raw
            pp.write_text(raw, encoding="utf-8")

    # Use a unique project name so re-runs don't collide (MIDI raises on duplicate)
    project_name = f"midi_run_{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "input.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "file_name", "Age"])
            writer.writerow(["subj", str(nifti_path), ""])

        cmd = [
            "python", str(midi_dir / "run_inference.py"),
            "--csv_file", str(csv_path),
            "--project_name", project_name,
            "--sequence", sequence,
            "--ensemble",
        ]
        if skull_strip:
            cmd.append("--skull_strip")
        if device == "cuda":
            cmd.append("--gpu")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(midi_dir))

        if result.returncode != 0:
            raise RuntimeError(f"MIDIBrainAge failed:\n{result.stderr}\n{result.stdout}")

        # Without --return_metrics, output is written to ./{project_name}/brain_age_output.csv
        out_csv = midi_dir / project_name / "brain_age_output.csv"
        if not out_csv.exists():
            raise FileNotFoundError(
                f"MIDIBrainAge finished but output CSV not found at {out_csv}. "
                f"stdout:\n{result.stdout}"
            )

        df = pd.read_csv(out_csv)
        # MIDI output column is 'Predicted_age (years)'
        col = "Predicted_age (years)"
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in MIDI output. Got: {list(df.columns)}")
        return float(df[col].iloc[0])


def predict_sfcn_tta(
    nifti_path: str | Path,
    model_dir: str | Path,
    device: str = "cpu",
    weight_path: str | Path | None = None,
    age_bin_start: float = 42.0,
    age_bin_step: float = 1.0,
    age_bin_count: int = 40,
    n_shifts: int = 5,
) -> dict:
    """
    SFCN inference with test-time augmentation (TTA).

    Augmentations match SFCN training (Peng et al. 2021, Table 2):
      - L-R mirroring
      - voxel shifting ±n_shifts along each axis (zero-padded)

    Total: 1 + 1 + 6 = 8 forward passes. Model is loaded once.
    Returns mean and std of predicted ages across augmentations.
    """
    import tempfile

    import torch

    # Load model ONCE; reuse for all 8 augmented forward passes.
    model = _load_sfcn_model(model_dir, device=device, weight_path=weight_path)

    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    def _shift_zero_pad(arr: np.ndarray, shift: int, axis: int) -> np.ndarray:
        result = np.zeros_like(arr)
        if shift == 0:
            return arr.copy()
        if shift > 0:
            dst = [slice(None)] * 3; src = [slice(None)] * 3
            dst[axis] = slice(shift, None)
            src[axis] = slice(None, arr.shape[axis] - shift)
            result[tuple(dst)] = arr[tuple(src)]
        else:
            s = -shift
            dst = [slice(None)] * 3; src = [slice(None)] * 3
            dst[axis] = slice(None, arr.shape[axis] - s)
            src[axis] = slice(s, None)
            result[tuple(dst)] = arr[tuple(src)]
        return result

    # Build augmented volumes: original + LR flip + ±shift on each axis
    augmented = [data, np.flip(data, axis=0).copy()]
    for axis in range(3):
        for shift in (n_shifts, -n_shifts):
            augmented.append(_shift_zero_pad(data, shift, axis))

    age_bins = build_sfcn_age_bins(start=age_bin_start, step=age_bin_step, count=age_bin_count)
    preds = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, aug_data in enumerate(augmented):
            tmp_path = Path(tmp_dir) / f"aug_{i}.nii.gz"
            nib.save(nib.Nifti1Image(aug_data, img.affine), str(tmp_path))
            p = _predict_sfcn_with_model(
                model, tmp_path, device, age_bin_start, age_bin_step, age_bin_count
            )
            preds.append(p)

    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "n_aug": len(preds),
    }


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
