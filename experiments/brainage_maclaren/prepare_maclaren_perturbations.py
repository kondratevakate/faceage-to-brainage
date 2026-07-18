#!/usr/bin/env python3
"""Create the locked Maclaren run-20 NeuroFM perturbation screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy import ndimage


DEFAULT_PREPROCESSING_STATUS = Path(
    "/mnt/d/data/faceage-to-brainage/derivatives/hdbet/2.0.1/"
    "maclaren_ds000239/R1.0.1/preprocessing_status.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/d/data/faceage-to-brainage/derivatives/perturbations/1.0/"
    "maclaren_ds000239/R1.0.1"
)
PARTICIPANTS = ["sub-01", "sub-02", "sub-03"]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: float) -> str:
    return f"{value:.6g}".replace("-", "minus").replace(".", "p")


def scale_in_fixed_grid(data: np.ndarray, scale: float) -> np.ndarray:
    shape = np.asarray(data.shape, dtype=float)
    center = (shape - 1.0) / 2.0
    matrix = np.eye(3) / scale
    offset = center - matrix @ center
    return ndimage.affine_transform(
        data,
        matrix=matrix,
        offset=offset,
        output_shape=data.shape,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def rotate_in_fixed_grid(data: np.ndarray, axis: int, angle: float) -> np.ndarray:
    planes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    return ndimage.rotate(
        data,
        angle=angle,
        axes=planes[axis],
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    ).astype(np.float32, copy=False)


def perturbation_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "perturbation": "baseline",
            "family": "baseline",
            "level": "0",
            "axis": "",
            "unit": "",
        }
    ]
    for axis in range(3):
        for angle in (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0):
            specs.append(
                {
                    "perturbation": f"rotation_axis{axis}_{safe_float(angle)}deg",
                    "family": "rotation",
                    "level": f"{angle:.6g}",
                    "axis": str(axis),
                    "unit": "degree",
                }
            )
    for resolution in (0.8, 1.0, 1.2):
        specs.append(
            {
                "perturbation": f"resolution_{safe_float(resolution)}mm",
                "family": "resolution",
                "level": f"{resolution:.6g}",
                "axis": "isotropic",
                "unit": "mm",
            }
        )
    for scale in (0.95, 1.05):
        specs.append(
            {
                "perturbation": f"scale_{safe_float(scale)}",
                "family": "scale",
                "level": f"{scale:.6g}",
                "axis": "isotropic",
                "unit": "factor",
            }
        )
    return specs


def create_perturbation(
    image: nib.Nifti1Image,
    spec: dict[str, object],
) -> nib.Nifti1Image:
    family = str(spec["family"])
    data = image.get_fdata(dtype=np.float32)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    float_image = nib.Nifti1Image(data, image.affine, header)
    if family == "rotation":
        output = rotate_in_fixed_grid(data, int(spec["axis"]), float(spec["level"]))
        return nib.Nifti1Image(output, image.affine, header)
    if family == "scale":
        output = scale_in_fixed_grid(data, float(spec["level"]))
        return nib.Nifti1Image(output, image.affine, header)
    if family == "resolution":
        return resample_to_output(
            float_image,
            voxel_sizes=(float(spec["level"]),) * 3,
            order=3,
            mode="constant",
            cval=0.0,
        )
    raise ValueError(family)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessing-status", type=Path, default=DEFAULT_PREPROCESSING_STATUS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    status_rows = [row for row in read_csv(args.preprocessing_status) if row["status"] == "ok"]
    if len(status_rows) != 120:
        raise ValueError(f"Expected 120 successful HD-BET rows, found {len(status_rows)}")
    selected = [row for row in status_rows if int(row["run_index"]) == 20]
    selected.sort(key=lambda row: row["participant_id"])
    if [row["participant_id"] for row in selected] != PARTICIPANTS:
        raise ValueError("Locked run-20 selection is incomplete")

    output_dir = args.output_root / "nifti"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = perturbation_specs()
    rows: list[dict[str, str]] = []

    for source in selected:
        source_path = Path(source["input"])
        source_hash = sha256_file(source_path)
        if source_hash != source["output_sha256"]:
            raise ValueError(f"Skull-stripped source hash changed: {source_path}")
        image = nib.load(str(source_path))
        orientation = "".join(nib.aff2axcodes(image.affine))
        for spec in specs:
            if spec["family"] == "baseline":
                output_path = source_path
            else:
                output_path = output_dir / (
                    f"{source['participant_id']}_run-20_{spec['perturbation']}.nii.gz"
                )
                if args.overwrite or not output_path.exists():
                    perturbed = create_perturbation(image, spec)
                    nib.save(perturbed, str(output_path))
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            output_image = nib.load(str(output_path))
            data = np.asanyarray(output_image.dataobj)
            if data.ndim != 3 or not np.isfinite(data).all() or not np.any(data):
                raise ValueError(f"Invalid perturbation output: {output_path}")
            rows.append(
                {
                    "dataset_id": source["dataset_id"],
                    "release": source["release"],
                    "participant_id": source["participant_id"],
                    "run_index": source["run_index"],
                    "chronological_age_years": source["chronological_age_years"],
                    "reported_sex": source["reported_sex"],
                    "relative_path": source["relative_path"],
                    "base_input_sha256": source_hash,
                    "perturbation": str(spec["perturbation"]),
                    "perturbation_family": str(spec["family"]),
                    "perturbation_level": str(spec["level"]),
                    "perturbation_axis": str(spec["axis"]),
                    "perturbation_unit": str(spec["unit"]),
                    "input": str(output_path),
                    "input_sha256": sha256_file(output_path),
                    "shape": "x".join(str(value) for value in output_image.shape),
                    "voxel_size_mm": "x".join(
                        f"{float(value):.6g}"
                        for value in output_image.header.get_zooms()[:3]
                    ),
                    "orientation": "".join(nib.aff2axcodes(output_image.affine)),
                    "base_orientation": orientation,
                    "interpolation": "none" if spec["family"] == "baseline" else "cubic_order_3",
                    "claim_level": "numerical_robustness_probe_only",
                }
            )

    if len(rows) != 72:
        raise ValueError(f"Expected 72 perturbation rows, found {len(rows)}")
    manifest_path = args.output_root / "perturbation_inputs.csv"
    write_csv(manifest_path, rows)
    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "selection_rule": "run_index_equals_20_for_each_participant",
        "selected_participants": PARTICIPANTS,
        "n_selected_baselines": 3,
        "n_manifest_rows": len(rows),
        "levels": specs,
        "age_delta_engineering_margin_years": 2.0,
        "volume_delta_engineering_margin_fraction": 0.05,
        "equivalence_test_permitted": False,
        "reason": "Only three participants; report observed deltas without a TOST claim.",
    }
    (args.output_root / "perturbation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Locked perturbation manifest: {manifest_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
