#!/usr/bin/env python3
"""Run local BrainIAC brain-age inference on a manifest without volume outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resized, ScaleIntensityd


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_volume(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    original_shape = tuple(int(x) for x in data.shape)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume after squeeze, got shape {original_shape}")
    return data, original_shape


def prepare_tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    resize_transform = Resized(keys=["image"], spatial_size=(128, 128, 128))
    scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
    sample = {"image": torch.tensor(data, dtype=torch.float32).unsqueeze(0)}
    sample = resize_transform(sample)
    sample = scale_transform(sample)
    tensor = sample["image"].unsqueeze(0).to(device)
    if tensor.ndim != 5:
        raise ValueError(f"Unexpected prepared tensor shape: {tuple(tensor.shape)}")
    return tensor


def load_model(space_dir: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(space_dir))
    from model import Backbone, Classifier, SingleScanModel

    model = SingleScanModel(Backbone(), Classifier(d_model=2048))
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--brainiac-space-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError(f"No rows found in manifest: {args.manifest}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.brainiac_space_dir, args.checkpoint, device)
    output_rows: list[dict[str, str | float]] = []
    started = time.time()

    with torch.inference_mode():
        for idx, row in enumerate(rows, start=1):
            path = Path(row["path"])
            out = dict(row)
            out.update(
                {
                    "status": "ok",
                    "error": "",
                    "original_shape": "",
                    "prepared_shape": "",
                    "input_min": "",
                    "input_max": "",
                    "input_nonzero_fraction": "",
                    "raw_model_output": "",
                    "predicted_age_years_if_months": "",
                    "predicted_age_years_if_raw_years": "",
                    "brain_age_delta_years_if_months": "",
                    "brain_age_delta_years_if_raw_years": "",
                }
            )
            try:
                if not path.exists():
                    raise FileNotFoundError(path)
                data, original_shape = load_volume(path)
                tensor = prepare_tensor(data, device)
                raw_output = float(model(tensor).detach().cpu().item())
                predicted_years_if_months = raw_output / 12.0
                chronological_age = parse_float(row.get("chronological_age_years", ""))

                out.update(
                    {
                        "original_shape": "x".join(str(x) for x in original_shape),
                        "prepared_shape": "x".join(str(x) for x in tensor.shape),
                        "input_min": float(np.nanmin(data)),
                        "input_max": float(np.nanmax(data)),
                        "input_nonzero_fraction": float(np.count_nonzero(data) / data.size),
                        "raw_model_output": raw_output,
                        "predicted_age_years_if_months": predicted_years_if_months,
                        "predicted_age_years_if_raw_years": raw_output,
                    }
                )
                if chronological_age is not None:
                    out["brain_age_delta_years_if_months"] = predicted_years_if_months - chronological_age
                    out["brain_age_delta_years_if_raw_years"] = raw_output - chronological_age
            except Exception as exc:  # keep batch robust and auditable
                out["status"] = "failed"
                out["error"] = repr(exc)
            output_rows.append(out)
            print(f"[{idx}/{len(rows)}] {row.get('scan_id', path.name)} {out['status']}", flush=True)

    fieldnames = list(output_rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "method": "BrainIAC Brainage Space local inference",
        "source_space": "https://huggingface.co/spaces/Divytak/BrainIAC-Brainage-V0",
        "brainiac_space_dir": str(args.brainiac_space_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "manifest": str(args.manifest),
        "output_csv": str(args.output_csv),
        "n_inputs": len(rows),
        "n_ok": sum(1 for row in output_rows if row["status"] == "ok"),
        "n_failed": sum(1 for row in output_rows if row["status"] != "ok"),
        "elapsed_seconds": time.time() - started,
        "preprocessing_applied_by_wrapper": "load volume, resize to 128x128x128, scale intensity to 0..1 only",
        "raw_output_unit_note": (
            "The Space app divides model output by 12 before displaying years, and the sample "
            "subpixar009 label 42.0 predicts raw output about 43.15. Results therefore retain "
            "raw_model_output and report predicted_age_years_if_months=raw/12. This is an "
            "exploratory research output, not a clinical or biological age claim."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {args.output_csv}")
    print(f"Wrote metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
