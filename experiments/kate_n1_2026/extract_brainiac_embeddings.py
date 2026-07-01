#!/usr/bin/env python3
"""Extract BrainIAC 768-d embeddings from preprocessed NIfTI volumes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_images(image_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(image_dir.glob("*.nii.gz")):
        if "mask" in path.name.lower():
            continue
        scan_id = path.name.removesuffix(".nii.gz").removesuffix("_0000")
        rows.append(
            {
                "scan_id": scan_id,
                "session": "",
                "modality_hint": "",
                "analysis_role": "brainiac_preprocessed",
                "path": str(path),
                "source_relative_path": "",
                "source_sha256": "",
                "notes": "",
            }
        )
    return rows


def load_volume_tensor(path: Path, image_size: tuple[int, int, int]) -> torch.Tensor:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.nan_to_num(np.squeeze(data), nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI for {path}, got shape {data.shape}")

    nonzero = data[data != 0]
    if nonzero.size:
        mean = float(nonzero.mean())
        std = float(nonzero.std())
    else:
        mean = float(data.mean())
        std = float(data.std())
    if std < 1e-6:
        std = 1.0
    data = (data - mean) / std

    tensor = torch.from_numpy(data.astype(np.float32))[None, None]
    tensor = F.interpolate(tensor, size=image_size, mode="trilinear", align_corners=False)
    return tensor


def load_brainiac_model(checkpoint: Path, brainiac_repo: Path | None, device: torch.device) -> torch.nn.Module:
    suffix = checkpoint.suffix.lower()
    if suffix == ".safetensors":
        from monai.networks.nets import ViT
        from safetensors.torch import load_file

        model = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=False,
        )
        weights = load_file(str(checkpoint))
        if any(key.startswith("backbone.") for key in weights):
            weights = {key.removeprefix("backbone."): value for key, value in weights.items() if key.startswith("backbone.")}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            print(f"Warning: unexpected keys in safetensors checkpoint: {len(unexpected)}", file=sys.stderr)
        if missing:
            print(f"Warning: missing keys in safetensors checkpoint: {len(missing)}", file=sys.stderr)
        return model.to(device).eval()

    if brainiac_repo is None:
        raise ValueError("--brainiac-repo is required for .ckpt BrainIAC checkpoints")
    sys.path.insert(0, str(brainiac_repo / "src"))
    from model import ViTBackboneNet

    return ViTBackboneNet(str(checkpoint)).to(device).eval()


def model_vector(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    output = model(batch)
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim == 3:
        output = output[:, 0, :]
    if output.ndim != 2:
        raise ValueError(f"Unexpected BrainIAC output shape: {tuple(output.shape)}")
    return output.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--brainiac-repo", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.input_manifest is None and args.image_dir is None:
        raise ValueError("Provide --input-manifest or --image-dir")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing BrainIAC checkpoint: {args.checkpoint}")

    rows = load_manifest(args.input_manifest) if args.input_manifest else list_images(args.image_dir)
    if not rows:
        raise ValueError("No BrainIAC inputs found")

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = load_brainiac_model(args.checkpoint, args.brainiac_repo, device)

    feature_rows = []
    with torch.no_grad():
        for row in rows:
            path = Path(row["path"])
            tensor = load_volume_tensor(path, (96, 96, 96)).to(device)
            vec = model_vector(model, tensor)[0]
            out = {
                "scan_id": row["scan_id"],
                "session": row.get("session", ""),
                "modality_hint": row.get("modality_hint", ""),
                "analysis_role": row.get("analysis_role", ""),
                "image_path": str(path),
                "image_sha256": sha256_file(path),
            }
            out.update({f"feature_{idx:04d}": float(value) for idx, value in enumerate(vec)})
            feature_rows.append(out)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    metadata = {
        "method": "BrainIAC",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "brainiac_repo": str(args.brainiac_repo) if args.brainiac_repo else "",
        "device": str(device),
        "n_images": len(feature_rows),
        "output_csv": str(args.output_csv),
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote BrainIAC embeddings: {args.output_csv}")


if __name__ == "__main__":
    main()
