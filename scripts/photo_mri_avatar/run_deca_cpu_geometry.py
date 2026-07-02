"""Run DECA coarse FLAME geometry on CPU without CUDA rendering.

The official DECA demo initializes a rasterizer even when the main output we
need is a surface. On Windows CPU-only machines that path fails because the
standard rasterizer JIT-compiles CUDA sources. This runner keeps only the
geometry contract needed for photo-to-MRI evaluation:

image crop -> DECA E_flame encoder -> FLAME vertices -> OBJ + NPZ sidecar.

It requires the licensed FLAME model file and the DECA checkpoint to be present
in the external DECA checkout. It does not download or upload private photos.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _patch_legacy_deps() -> None:
    """Patch old DECA/chumpy expectations on modern Python/numpy."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    aliases: dict[str, Any] = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _iter_images(input_dir: Path | None, images: list[Path], pattern: str) -> list[Path]:
    found: list[Path] = []
    if input_dir is not None:
        for path in sorted(input_dir.glob(pattern)):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                found.append(path)
    found.extend(images)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def _decompose_code(code, model_cfg) -> dict[str, Any]:
    code_dict = {}
    start = 0
    for key in model_cfg.param_list:
        width = int(model_cfg.get("n_" + key))
        end = start + width
        value = code[:, start:end]
        if key == "light":
            value = value.reshape(value.shape[0], 9, 3)
        code_dict[key] = value
        start = end
    return code_dict


def _load_image_tensor(path: Path, image_size: int, device: str):
    import torch

    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("# DECA coarse FLAME geometry, CPU geometry-only export\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def _check_assets(deca_repo: Path) -> dict[str, Path]:
    data = deca_repo / "data"
    required = {
        "flame_model": data / "generic_model.pkl",
        "deca_model": data / "deca_model.tar",
        "landmark_embedding": data / "landmark_embedding.npy",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "DECA CPU geometry cannot run until these model assets exist:\n"
            f"{detail}\n\n"
            "FLAME generic_model.pkl must be downloaded by the user under the "
            "FLAME license. DECA deca_model.tar can be fetched from the DECA "
            "release link/GDrive id used by upstream fetch_data.sh."
        )
    return required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-repo", required=True, type=Path, help="Path to the external DECA checkout")
    parser.add_argument("--input-dir", type=Path, help="Directory of already-cropped face images")
    parser.add_argument("--pattern", default="*", help="Glob pattern inside --input-dir")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Single input image; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", default="case_a")
    parser.add_argument("--session", default="single_photo")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _patch_legacy_deps()

    deca_repo = args.deca_repo.resolve()
    if not deca_repo.exists():
        raise FileNotFoundError(deca_repo)
    _check_assets(deca_repo)
    sys.path.insert(0, str(deca_repo))

    import torch
    from decalib.models.FLAME import FLAME
    from decalib.models.encoders import ResnetEncoder
    from decalib.utils import util
    from decalib.utils.config import cfg
    from decalib.utils.rotation_converter import batch_euler2axis

    images = _iter_images(args.input_dir, args.image, args.pattern)
    if not images:
        raise ValueError("No input images found")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.model.clone()
    n_param = sum(int(model_cfg.get("n_" + key)) for key in model_cfg.param_list)
    encoder = ResnetEncoder(outsize=n_param).to(device)
    flame = FLAME(model_cfg).to(device)

    checkpoint = torch.load(cfg.pretrained_modelpath, map_location=device)
    util.copy_state_dict(encoder.state_dict(), checkpoint["E_flame"])
    encoder.eval()
    flame.eval()

    faces = flame.faces_tensor.detach().cpu().numpy().astype(np.int64)
    summary = []

    for image_path in images:
        stem = _safe_stem(image_path)
        output_base = f"{args.subject}_{args.session}_{stem}_deca_cpu"
        obj_path = args.output_dir / f"{output_base}.obj"
        npz_path = args.output_dir / f"{output_base}.npz"
        metadata_path = args.output_dir / f"{output_base}_metadata.json"

        image_tensor = _load_image_tensor(image_path, cfg.dataset.image_size, str(device))
        with torch.no_grad():
            parameters = encoder(image_tensor)
            codedict = _decompose_code(parameters, model_cfg)
            if model_cfg.jaw_type == "euler":
                posecode = codedict["pose"]
                euler_jaw_pose = posecode[:, 3:].clone()
                posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
                codedict["pose"] = posecode
            verts, landmarks2d, landmarks3d = flame(
                shape_params=codedict["shape"],
                expression_params=codedict["exp"],
                pose_params=codedict["pose"],
            )

        vertices = verts[0].detach().cpu().numpy()
        _write_obj(obj_path, vertices, faces)
        np.savez_compressed(
            npz_path,
            vertices=vertices,
            faces=faces,
            landmarks2d=landmarks2d[0].detach().cpu().numpy(),
            landmarks3d=landmarks3d[0].detach().cpu().numpy(),
            shape=codedict["shape"][0].detach().cpu().numpy(),
            expression=codedict["exp"][0].detach().cpu().numpy(),
            pose=codedict["pose"][0].detach().cpu().numpy(),
            camera=codedict["cam"][0].detach().cpu().numpy(),
        )
        metadata = {
            "method": "DECA CPU geometry-only",
            "source_image": str(image_path),
            "deca_repo": str(deca_repo),
            "output_obj": str(obj_path),
            "output_npz": str(npz_path),
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "coordinate_note": "FLAME canonical coordinates; no CUDA rendering, texture, or image-space rasterizer was used.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary.append(metadata | {"metadata": str(metadata_path)})

    summary_path = args.output_dir / f"{args.subject}_{args.session}_deca_cpu_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "items": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
