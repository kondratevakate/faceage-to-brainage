"""Run MICA metric FLAME geometry on CPU.

The upstream MICA demo hardcodes CUDA in two places: model device selection and
InsightFace landmark detection. This runner keeps the same scientific output
contract while forcing CPU execution:

image crop -> InsightFace CPU alignment -> MICA -> metric FLAME mesh.

It requires the licensed FLAME2020 `generic_model.pkl`, MICA checkpoint, and
InsightFace model assets to be present locally. It does not upload images.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _patch_legacy_deps() -> None:
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


def _check_assets(mica_repo: Path, model_path: Path) -> None:
    flame = mica_repo / "data" / "FLAME2020"
    insight = Path.home() / ".insightface" / "models"
    required = {
        "mica_model": model_path,
        "flame_model": flame / "generic_model.pkl",
        "landmark_embedding": flame / "landmark_embedding.npy",
        "head_template": flame / "head_template.obj",
        "flame_masks": flame / "FLAME_masks" / "FLAME_masks.pkl",
        "insightface_antelopev2": insight / "antelopev2",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "MICA CPU geometry cannot run until these assets exist:\n"
            f"{detail}\n\n"
            "FLAME2020 generic_model.pkl must be downloaded by the user under "
            "the FLAME license."
        )


def _select_center_face(faces, image_shape):
    if not faces:
        return None
    h, w = image_shape[:2]
    center = np.array([w / 2.0, h / 2.0])

    def score(face):
        x1, y1, x2, y2 = face.bbox[:4]
        face_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        return float(np.linalg.norm(face_center - center))

    return sorted(faces, key=score)[0]


def _make_mica_tensors(image_bgr, face, device):
    import torch
    from insightface.utils import face_align

    input_mean = 127.5
    input_std = 127.5
    aligned_bgr = face_align.norm_crop(image_bgr, landmark=face.kps)
    arcface = cv2.dnn.blobFromImages(
        [aligned_bgr],
        1.0 / input_std,
        (112, 112),
        (input_mean, input_mean, input_mean),
        swapRB=True,
    )[0]
    aligned_rgb = cv2.cvtColor(cv2.resize(aligned_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    image = aligned_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    arcface_tensor = torch.from_numpy(arcface).unsqueeze(0).to(device)
    return image_tensor, arcface_tensor, aligned_bgr


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("# MICA metric FLAME geometry, CPU export, coordinates in millimeters\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mica-repo", required=True, type=Path, help="Path to the external MICA checkout")
    parser.add_argument("--model", type=Path, default=None, help="Path to data/pretrained/mica.tar")
    parser.add_argument("--input-dir", type=Path, help="Directory of input face photos/crops")
    parser.add_argument("--pattern", default="*", help="Glob pattern inside --input-dir")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Single input image; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", default="case_a")
    parser.add_argument("--session", default="single_photo")
    args = parser.parse_args()

    _patch_legacy_deps()

    mica_repo = args.mica_repo.resolve()
    if not mica_repo.exists():
        raise FileNotFoundError(mica_repo)
    model_path = (args.model or mica_repo / "data" / "pretrained" / "mica.tar").resolve()
    _check_assets(mica_repo, model_path)
    sys.path.insert(0, str(mica_repo))

    import torch
    import torch.nn.functional as F
    import trimesh
    from configs.config import get_cfg_defaults
    from insightface.app import FaceAnalysis
    from micalib.models.mica import MICA

    images = _iter_images(args.input_dir, args.image, args.pattern)
    if not images:
        raise ValueError("No input images found")

    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg_defaults()
    cfg.device = "cpu"
    cfg.model.testing = True
    cfg.pretrained_model_path = ""
    cfg.output_dir = str(args.output_dir / "_no_auto_load")
    mica = MICA(cfg, device)

    checkpoint = torch.load(model_path, map_location=device)
    if "arcface" in checkpoint:
        mica.arcface.load_state_dict(checkpoint["arcface"])
    if "flameModel" in checkpoint:
        mica.flameModel.load_state_dict(checkpoint["flameModel"])
    mica.eval()

    detector = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(224, 224))

    faces = mica.flameModel.generator.faces_tensor.detach().cpu().numpy().astype(np.int64)
    summary = []

    with torch.no_grad():
        for image_path in images:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                summary.append({"source_image": str(image_path), "status": "failed", "error": "cv2.imread returned None"})
                continue
            face = _select_center_face(detector.get(image_bgr), image_bgr.shape)
            if face is None:
                summary.append({"source_image": str(image_path), "status": "failed", "error": "no face detected"})
                continue

            image_tensor, arcface_tensor, aligned_bgr = _make_mica_tensors(image_bgr, face, device)
            codedict = {
                "arcface": F.normalize(mica.arcface(arcface_tensor)),
                "images": image_tensor,
            }
            opdict = mica.decode(codedict)
            mesh_m = opdict["pred_canonical_shape_vertices"][0].detach().cpu().numpy()
            shape_code = opdict["pred_shape_code"][0].detach().cpu().numpy()
            landmarks_m = mica.flame.compute_landmarks(opdict["pred_canonical_shape_vertices"])[0].detach().cpu().numpy()

            mesh_mm = mesh_m * 1000.0
            landmarks_mm = landmarks_m * 1000.0
            stem = _safe_stem(image_path)
            output_base = f"{args.subject}_{args.session}_{stem}_mica_cpu"
            obj_path = args.output_dir / f"{output_base}.obj"
            ply_path = args.output_dir / f"{output_base}.ply"
            npz_path = args.output_dir / f"{output_base}.npz"
            aligned_path = args.output_dir / f"{output_base}_aligned.jpg"
            metadata_path = args.output_dir / f"{output_base}_metadata.json"

            _write_obj(obj_path, mesh_mm, faces)
            trimesh.Trimesh(vertices=mesh_mm, faces=faces, process=False).export(ply_path)
            cv2.imwrite(str(aligned_path), aligned_bgr)
            np.savez_compressed(
                npz_path,
                vertices_mm=mesh_mm,
                faces=faces,
                landmarks68_mm=landmarks_mm,
                identity_code=shape_code,
                bbox_xyxy_score=np.asarray(face.bbox),
            )
            metadata = {
                "method": "MICA CPU geometry",
                "source_image": str(image_path),
                "mica_repo": str(mica_repo),
                "model": str(model_path),
                "output_obj": str(obj_path),
                "output_ply": str(ply_path),
                "output_npz": str(npz_path),
                "aligned_image": str(aligned_path),
                "vertex_count": int(mesh_mm.shape[0]),
                "face_count": int(faces.shape[0]),
                "coordinate_note": "MICA metric FLAME coordinates exported in millimeters.",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            summary.append(metadata | {"status": "ok", "metadata": str(metadata_path)})

    summary_path = args.output_dir / f"{args.subject}_{args.session}_mica_cpu_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "items": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
