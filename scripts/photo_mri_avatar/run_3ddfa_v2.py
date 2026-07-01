"""Run 3DDFA_V2 as a local single-photo face-mesh baseline.

The upstream demo imports the optional renderer even when only PLY output is
needed. This runner uses only FaceBoxes + TDDFA and writes dense PLY meshes plus
small QC overlays, so it can run on CPU without Sim3DR.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_repo_path(repo: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo / path).resolve())


def _iter_images(input_dir: Path | None, images: list[Path], pattern: str) -> list[Path]:
    found = []
    if input_dir is not None:
        for path in sorted(input_dir.glob(pattern)):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                found.append(path)
    found.extend(images)
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def _write_ply(vertices, triangles, image_height: int, output_path: Path) -> None:
    n_vertex = vertices.shape[1]
    n_face = triangles.shape[0]
    with output_path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertex}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {n_face}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n_vertex):
            x, y, z = vertices[:, i]
            f.write(f"{x:.4f} {image_height - y:.4f} {z:.4f}\n")
        for i in range(n_face):
            idx1, idx2, idx3 = triangles[i]
            f.write(f"3 {idx3} {idx2} {idx1}\n")


def _draw_overlay(img, vertices, box):
    from utils.functions import cv_draw_landmark

    return cv_draw_landmark(img.copy(), vertices, box=box, color=(0, 255, 255), size=1)


def _install_python_nms_fallback(repo: Path) -> None:
    """Use pure-Python NMS if the optional FaceBoxes Cython module is absent."""
    nms_dir = repo / "FaceBoxes" / "utils" / "nms"
    compiled = list(nms_dir.glob("cpu_nms*.pyd")) + list(nms_dir.glob("cpu_nms*.so"))
    if compiled:
        return

    def py_cpu_nms(dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            overlap = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(overlap <= thresh)[0]
            order = order[inds + 1]
        return keep

    fallback = types.ModuleType("FaceBoxes.utils.nms_wrapper")

    def nms(dets, thresh):
        if dets.shape[0] == 0:
            return []
        return py_cpu_nms(dets, thresh)

    fallback.nms = nms
    sys.modules["FaceBoxes.utils.nms_wrapper"] = fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to the 3DDFA_V2 checkout")
    parser.add_argument("--config", default="configs/mb1_120x120.yml", help="3DDFA_V2 YAML config")
    parser.add_argument("--input-dir", type=Path, help="Directory of input face photos")
    parser.add_argument("--pattern", default="*", help="Glob pattern inside --input-dir")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Single input image; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", default="subject")
    parser.add_argument("--session", default="single_photo")
    parser.add_argument("--max-faces", type=int, default=1)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    sys.path.insert(0, str(repo))
    _install_python_nms_fallback(repo)

    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = _resolve_repo_path(repo, cfg.get("checkpoint_fp"))
    cfg["bfm_fp"] = _resolve_repo_path(repo, cfg.get("bfm_fp"))

    images = _iter_images(args.input_dir, args.image, args.pattern)
    if not images:
        raise ValueError("No input images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)

    summary = []
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            summary.append({"image": str(image_path), "status": "failed", "error": "cv2.imread returned None"})
            continue

        boxes = sorted(face_boxes(img), key=lambda b: b[4], reverse=True)
        if args.max_faces > 0:
            boxes = boxes[: args.max_faces]
        if not boxes:
            summary.append({"image": str(image_path), "status": "failed", "error": "no face detected"})
            continue

        param_lst, roi_box_lst = tddfa(img, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        stem = _safe_stem(image_path)

        outputs = []
        for face_idx, (vertices, roi_box, det_box) in enumerate(zip(ver_lst, roi_box_lst, boxes), start=1):
            output_base = f"{args.subject}_{args.session}_{stem}_3ddfa_v2_face{face_idx}"
            ply_path = args.output_dir / f"{output_base}.ply"
            overlay_path = args.output_dir / f"{output_base}_overlay.jpg"
            metadata_path = args.output_dir / f"{output_base}_metadata.json"

            _write_ply(vertices, tddfa.tri, img.shape[0], ply_path)
            overlay = _draw_overlay(img, vertices, roi_box)
            cv2.imwrite(str(overlay_path), overlay)
            metadata = {
                "method": "3DDFA_V2",
                "source_image": str(image_path),
                "config": str(config_path),
                "detected_box_xyxy_score": [float(x) for x in det_box],
                "roi_box_xyxy": [float(x) for x in roi_box],
                "vertex_count": int(vertices.shape[1]),
                "face_count": int(tddfa.tri.shape[0]),
                "coordinate_note": "PLY is in image-pixel-like 3DDFA coordinates; y is flipped to match upstream serialization.",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            outputs.append({"ply": str(ply_path), "overlay": str(overlay_path), "metadata": str(metadata_path)})

        summary.append({"image": str(image_path), "status": "ok", "faces": len(outputs), "outputs": outputs})

    summary_path = args.output_dir / f"{args.subject}_{args.session}_3ddfa_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "items": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
