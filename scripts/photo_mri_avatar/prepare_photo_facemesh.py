"""Build a lightweight photo-derived 3D face mesh from a single image.

This is a fast baseline, not a metrically accurate avatar. It uses MediaPipe
FaceMesh landmarks, Delaunay triangulation in image space, and sampled vertex
colors from the source photo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from PIL import Image, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import Delaunay


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image)


def detect_landmarks(rgb: np.ndarray, model: Path) -> np.ndarray:
    if not model.exists():
        raise FileNotFoundError(
            f"Face landmarker model not found: {model}. "
            "Download face_landmarker.task from MediaPipe first."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    detector.close()

    if not result.face_landmarks:
        raise ValueError("MediaPipe FaceMesh did not detect a face.")

    landmarks = result.face_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def landmarks_to_mesh(rgb: np.ndarray, landmarks: np.ndarray) -> trimesh.Trimesh:
    height, width = rgb.shape[:2]
    xy = landmarks[:, :2].copy()

    vertices = np.column_stack(
        [
            (landmarks[:, 0] - 0.5) * width,
            (0.5 - landmarks[:, 1]) * height,
            -landmarks[:, 2] * width,
        ]
    )
    vertices -= vertices.mean(axis=0, keepdims=True)

    tri = Delaunay(xy)
    faces = tri.simplices.astype(np.int64)

    px = np.clip((landmarks[:, 0] * (width - 1)).round().astype(int), 0, width - 1)
    py = np.clip((landmarks[:, 1] * (height - 1)).round().astype(int), 0, height - 1)
    colors = rgb[py, px]
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.concatenate([colors, alpha], axis=1)
    return mesh


def write_overlay(rgb: np.ndarray, landmarks: np.ndarray, output: Path) -> None:
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    points = np.column_stack(
        [
            np.clip((landmarks[:, 0] * (width - 1)).round().astype(int), 0, width - 1),
            np.clip((landmarks[:, 1] * (height - 1)).round().astype(int), 0, height - 1),
        ]
    )
    for x, y in points:
        cv2.circle(bgr, (int(x), int(y)), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    cv2.imwrite(str(output), bgr)


def write_preview(mesh: trimesh.Trimesh, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4), dpi=160)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    views = [(0, -90, "front"), (0, 0, "side"), (90, -90, "top")]
    colors = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255.0

    for ax, (elev, azim, title) in zip(axes, views):
        ax.scatter(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            c=colors,
            s=2,
            alpha=0.9,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1.2, 0.6])

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Face photo.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="MediaPipe face_landmarker.task.")
    parser.add_argument("--subject", default="subject")
    parser.add_argument("--session", default="photo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(args.input)
    landmarks = detect_landmarks(rgb, args.model)
    mesh = landmarks_to_mesh(rgb, landmarks)

    stem = f"{args.subject}_{args.session}"
    mesh_path = args.output_dir / f"{stem}_mediapipe_facemesh.ply"
    overlay_path = args.output_dir / f"{stem}_landmarks_overlay.jpg"
    preview_path = args.output_dir / f"{stem}_facemesh_preview.png"
    metadata_path = args.output_dir / f"{stem}_metadata.json"

    mesh.export(mesh_path)
    write_overlay(rgb, landmarks, overlay_path)
    write_preview(mesh, preview_path)

    metadata = {
        "input": str(args.input),
        "model": str(args.model),
        "output_mesh": str(mesh_path),
        "overlay": str(overlay_path),
        "preview": str(preview_path),
        "subject": args.subject,
        "session": args.session,
        "image_shape": list(rgb.shape),
        "landmarks": int(len(landmarks)),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "warning": "MediaPipe FaceMesh is a lightweight single-photo baseline, not metric 3D ground truth.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
