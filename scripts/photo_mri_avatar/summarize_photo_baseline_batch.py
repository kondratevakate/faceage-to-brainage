"""Summarize batch photo-avatar baselines.

Outputs:
- photo inventory for the requested folders;
- per-method detection/reconstruction status;
- per-mesh stability against other meshes from the same method;
- optional MRI comparison metrics joined by mesh filename.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def inventory_photos(photo_dirs: list[tuple[str, Path]], pattern: str) -> list[dict]:
    rows = []
    for label, photo_dir in photo_dirs:
        for path in sorted(photo_dir.glob(pattern)):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
            rows.append(
                {
                    "group": label,
                    "image_name": path.name,
                    "image_path": str(path.resolve()),
                    "width": width,
                    "height": height,
                    "megapixels": round(width * height / 1_000_000, 4),
                    "file_bytes": path.stat().st_size,
                }
            )
    return rows


def load_mesh(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def similarity_from_correspondence(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = src_c.T @ dst_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    var = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s) / max(var, 1e-12))
    t = dst_mean - scale * (r @ src_mean)
    return r, scale, t


def apply_similarity(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ points.T)).T + t


def bbox_diag(points: np.ndarray) -> float:
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extents))


def load_mediapipe_outputs(output_dir: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(output_dir.glob("*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        mesh_path = Path(meta["output_mesh"])
        vertices = load_mesh(mesh_path)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        height, width = meta.get("image_shape", [0, 0])[:2]
        face_area_pct = 100 * extents[0] * extents[1] / max(width * height, 1)
        rows.append(
            {
                "method": "mediapipe",
                "source_image": str(Path(meta["input"]).resolve()),
                "mesh_path": str(mesh_path.resolve()),
                "mesh_name": mesh_path.name,
                "metadata": str(metadata_path.resolve()),
                "vertex_count": int(meta.get("mesh_vertices", 0)),
                "face_count": int(meta.get("mesh_faces", 0)),
                "face_bbox_width_px": float(extents[0]),
                "face_bbox_height_px": float(extents[1]),
                "face_area_pct": float(face_area_pct),
                "detector_score": "",
            }
        )
    return rows


def load_3ddfa_outputs(output_dir: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(output_dir.glob("*_3ddfa_v2_face*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        mesh_path = Path(str(metadata_path).replace("_metadata.json", ".ply"))
        if not mesh_path.exists():
            continue
        source_image = Path(meta["source_image"]).resolve()
        with Image.open(source_image) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
        det_box = meta.get("detected_box_xyxy_score", ["", "", "", "", ""])
        box_width = float(det_box[2] - det_box[0]) if len(det_box) >= 4 else ""
        box_height = float(det_box[3] - det_box[1]) if len(det_box) >= 4 else ""
        face_area_pct = (
            100 * box_width * box_height / max(width * height, 1)
            if box_width != "" and box_height != ""
            else ""
        )
        rows.append(
            {
                "method": "3ddfa_v2",
                "source_image": str(source_image),
                "mesh_path": str(mesh_path.resolve()),
                "mesh_name": mesh_path.name,
                "metadata": str(metadata_path.resolve()),
                "vertex_count": int(meta.get("vertex_count", 0)),
                "face_count": int(meta.get("face_count", 0)),
                "face_bbox_width_px": box_width,
                "face_bbox_height_px": box_height,
                "face_area_pct": face_area_pct,
                "detector_score": float(det_box[4]) if len(det_box) >= 5 else "",
            }
        )
    return rows


def load_mri_scores(values: list[str]) -> dict[tuple[str, str], dict]:
    scores = {}
    for value in values:
        method, csv_path = parse_label_path(value)
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mesh_name = Path(row["photo_mesh"]).name
                scores[(method, mesh_name)] = {
                    "mri_score": row.get("score", ""),
                    "mri_median_mm": row.get("median_mm", ""),
                    "mri_p90_mm": row.get("p90_mm", ""),
                    "mri_mean_mm": row.get("mean_mm", ""),
                    "mri_alignment_preview": row.get("alignment_preview", ""),
                }
    return scores


def pairwise_stability(outputs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    pairs = []
    by_method = defaultdict(list)
    vertices_cache = {}
    for row in outputs:
        by_method[row["method"]].append(row)

    for method, method_rows in by_method.items():
        for i, left in enumerate(method_rows):
            left_vertices = vertices_cache.setdefault(left["mesh_path"], load_mesh(Path(left["mesh_path"])))
            for right in method_rows[i + 1 :]:
                right_vertices = vertices_cache.setdefault(right["mesh_path"], load_mesh(Path(right["mesh_path"])))
                if len(left_vertices) != len(right_vertices):
                    continue
                r, scale, t = similarity_from_correspondence(left_vertices, right_vertices)
                aligned = apply_similarity(left_vertices, r, scale, t)
                distances = np.linalg.norm(aligned - right_vertices, axis=1)
                norm = bbox_diag(right_vertices)
                pairs.append(
                    {
                        "method": method,
                        "left_mesh": left["mesh_name"],
                        "right_mesh": right["mesh_name"],
                        "left_image": Path(left["source_image"]).name,
                        "right_image": Path(right["source_image"]).name,
                        "median_distance": float(np.median(distances)),
                        "p90_distance": float(np.percentile(distances, 90)),
                        "mean_distance": float(np.mean(distances)),
                        "median_pct_bbox": float(100 * np.median(distances) / max(norm, 1e-12)),
                        "p90_pct_bbox": float(100 * np.percentile(distances, 90) / max(norm, 1e-12)),
                        "scale_left_to_right": scale,
                    }
                )

    per_mesh = defaultdict(list)
    for pair in pairs:
        per_mesh[pair["left_mesh"]].append(pair)
        per_mesh[pair["right_mesh"]].append(pair)
    return pairs, per_mesh


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo-dir", action="append", type=parse_label_path, required=True)
    parser.add_argument("--photo-pattern", default="*")
    parser.add_argument("--mediapipe-dir", type=Path)
    parser.add_argument("--3ddfa-dir", dest="dddfa_dir", type=Path)
    parser.add_argument("--mri-csv", action="append", default=[], help="METHOD=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    inventory = inventory_photos(args.photo_dir, args.photo_pattern)
    outputs = []
    if args.mediapipe_dir:
        outputs.extend(load_mediapipe_outputs(args.mediapipe_dir))
    if args.dddfa_dir:
        outputs.extend(load_3ddfa_outputs(args.dddfa_dir))

    mri_scores = load_mri_scores(args.mri_csv)
    pairs, per_mesh_pairs = pairwise_stability(outputs)

    source_to_outputs = defaultdict(list)
    for row in outputs:
        source_to_outputs[(row["method"], row["source_image"])].append(row)

    methods = sorted({row["method"] for row in outputs})
    detection_rows = []
    for photo in inventory:
        for method in methods:
            method_outputs = source_to_outputs.get((method, photo["image_path"]), [])
            detection_rows.append(
                {
                    **photo,
                    "method": method,
                    "status": "ok" if method_outputs else "failed_or_not_run",
                    "mesh_count": len(method_outputs),
                    "mesh_names": ";".join(row["mesh_name"] for row in method_outputs),
                }
            )

    stability_rows = []
    for row in outputs:
        mesh_pairs = per_mesh_pairs.get(row["mesh_name"], [])
        pair_medians = [pair["median_pct_bbox"] for pair in mesh_pairs]
        pair_p90s = [pair["p90_pct_bbox"] for pair in mesh_pairs]
        mri = mri_scores.get((row["method"], row["mesh_name"]), {})
        stability_rows.append(
            {
                **row,
                "source_image_name": Path(row["source_image"]).name,
                "pair_count": len(mesh_pairs),
                "median_pair_median_pct_bbox": float(np.median(pair_medians)) if pair_medians else "",
                "median_pair_p90_pct_bbox": float(np.median(pair_p90s)) if pair_p90s else "",
                **mri,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "photo_inventory_batch.csv",
        inventory,
        ["group", "image_name", "image_path", "width", "height", "megapixels", "file_bytes"],
    )
    write_csv(
        args.output_dir / "baseline_detection_summary.csv",
        detection_rows,
        ["group", "image_name", "image_path", "method", "status", "mesh_count", "mesh_names"],
    )
    write_csv(
        args.output_dir / "mesh_stability_pairs.csv",
        pairs,
        [
            "method",
            "left_image",
            "right_image",
            "left_mesh",
            "right_mesh",
            "median_distance",
            "p90_distance",
            "mean_distance",
            "median_pct_bbox",
            "p90_pct_bbox",
            "scale_left_to_right",
        ],
    )
    write_csv(
        args.output_dir / "mesh_stability_summary.csv",
        stability_rows,
        [
            "method",
            "source_image_name",
            "mesh_name",
            "vertex_count",
            "face_count",
            "face_bbox_width_px",
            "face_bbox_height_px",
            "face_area_pct",
            "detector_score",
            "pair_count",
            "median_pair_median_pct_bbox",
            "median_pair_p90_pct_bbox",
            "mri_score",
            "mri_median_mm",
            "mri_p90_mm",
            "mri_mean_mm",
            "mri_alignment_preview",
            "source_image",
            "mesh_path",
            "metadata",
        ],
    )

    print(
        json.dumps(
            {
                "inventory": len(inventory),
                "outputs": len(outputs),
                "pairs": len(pairs),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
