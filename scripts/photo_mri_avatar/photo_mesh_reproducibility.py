"""Measure reproducibility of one-photo avatar meshes across photos.

This compares meshes from the same method after a front-plane landmark
alignment. It is intentionally independent of MRI. If a method is not stable
across photos of the same subject, MRI-vs-avatar differences are not directly
interpretable as reconstruction accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from landmark_mask_overlay import (
    SUPPORT_NAMES,
    apply_similarity_2d,
    extract_source_landmarks,
    load_3ddfa_keypoint_vertex_ids,
    load_geometry,
    mask_metrics,
    mesh_to_mask,
    raster_grid,
    similarity_2d,
    source_front_points,
    source_region_mask,
)


def photo_id(path: Path) -> str:
    matches = re.findall(r"(\d{2})[-_](\d{2})[-_](\d{2})", path.name)
    if matches:
        return "-".join(matches[-1])
    return path.stem


def load_case(mesh_path: Path, method: str, tddfa_keypoint_ids: np.ndarray | None, fraction_below_nose: float) -> dict:
    vertices, faces = load_geometry(mesh_path)
    landmarks = extract_source_landmarks(vertices, method, tddfa_keypoint_ids)
    return {
        "mesh_path": str(mesh_path),
        "mesh_name": mesh_path.name,
        "photo_id": photo_id(mesh_path),
        "vertices": vertices,
        "faces": faces,
        "landmarks": landmarks,
        "points_2d": source_front_points(vertices),
        "vertex_keep": source_region_mask(vertices, landmarks, fraction_below_nose),
    }


def align_left_to_right(left: dict, right: dict) -> dict:
    src_landmarks_2d = {
        name: source_front_points(left["landmarks"][name].reshape(1, 3))[0]
        for name in SUPPORT_NAMES
    }
    dst_landmarks_2d = {
        name: source_front_points(right["landmarks"][name].reshape(1, 3))[0]
        for name in SUPPORT_NAMES
    }
    src = np.vstack([src_landmarks_2d[name] for name in SUPPORT_NAMES])
    dst = np.vstack([dst_landmarks_2d[name] for name in SUPPORT_NAMES])
    r, scale, _t = similarity_2d(src, dst)
    t = dst_landmarks_2d["nose_tip"] - scale * (r @ src_landmarks_2d["nose_tip"])
    aligned_points = apply_similarity_2d(left["points_2d"], r, scale, t)
    aligned_landmarks = {
        name: apply_similarity_2d(src_landmarks_2d[name].reshape(1, 2), r, scale, t)[0]
        for name in SUPPORT_NAMES
    }
    residuals = np.linalg.norm(
        np.vstack([aligned_landmarks[name] for name in SUPPORT_NAMES]) - dst,
        axis=1,
    )
    return {
        "aligned_points": aligned_points,
        "aligned_landmarks": aligned_landmarks,
        "scale": float(scale),
        "landmark_rmse": float(np.sqrt(np.mean(residuals**2))),
        "landmark_residuals_json": json.dumps({name: float(value) for name, value in zip(SUPPORT_NAMES, residuals)}),
    }


def pair_metrics(left: dict, right: dict, resolution_mm: float) -> dict:
    aligned = align_left_to_right(left, right)
    right_points = right["points_2d"]
    grid = raster_grid(aligned["aligned_points"][left["vertex_keep"]], right_points[right["vertex_keep"]], resolution_mm, 8.0)
    left_mask = mesh_to_mask(aligned["aligned_points"], left["faces"], left["vertex_keep"], grid, dilation_iters=1)
    right_mask = mesh_to_mask(right_points, right["faces"], right["vertex_keep"], grid, dilation_iters=1)
    metrics = mask_metrics(left_mask, right_mask, resolution_mm)
    return {
        "left_photo_id": left["photo_id"],
        "right_photo_id": right["photo_id"],
        "left_mesh_name": left["mesh_name"],
        "right_mesh_name": right["mesh_name"],
        "scale_left_to_right": aligned["scale"],
        "landmark_rmse_mm": aligned["landmark_rmse"],
        "landmark_residuals_json": aligned["landmark_residuals_json"],
        "left_vertices_in_mask": int(left["vertex_keep"].sum()),
        "right_vertices_in_mask": int(right["vertex_keep"].sum()),
        **metrics,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    out = {"n_pairs": len(rows)}
    for metric in ["dice", "boundary_hd95_mm", "boundary_assd_mm", "landmark_rmse_mm", "scale_left_to_right"]:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        out[f"{metric}_median"] = float(np.median(values))
        out[f"{metric}_p90"] = float(np.percentile(values, 90))
        out[f"{metric}_min"] = float(np.min(values))
        out[f"{metric}_max"] = float(np.max(values))
    return out


def plot(rows_by_method: dict[str, list[dict]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for ax, metric, ylabel in [
        (axes[0], "dice", "Dice, higher is more reproducible"),
        (axes[1], "boundary_hd95_mm", "HD95, lower is more reproducible"),
    ]:
        labels = []
        data = []
        for method, rows in rows_by_method.items():
            labels.append(method)
            data.append([float(row[metric]) for row in rows])
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Photo-to-photo front-mask reproducibility")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run_one(args: argparse.Namespace, method: str, mesh_dir: Path, pattern: str, tddfa_keypoint_ids: np.ndarray | None) -> list[dict]:
    meshes = sorted(mesh_dir.glob(pattern))
    cases = [load_case(path, method, tddfa_keypoint_ids, args.source_fraction_below_nose) for path in meshes]
    rows = []
    for left, right in combinations(cases, 2):
        row = pair_metrics(left, right, args.resolution_mm)
        row["method"] = method
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--3ddfa-dir", dest="dddfa_dir", type=Path)
    parser.add_argument("--3ddfa-pattern", dest="dddfa_pattern", default="*.ply")
    parser.add_argument("--mediapipe-dir", type=Path)
    parser.add_argument("--mediapipe-pattern", default="*.ply")
    parser.add_argument("--bfm-pkl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-fraction-below-nose", type=float, default=0.35)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_method: dict[str, list[dict]] = {}
    if args.dddfa_dir:
        if args.bfm_pkl is None:
            raise ValueError("--bfm-pkl is required for 3DDFA")
        ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)
        rows_by_method["3ddfa_v2"] = run_one(args, "3ddfa_v2", args.dddfa_dir, args.dddfa_pattern, ids)
    if args.mediapipe_dir:
        rows_by_method["mediapipe"] = run_one(args, "mediapipe", args.mediapipe_dir, args.mediapipe_pattern, None)

    all_rows = []
    summary = {}
    for method, rows in rows_by_method.items():
        for row in rows:
            all_rows.append(row)
        summary[method] = summarize(rows)
    write_rows(args.output_dir / "photo_reproducibility_pairs.csv", all_rows)
    (args.output_dir / "photo_reproducibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows_by_method, args.output_dir / "photo_reproducibility_boxplot.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
