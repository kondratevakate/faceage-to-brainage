"""Run an algorithmic MRI face-target cleaning experiment.

The experiment separates two decisions:

1. MRI-only target quality: crop shape, margin, smoothing, subdivision, and
   sampling stability are scored without looking at avatar fit.
2. Avatar diagnostics: 3DDFA and MediaPipe are compared to a small number of
   MRI candidates only after those candidates pass MRI-only gates.

The output is exploratory QC, not a validated anatomical ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from landmark_mask_overlay import (
    SUPPORT_NAMES,
    apply_similarity_2d,
    extract_source_landmarks,
    load_3ddfa_keypoint_vertex_ids,
    load_geometry,
    mask_metrics,
    mesh_to_mask,
    maybe_swap_lr,
    mri_front_points,
    points_to_mask,
    raster_grid,
    similarity_2d,
    source_front_points,
    source_region_mask,
)


@dataclass(frozen=True)
class CandidateSpec:
    shape: str
    margin_fraction: float
    taubin_iters: int
    subdivide_iters: int
    decimate_fraction: float


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(tuple(geom.geometry.values()))
    if not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"Cannot read mesh from {path}")
    return geom


def finite_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    out = max(parts, key=lambda part: len(part.vertices))
    out.remove_unreferenced_vertices()
    return out


def component_fraction(mesh: trimesh.Trimesh) -> float:
    parts = mesh.split(only_watertight=False)
    if not parts or len(mesh.vertices) == 0:
        return 0.0
    return float(max(len(part.vertices) for part in parts) / len(mesh.vertices))


def crop_by_landmarks(
    source_mesh: trimesh.Trimesh,
    landmarks: dict[str, np.ndarray],
    shape: str,
    margin_fraction: float,
) -> trimesh.Trimesh:
    vertices = finite_vertices(source_mesh)
    faces = np.asarray(source_mesh.faces, dtype=np.int64)
    nose = landmarks["nose_tip"]
    brow = landmarks["brow_center"]
    chin = landmarks["chin"]
    left = landmarks["left_cheek"]
    right = landmarks["right_cheek"]

    cheek_width = float(np.linalg.norm(right[[0, 2]] - left[[0, 2]]))
    x_min = min(left[0], right[0], nose[0], brow[0], chin[0]) - margin_fraction * cheek_width
    x_max = max(left[0], right[0], nose[0], brow[0], chin[0]) + margin_fraction * cheek_width
    z_min = min(chin[2], nose[2], left[2], right[2]) - (0.15 + margin_fraction) * cheek_width
    z_max = max(brow[2], nose[2], left[2], right[2]) + (0.25 + margin_fraction) * cheek_width

    if shape == "rect":
        keep = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) & (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
    elif shape == "ellipse":
        cx = 0.5 * (x_min + x_max)
        cz = 0.5 * (z_min + z_max)
        rx = max(0.5 * (x_max - x_min), 1e-6)
        rz = max(0.5 * (z_max - z_min), 1e-6)
        keep = ((vertices[:, 0] - cx) / rx) ** 2 + ((vertices[:, 2] - cz) / rz) ** 2 <= 1.0
    else:
        raise ValueError(shape)

    face_keep = keep[faces].all(axis=1)
    if face_keep.sum() == 0:
        raise ValueError(f"No faces after {shape} crop margin={margin_fraction}")
    out = source_mesh.submesh([face_keep], append=True, repair=False)
    out.remove_unreferenced_vertices()
    return largest_component(out)


def process_mesh(mesh: trimesh.Trimesh, spec: CandidateSpec) -> trimesh.Trimesh:
    out = mesh.copy()
    for _ in range(spec.subdivide_iters):
        vertices, faces = trimesh.remesh.subdivide(np.asarray(out.vertices), np.asarray(out.faces))
        out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if spec.taubin_iters > 0:
        trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=spec.taubin_iters)
    if spec.decimate_fraction < 0.999:
        try:
            out = out.simplify_quadric_decimation(percent=float(spec.decimate_fraction), aggression=4)
        except Exception as exc:
            out.metadata["decimation_warning"] = str(exc)
    out.remove_unreferenced_vertices()
    return largest_component(out)


def deterministic_points(vertices: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(vertices) <= n:
        return vertices
    rng = np.random.default_rng(seed)
    return vertices[rng.choice(len(vertices), size=n, replace=False)]


def distance_block(points_a: np.ndarray, points_b: np.ndarray, prefix: str) -> dict[str, float]:
    a = deterministic_points(points_a, 12000, 11)
    b = deterministic_points(points_b, 12000, 13)
    a2b, _ = cKDTree(b).query(a, k=1, workers=-1)
    b2a, _ = cKDTree(a).query(b, k=1, workers=-1)
    return {
        f"{prefix}_assd_mm": float((np.mean(a2b) + np.mean(b2a)) / 2.0),
        f"{prefix}_hd95_mm": float(max(np.percentile(a2b, 95), np.percentile(b2a, 95))),
        f"{prefix}_a2b_p95_mm": float(np.percentile(a2b, 95)),
        f"{prefix}_b2a_p95_mm": float(np.percentile(b2a, 95)),
    }


def local_roughness(vertices: np.ndarray, sample_n: int, k: int) -> dict[str, float]:
    sample = deterministic_points(vertices, sample_n, 23)
    tree = cKDTree(vertices)
    _distances, indices = tree.query(sample, k=min(k, len(vertices)), workers=-1)
    residuals = []
    for neighbor_ids in indices:
        points = vertices[np.atleast_1d(neighbor_ids)]
        centered = points - points.mean(axis=0)
        cov = centered.T @ centered / max(len(points), 1)
        eigvals = np.linalg.eigvalsh(cov)
        residuals.append(np.sqrt(max(float(eigvals[0]), 0.0)))
    residuals = np.asarray(residuals)
    return {
        "roughness_median_mm": float(np.median(residuals)),
        "roughness_p95_mm": float(np.percentile(residuals, 95)),
    }


def xz_mask_quality(points_xz: np.ndarray, resolution_mm: float) -> dict[str, float]:
    grid = raster_grid(points_xz, points_xz, resolution_mm, 3.0)
    mask = points_to_mask(points_xz, grid, dilation_iters=1)
    area = float(mask.sum() * resolution_mm * resolution_mm)
    boundary = mask ^ binary_erosion(mask)
    boundary_length = float(boundary.sum() * resolution_mm)
    column_occ = mask.sum(axis=0).astype(np.float64)
    active = column_occ[column_occ > 0]
    if len(active) < 8:
        stripe_ratio = 1.0
        empty_column_fraction = 1.0
    else:
        centered = active - np.mean(active)
        power = np.abs(np.fft.rfft(centered)) ** 2
        stripe_ratio = float(np.sum(power[len(power) // 3 :]) / max(np.sum(power[1:]), 1e-12))
        empty_column_fraction = float(np.mean(column_occ == 0))
    return {
        "front_area_mm2": area,
        "front_boundary_length_mm": boundary_length,
        "front_boundary_per_area": float(boundary_length / max(area, 1e-12)),
        "stripe_highfreq_ratio": stripe_ratio,
        "empty_column_fraction": empty_column_fraction,
    }


def window_y(vertices: np.ndarray, center: np.ndarray, radius: float) -> dict[str, float]:
    keep = (np.abs(vertices[:, 0] - center[0]) <= radius) & (np.abs(vertices[:, 2] - center[2]) <= radius)
    if keep.sum() == 0:
        return {"n": 0, "median_y_mm": np.nan, "p95_y_mm": np.nan, "nearest_xz_y_mm": np.nan}
    local = vertices[keep]
    y = local[:, 1]
    xz_distance = np.linalg.norm(local[:, [0, 2]] - center[[0, 2]], axis=1)
    return {
        "n": int(len(y)),
        "median_y_mm": float(np.median(y)),
        "p95_y_mm": float(np.percentile(y, 95)),
        "nearest_xz_y_mm": float(local[int(np.argmin(xz_distance)), 1]),
    }


def anatomical_metrics(vertices: np.ndarray, landmarks: dict[str, np.ndarray]) -> dict[str, float]:
    tree = cKDTree(vertices)
    landmark_dist = {}
    for name, point in landmarks.items():
        dist, _idx = tree.query(point, k=1)
        landmark_dist[f"landmark_{name}_nearest_mm"] = float(dist)

    nose = window_y(vertices, landmarks["nose_tip"], 6.0)
    left = window_y(vertices, landmarks["left_cheek"], 8.0)
    right = window_y(vertices, landmarks["right_cheek"], 8.0)
    cheek_mean = np.nanmean([left["median_y_mm"], right["median_y_mm"]])
    cheek_nearest_mean = np.nanmean([left["nearest_xz_y_mm"], right["nearest_xz_y_mm"]])
    nose_prominence_p95 = float(nose["p95_y_mm"] - cheek_mean) if np.isfinite(cheek_mean) else np.nan
    nose_prominence_nearest = (
        float(nose["nearest_xz_y_mm"] - cheek_nearest_mean)
        if np.isfinite(cheek_nearest_mean)
        else np.nan
    )

    return {
        **landmark_dist,
        "nose_window_vertices": int(nose["n"]),
        "nose_window_p95_y_mm": float(nose["p95_y_mm"]),
        "nose_window_nearest_xz_y_mm": float(nose["nearest_xz_y_mm"]),
        "cheek_window_mean_y_mm": float(cheek_mean),
        "nose_prominence_window_p95_mm": nose_prominence_p95,
        "nose_prominence_nearest_xz_mm": nose_prominence_nearest,
        "lower_coverage_below_nose_mm": float(landmarks["nose_tip"][2] - vertices[:, 2].min()),
        "upper_coverage_above_nose_mm": float(vertices[:, 2].max() - landmarks["nose_tip"][2]),
        "xz_width_mm": float(vertices[:, 0].max() - vertices[:, 0].min()),
        "z_height_mm": float(vertices[:, 2].max() - vertices[:, 2].min()),
    }


def mri_only_row(
    candidate_id: str,
    spec: CandidateSpec,
    mesh: trimesh.Trimesh,
    landmarks: dict[str, np.ndarray],
    reference_vertices: np.ndarray,
    crop_reference_vertices: np.ndarray,
) -> dict[str, float | int | str | bool]:
    vertices = finite_vertices(mesh)
    row: dict[str, float | int | str | bool] = {
        "candidate_id": candidate_id,
        "shape": spec.shape,
        "margin_fraction": float(spec.margin_fraction),
        "taubin_iters": int(spec.taubin_iters),
        "subdivide_iters": int(spec.subdivide_iters),
        "decimate_fraction": float(spec.decimate_fraction),
        "vertices": int(len(vertices)),
        "faces": int(len(mesh.faces)),
        "largest_component_fraction": component_fraction(mesh),
        "surface_area_mm2": float(mesh.area) if len(mesh.faces) else np.nan,
    }
    row.update(xz_mask_quality(vertices[:, [0, 2]], resolution_mm=1.0))
    row.update(local_roughness(vertices, sample_n=2500, k=24))
    row.update(anatomical_metrics(vertices, landmarks))
    row.update(distance_block(vertices, reference_vertices, "to_reference"))
    row.update(distance_block(vertices, crop_reference_vertices, "to_unsmoothed_crop"))

    landmark_max = max(float(row[f"landmark_{name}_nearest_mm"]) for name in landmarks)
    pass_gates = (
        row["faces"] > 1000
        and row["largest_component_fraction"] >= 0.98
        and row["nose_prominence_nearest_xz_mm"] >= 12.0
        and row["lower_coverage_below_nose_mm"] >= 30.0
        and row["upper_coverage_above_nose_mm"] >= 20.0
        and landmark_max <= 8.0
        and row["to_unsmoothed_crop_hd95_mm"] <= 4.0
    )
    row["pass_mri_only_gates"] = bool(pass_gates)
    row["mri_only_score"] = float(
        row["roughness_median_mm"]
        + 8.0 * row["stripe_highfreq_ratio"]
        + 8.0 * row["front_boundary_per_area"]
        + max(0.0, 18.0 - row["nose_prominence_nearest_xz_mm"]) * 0.8
        + max(0.0, landmark_max - 3.0) * 0.8
        + 0.02 * row["to_reference_hd95_mm"]
        + max(0.0, row["to_unsmoothed_crop_hd95_mm"] - 1.5) * 0.8
    )
    return row


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


def avatar_alignment(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    landmarks_3d: dict[str, np.ndarray],
    target_landmarks_2d: dict[str, np.ndarray],
    source_fraction_below_nose: float,
) -> dict:
    candidates = [("normal", landmarks_3d), ("lr_swapped", maybe_swap_lr(landmarks_3d))]
    best = None
    source_points_2d = source_front_points(vertices)
    dst = np.vstack([target_landmarks_2d[name] for name in SUPPORT_NAMES])
    for orientation, candidate_landmarks in candidates:
        src_landmarks_2d = {
            name: source_front_points(candidate_landmarks[name].reshape(1, 3))[0]
            for name in SUPPORT_NAMES
        }
        src = np.vstack([src_landmarks_2d[name] for name in SUPPORT_NAMES])
        r, scale, _t = similarity_2d(src, dst)
        t = target_landmarks_2d["nose_tip"] - scale * (r @ src_landmarks_2d["nose_tip"])
        aligned_points = apply_similarity_2d(source_points_2d, r, scale, t)
        aligned_landmarks = {
            name: apply_similarity_2d(src_landmarks_2d[name].reshape(1, 2), r, scale, t)[0]
            for name in SUPPORT_NAMES
        }
        residuals = np.linalg.norm(np.vstack([aligned_landmarks[name] for name in SUPPORT_NAMES]) - dst, axis=1)
        row = {
            "orientation": orientation,
            "scale_2d": float(scale),
            "landmark_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
            "landmark_residuals_json": json.dumps({name: float(value) for name, value in zip(SUPPORT_NAMES, residuals)}),
            "aligned_points_2d": aligned_points,
            "vertex_keep": source_region_mask(vertices, candidate_landmarks, source_fraction_below_nose),
            "faces": faces,
        }
        score = row["landmark_rmse_mm"]
        if best is None or score < best["landmark_rmse_mm"]:
            best = row
    if best is None:
        raise ValueError("No avatar alignment candidate")
    return best


def avatar_vs_target_metrics(
    target_vertices: np.ndarray,
    target_landmarks_2d: dict[str, np.ndarray],
    avatar: dict,
    resolution_mm: float,
) -> dict[str, float]:
    target_points_2d = mri_front_points(target_vertices)
    grid = raster_grid(target_points_2d, avatar["aligned_points_2d"][avatar["vertex_keep"]], resolution_mm, 8.0)
    target_mask = points_to_mask(target_points_2d, grid, dilation_iters=1)
    source_mask = mesh_to_mask(
        avatar["aligned_points_2d"],
        avatar["faces"],
        avatar["vertex_keep"],
        grid,
        dilation_iters=1,
    )
    out = mask_metrics(source_mask, target_mask, resolution_mm)
    out.update(
        {
            "orientation": avatar["orientation"],
            "scale_2d": avatar["scale_2d"],
            "landmark_rmse_mm": avatar["landmark_rmse_mm"],
            "landmark_residuals_json": avatar["landmark_residuals_json"],
        }
    )
    return out


def avatar_disagreement_metrics(avatar_a: dict, avatar_b: dict, resolution_mm: float) -> dict[str, float]:
    a_points = avatar_a["aligned_points_2d"][avatar_a["vertex_keep"]]
    b_points = avatar_b["aligned_points_2d"][avatar_b["vertex_keep"]]
    grid = raster_grid(a_points, b_points, resolution_mm, 8.0)
    mask_a = mesh_to_mask(avatar_a["aligned_points_2d"], avatar_a["faces"], avatar_a["vertex_keep"], grid, dilation_iters=1)
    mask_b = mesh_to_mask(avatar_b["aligned_points_2d"], avatar_b["faces"], avatar_b["vertex_keep"], grid, dilation_iters=1)
    out = mask_metrics(mask_a, mask_b, resolution_mm)
    out["scale_ratio_3ddfa_to_mediapipe"] = float(avatar_a["scale_2d"] / max(avatar_b["scale_2d"], 1e-12))
    return out


def plot_qc(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    colors = {"rect": "#2563eb", "ellipse": "#dc2626"}
    for shape in sorted({row["shape"] for row in rows}):
        subset = [row for row in rows if row["shape"] == shape]
        x = [row["taubin_iters"] + 0.7 * row["subdivide_iters"] for row in subset]
        axes[0, 0].scatter(x, [row["roughness_median_mm"] for row in subset], c=colors.get(shape, "black"), label=shape, alpha=0.75)
        axes[0, 1].scatter(x, [row["stripe_highfreq_ratio"] for row in subset], c=colors.get(shape, "black"), label=shape, alpha=0.75)
        axes[1, 0].scatter(x, [row["nose_prominence_nearest_xz_mm"] for row in subset], c=colors.get(shape, "black"), label=shape, alpha=0.75)
        axes[1, 1].scatter(x, [row["mri_only_score"] for row in subset], c=colors.get(shape, "black"), label=shape, alpha=0.75)
    axes[0, 0].set_title("Local roughness")
    axes[0, 1].set_title("Front-plane stripe index")
    axes[1, 0].set_title("Nose prominence")
    axes[1, 1].set_title("Exploratory MRI-only score")
    for ax in axes.ravel():
        ax.set_xlabel("Taubin iterations (+0.7 if subdivided)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-broad-mesh", required=True, type=Path)
    parser.add_argument("--reference-mesh", required=True, type=Path)
    parser.add_argument("--landmarks-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shapes", default="rect,ellipse")
    parser.add_argument("--margin-fractions", default="0.06,0.12,0.18,0.24")
    parser.add_argument("--taubin-iters", default="0,5,15,30,60")
    parser.add_argument("--subdivide-iters", default="0,1")
    parser.add_argument("--decimate-fractions", default="1.0")
    parser.add_argument("--save-top-n", type=int, default=8)
    parser.add_argument("--avatar-top-n", type=int, default=6)
    parser.add_argument("--photo-3ddfa-mesh", type=Path)
    parser.add_argument("--photo-mediapipe-mesh", type=Path)
    parser.add_argument("--bfm-pkl", type=Path)
    parser.add_argument("--source-fraction-below-nose", type=float, default=0.35)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.output_dir / "candidate_meshes"
    mesh_dir.mkdir(exist_ok=True)

    source_mesh = load_mesh(args.source_broad_mesh)
    reference_vertices = finite_vertices(load_mesh(args.reference_mesh))
    metadata = json.loads(args.landmarks_metadata.read_text(encoding="utf-8"))
    landmarks = {name: np.asarray(point, dtype=np.float64) for name, point in metadata["landmarks"].items()}

    shapes = [value.strip() for value in args.shapes.split(",") if value.strip()]
    margins = parse_float_list(args.margin_fractions)
    taubin_iters = parse_int_list(args.taubin_iters)
    subdivide_iters = parse_int_list(args.subdivide_iters)
    decimate_fractions = parse_float_list(args.decimate_fractions)

    rows = []
    candidate_meshes: dict[str, trimesh.Trimesh] = {}
    for shape in shapes:
        for margin in margins:
            try:
                cropped = crop_by_landmarks(source_mesh, landmarks, shape, margin)
            except Exception as exc:
                rows.append(
                    {
                        "candidate_id": f"{shape}_m{margin:.2f}_crop_failed",
                        "shape": shape,
                        "margin_fraction": margin,
                        "pass_mri_only_gates": False,
                        "error": str(exc),
                    }
                )
                continue
            for taubin in taubin_iters:
                for subdivide in subdivide_iters:
                    for decimate in decimate_fractions:
                        spec = CandidateSpec(shape, margin, taubin, subdivide, decimate)
                        candidate_id = (
                            f"{shape}_m{margin:.2f}_t{taubin}_sub{subdivide}_dec{decimate:.2f}"
                            .replace(".", "p")
                        )
                        try:
                            mesh = process_mesh(cropped, spec)
                            row = mri_only_row(
                                candidate_id,
                                spec,
                                mesh,
                                landmarks,
                                reference_vertices,
                                finite_vertices(cropped),
                            )
                            rows.append(row)
                            if row["pass_mri_only_gates"]:
                                candidate_meshes[candidate_id] = mesh
                        except Exception as exc:
                            rows.append(
                                {
                                    "candidate_id": candidate_id,
                                    "shape": shape,
                                    "margin_fraction": margin,
                                    "taubin_iters": taubin,
                                    "subdivide_iters": subdivide,
                                    "decimate_fraction": decimate,
                                    "pass_mri_only_gates": False,
                                    "error": str(exc),
                                }
                            )

    qc_csv = args.output_dir / "mri_candidate_qc.csv"
    write_rows(qc_csv, rows)
    plot_qc(rows, args.output_dir / "mri_candidate_qc_scatter.png")

    passing = [row for row in rows if row.get("pass_mri_only_gates") and row["candidate_id"] in candidate_meshes]
    passing_sorted = sorted(passing, key=lambda row: float(row["mri_only_score"]))
    for row in passing_sorted[: args.save_top_n]:
        candidate_id = str(row["candidate_id"])
        path = mesh_dir / f"{candidate_id}.ply"
        candidate_meshes[candidate_id].export(path)
        row["saved_mesh"] = str(path)

    avatar_rows = []
    method_disagreement = None
    if args.photo_3ddfa_mesh and args.photo_mediapipe_mesh and args.bfm_pkl:
        target_landmarks_2d = {name: landmarks[name][[0, 2]] for name in SUPPORT_NAMES}
        tddfa_keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)

        vertices_3ddfa, faces_3ddfa = load_geometry(args.photo_3ddfa_mesh)
        vertices_mp, faces_mp = load_geometry(args.photo_mediapipe_mesh)
        lm_3ddfa = extract_source_landmarks(vertices_3ddfa, "3ddfa_v2", tddfa_keypoint_ids)
        lm_mp = extract_source_landmarks(vertices_mp, "mediapipe", None)
        avatar_3ddfa = avatar_alignment(
            vertices_3ddfa,
            faces_3ddfa,
            lm_3ddfa,
            target_landmarks_2d,
            args.source_fraction_below_nose,
        )
        avatar_mp = avatar_alignment(
            vertices_mp,
            faces_mp,
            lm_mp,
            target_landmarks_2d,
            args.source_fraction_below_nose,
        )
        method_disagreement = avatar_disagreement_metrics(avatar_3ddfa, avatar_mp, resolution_mm=1.0)
        (args.output_dir / "avatar_method_disagreement.json").write_text(
            json.dumps(method_disagreement, indent=2),
            encoding="utf-8",
        )

        for row in passing_sorted[: args.avatar_top_n]:
            candidate_id = str(row["candidate_id"])
            mesh = candidate_meshes[candidate_id]
            target_vertices = finite_vertices(mesh)
            for method, avatar in (("3ddfa_v2", avatar_3ddfa), ("mediapipe", avatar_mp)):
                metrics = avatar_vs_target_metrics(target_vertices, target_landmarks_2d, avatar, resolution_mm=1.0)
                avatar_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "method": method,
                        "mri_only_score": row["mri_only_score"],
                        "pass_mri_only_gates": row["pass_mri_only_gates"],
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "boundary_hd95_mm": metrics["boundary_hd95_mm"],
                        "boundary_assd_mm": metrics["boundary_assd_mm"],
                        "source_area_mm2": metrics["source_area_mm2"],
                        "target_area_mm2": metrics["target_area_mm2"],
                        "scale_2d": metrics["scale_2d"],
                        "landmark_rmse_mm": metrics["landmark_rmse_mm"],
                        "landmark_residuals_json": metrics["landmark_residuals_json"],
                    }
                )
        write_rows(args.output_dir / "avatar_diagnostics.csv", avatar_rows)
        order_rows = []
        rows_by_candidate: dict[str, dict[str, dict]] = {}
        for row in avatar_rows:
            rows_by_candidate.setdefault(str(row["candidate_id"]), {})[str(row["method"])] = row
        mri_rows_by_id = {str(row["candidate_id"]): row for row in passing_sorted}
        method_hd95 = float(method_disagreement["boundary_hd95_mm"])
        method_assd = float(method_disagreement["boundary_assd_mm"])
        for candidate_id, by_method in rows_by_candidate.items():
            if "3ddfa_v2" not in by_method or "mediapipe" not in by_method:
                continue
            mri_row = mri_rows_by_id[candidate_id]
            hd95_3ddfa = float(by_method["3ddfa_v2"]["boundary_hd95_mm"])
            hd95_mp = float(by_method["mediapipe"]["boundary_hd95_mm"])
            assd_3ddfa = float(by_method["3ddfa_v2"]["boundary_assd_mm"])
            assd_mp = float(by_method["mediapipe"]["boundary_assd_mm"])
            hd95_signal = abs(hd95_3ddfa - hd95_mp)
            assd_signal = abs(assd_3ddfa - assd_mp)
            target_hd95 = float(mri_row["to_reference_hd95_mm"])
            order_rows.append(
                {
                    "candidate_id": candidate_id,
                    "mri_only_score": mri_row["mri_only_score"],
                    "target_to_reference_hd95_mm": target_hd95,
                    "target_to_unsmoothed_crop_hd95_mm": mri_row["to_unsmoothed_crop_hd95_mm"],
                    "method_disagreement_hd95_mm": method_hd95,
                    "method_disagreement_assd_mm": method_assd,
                    "3ddfa_to_mri_hd95_mm": hd95_3ddfa,
                    "mediapipe_to_mri_hd95_mm": hd95_mp,
                    "method_to_mri_hd95_delta_mm": hd95_signal,
                    "method_to_mri_hd95_delta_over_method_disagreement": hd95_signal / max(method_hd95, 1e-12),
                    "method_to_mri_hd95_delta_over_target_uncertainty": hd95_signal / max(target_hd95, 1e-12),
                    "3ddfa_to_mri_assd_mm": assd_3ddfa,
                    "mediapipe_to_mri_assd_mm": assd_mp,
                    "method_to_mri_assd_delta_mm": assd_signal,
                }
            )
        write_rows(args.output_dir / "distance_order_summary.csv", order_rows)

    summary = {
        "source_broad_mesh": str(args.source_broad_mesh),
        "reference_mesh": str(args.reference_mesh),
        "landmarks_metadata": str(args.landmarks_metadata),
        "n_candidates": len(rows),
        "n_passing_mri_only_gates": len(passing_sorted),
        "top_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "mri_only_score": row["mri_only_score"],
                "shape": row["shape"],
                "margin_fraction": row["margin_fraction"],
                "taubin_iters": row["taubin_iters"],
                "subdivide_iters": row["subdivide_iters"],
                "nose_prominence_nearest_xz_mm": row["nose_prominence_nearest_xz_mm"],
                "roughness_median_mm": row["roughness_median_mm"],
                "stripe_highfreq_ratio": row["stripe_highfreq_ratio"],
                "to_reference_hd95_mm": row["to_reference_hd95_mm"],
                "to_unsmoothed_crop_hd95_mm": row["to_unsmoothed_crop_hd95_mm"],
            }
            for row in passing_sorted[: args.save_top_n]
        ],
        "avatar_method_disagreement": method_disagreement,
        "qc_csv": str(qc_csv),
        "avatar_diagnostics_csv": str(args.output_dir / "avatar_diagnostics.csv") if avatar_rows else None,
        "distance_order_summary_csv": str(args.output_dir / "distance_order_summary.csv") if avatar_rows else None,
        "warning": "Exploratory target-QC experiment. Do not rank avatar baselines from these metrics until MRI target selection is frozen.",
    }
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
