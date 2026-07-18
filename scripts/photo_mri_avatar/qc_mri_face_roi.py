"""QC an MRI face ROI mesh against its source outer-head mesh.

The checks are geometric and provenance-based: they do not depend on choosing a
visually pleasing overlay. The intended use is to distinguish a real connected
surface submesh from point-cloud/frontmost-patch artifacts before avatar
metrics are interpreted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(tuple(geom.geometry.values()))
    if not hasattr(geom, "vertices"):
        raise ValueError(f"Cannot read vertices from {path}")
    return geom


def finite_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def connected_component_sizes(faces: np.ndarray, n_vertices: int) -> list[int]:
    if len(faces) == 0:
        return []
    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for a, b, c in np.asarray(faces, dtype=np.int64):
        adjacency[a].extend([b, c])
        adjacency[b].extend([a, c])
        adjacency[c].extend([a, b])

    seen = np.zeros(n_vertices, dtype=bool)
    sizes = []
    for start in range(n_vertices):
        if seen[start] or not adjacency[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def edge_length_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int] | None:
    if len(faces) == 0:
        return None
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return {
        "n_edges": int(len(edges)),
        "median_mm": float(np.median(lengths)),
        "p95_mm": float(np.percentile(lengths, 95)),
        "max_mm": float(np.max(lengths)),
    }


def pca_spread_ratio(vertices: np.ndarray) -> list[float]:
    centered = vertices - vertices.mean(axis=0)
    _u, singular_values, _vh = np.linalg.svd(centered, full_matrices=False)
    return (singular_values / max(float(singular_values[0]), 1e-12)).tolist()


def mesh_block(mesh: trimesh.Trimesh) -> dict:
    vertices = finite_vertices(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") else np.empty((0, 3), dtype=np.int64)
    components = connected_component_sizes(faces, len(vertices))
    largest_fraction = float(components[0] / len(vertices)) if components else 0.0
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_min_xyz": vertices.min(axis=0).tolist(),
        "bounds_max_xyz": vertices.max(axis=0).tolist(),
        "span_xyz_mm": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
        "pca_singular_ratio": pca_spread_ratio(vertices),
        "component_sizes": components[:10],
        "largest_component_fraction": largest_fraction,
        "edge_lengths": edge_length_summary(vertices, faces),
        "area_mm2": float(mesh.area) if hasattr(mesh, "area") and len(faces) else None,
        "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else None,
    }


def source_vertex_provenance(source_vertices: np.ndarray, roi_vertices: np.ndarray) -> dict:
    distances, _indices = cKDTree(source_vertices).query(roi_vertices, k=1, workers=-1)
    return {
        "nearest_source_vertex_median_mm": float(np.median(distances)),
        "nearest_source_vertex_p95_mm": float(np.percentile(distances, 95)),
        "nearest_source_vertex_max_mm": float(np.max(distances)),
        "near_exact_vertex_count_1e_5mm": int(np.sum(distances <= 1e-5)),
        "near_exact_vertex_percent_1e_5mm": float(100.0 * np.mean(distances <= 1e-5)),
    }


def selection_recheck(vertices: np.ndarray, metadata: dict) -> dict | None:
    selection = metadata.get("selection")
    if not selection:
        return None
    required = {"center_x", "center_z", "rx_mm", "rz_mm", "z_min_mm", "z_max_mm", "y_floor_mm"}
    if not required.issubset(selection):
        return {"warning": "selection block exists but lacks ROI-bound keys"}

    ellipse = ((vertices[:, 0] - selection["center_x"]) / selection["rx_mm"]) ** 2 + (
        (vertices[:, 2] - selection["center_z"]) / selection["rz_mm"]
    ) ** 2
    return {
        "selection_type": selection.get("type"),
        "ellipse_max": float(np.max(ellipse)),
        "ellipse_p99": float(np.percentile(ellipse, 99)),
        "violates_ellipse_gt_1p001": int(np.sum(ellipse > 1.001)),
        "z_min_actual": float(np.min(vertices[:, 2])),
        "z_max_actual": float(np.max(vertices[:, 2])),
        "below_metadata_z_min": int(np.sum(vertices[:, 2] < selection["z_min_mm"] - 1e-4)),
        "above_metadata_z_max": int(np.sum(vertices[:, 2] > selection["z_max_mm"] + 1e-4)),
        "below_metadata_y_floor": int(np.sum(vertices[:, 1] < selection["y_floor_mm"] - 1e-4)),
    }


def y_spread_within_xz_bins(vertices: np.ndarray, bin_mm: float) -> dict:
    bins = np.floor(vertices[:, [0, 2]] / bin_mm).astype(np.int64)
    by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key, y_value in zip(map(tuple, bins), vertices[:, 1]):
        by_cell[key].append(float(y_value))
    spreads = np.array([max(values) - min(values) for values in by_cell.values() if len(values) > 1])
    if len(spreads) == 0:
        return {"bin_mm": float(bin_mm), "cells": int(len(by_cell)), "multi_vertex_cells": 0}
    return {
        "bin_mm": float(bin_mm),
        "cells": int(len(by_cell)),
        "multi_vertex_cells": int(len(spreads)),
        "median_spread_mm": float(np.median(spreads)),
        "p95_spread_mm": float(np.percentile(spreads, 95)),
        "max_spread_mm": float(np.max(spreads)),
    }


def landmark_distances(vertices: np.ndarray, metadata: dict) -> dict[str, float]:
    landmarks = metadata.get("landmarks", {})
    if not landmarks:
        return {}
    tree = cKDTree(vertices)
    out = {}
    for name, point in landmarks.items():
        distance, _index = tree.query(np.asarray(point, dtype=np.float64), k=1)
        out[name] = float(distance)
    return out


def anatomical_landmark_qc(vertices: np.ndarray, metadata: dict) -> dict:
    landmarks = metadata.get("landmarks", {})
    required = {"nose_tip", "left_cheek", "right_cheek"}
    if not required.issubset(landmarks):
        return {"warning": "nose/cheek landmarks are unavailable; cannot check nasal prominence"}

    points = {name: np.asarray(point, dtype=np.float64) for name, point in landmarks.items()}
    nose = points["nose_tip"]
    left_cheek = points["left_cheek"]
    right_cheek = points["right_cheek"]
    cheek_mean_y = float((left_cheek[1] + right_cheek[1]) / 2.0)

    out = {
        "nose_y_mm": float(nose[1]),
        "left_cheek_y_mm": float(left_cheek[1]),
        "right_cheek_y_mm": float(right_cheek[1]),
        "nose_prominence_over_cheek_mean_mm": float(nose[1] - cheek_mean_y),
    }
    if "brow_center" in points:
        out["nose_prominence_over_brow_mm"] = float(nose[1] - points["brow_center"][1])
    if "chin" in points:
        out["chin_y_minus_nose_y_mm"] = float(points["chin"][1] - nose[1])

    for radius in (3.0, 5.0, 8.0, 12.0):
        mask = (np.abs(vertices[:, 0] - nose[0]) <= radius) & (np.abs(vertices[:, 2] - nose[2]) <= radius)
        key = f"central_nose_window_xz_radius_{radius:g}mm"
        if not np.any(mask):
            out[key] = {"vertices": 0}
            continue
        y_values = vertices[mask, 1]
        out[key] = {
            "vertices": int(len(y_values)),
            "y_min_mm": float(np.min(y_values)),
            "y_median_mm": float(np.median(y_values)),
            "y_max_mm": float(np.max(y_values)),
            "y_spread_mm": float(np.max(y_values) - np.min(y_values)),
        }
    return out


def verdict(
    roi: dict,
    provenance: dict,
    landmark_to_roi_mm: dict[str, float],
    metadata: dict,
    anatomical_qc: dict,
    min_nose_prominence_mm: float,
) -> dict:
    warnings = []
    is_exact_source_submesh = provenance["near_exact_vertex_percent_1e_5mm"] >= 99.9
    has_faces = roi["faces"] > 0
    is_connected = roi["largest_component_fraction"] >= 0.95
    selection_type = str(metadata.get("selection", {}).get("type", ""))
    contains_chin = landmark_to_roi_mm.get("chin", np.inf) <= 5.0
    nose_prominence = anatomical_qc.get("nose_prominence_over_cheek_mean_mm")
    has_nose_prominence = nose_prominence is not None and nose_prominence >= min_nose_prominence_mm
    if not has_faces:
        warnings.append("ROI has no faces; treat as point cloud/frontmost patch, not a surface target.")
    if not is_connected:
        warnings.append("ROI is fragmented; connected-surface assumption is weak.")
    if not is_exact_source_submesh:
        warnings.append("ROI vertices are not an exact subset of the source mesh vertices.")
    if "front_shell" in selection_type:
        warnings.append("ROI uses a front-shell rule; depth is a local MRI surface proxy, not full facial anatomy.")
    if not contains_chin:
        warnings.append("Chin/lower third is outside this ROI; do not use for full-face metrics.")
    if nose_prominence is None:
        warnings.append("Nose prominence could not be checked; do not treat this as an anatomical face target.")
    elif not has_nose_prominence:
        warnings.append(
            "Nose prominence over cheek mean is "
            f"{nose_prominence:.3f} mm (< {min_nose_prominence_mm:.3f} mm); "
            "ROI is too flat to act as an anatomical nose/midface target."
        )
    chin_y_minus_nose_y = anatomical_qc.get("chin_y_minus_nose_y_mm")
    if chin_y_minus_nose_y is not None and chin_y_minus_nose_y > 0:
        warnings.append(
            "Metadata chin proxy is more anterior than nose proxy "
            f"by {chin_y_minus_nose_y:.3f} mm; landmark proxies are anatomically inconsistent."
        )
    real_source_surface_target = bool(is_exact_source_submesh and has_faces and is_connected)
    return {
        "real_source_surface_target": real_source_surface_target,
        "anatomical_nose_midface_target": bool(real_source_surface_target and has_nose_prominence),
        "nose_prominence_threshold_mm": float(min_nose_prominence_mm),
        "full_face_target": bool(real_source_surface_target and contains_chin),
        "warnings": warnings,
    }


def analyze(
    source_mesh: Path,
    roi_mesh: Path,
    metadata_path: Path,
    reference_mesh: Path | None,
    min_nose_prominence_mm: float,
) -> dict:
    source = load_mesh(source_mesh)
    roi = load_mesh(roi_mesh)
    source_vertices = finite_vertices(source)
    roi_vertices = finite_vertices(roi)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selection = metadata.get("selection", {})
    bin_mm = float(selection.get("bin_mm", 2.0))

    roi_block = mesh_block(roi)
    provenance = source_vertex_provenance(source_vertices, roi_vertices)
    landmark_to_roi = landmark_distances(roi_vertices, metadata)
    anatomical_qc = anatomical_landmark_qc(roi_vertices, metadata)
    report = {
        "source_mesh": str(source_mesh),
        "roi_mesh": str(roi_mesh),
        "metadata": str(metadata_path),
        "source": mesh_block(source),
        "roi": roi_block,
        "roi_vertex_provenance": provenance,
        "selection_recheck": selection_recheck(roi_vertices, metadata),
        "roi_y_spread_within_xz_bins": y_spread_within_xz_bins(roi_vertices, bin_mm),
        "landmark_distance_to_roi_mm": landmark_to_roi,
        "anatomical_landmark_qc": anatomical_qc,
        "verdict": verdict(roi_block, provenance, landmark_to_roi, metadata, anatomical_qc, min_nose_prominence_mm),
    }
    if reference_mesh:
        reference = load_mesh(reference_mesh)
        reference_vertices = finite_vertices(reference)
        reference_block = mesh_block(reference)
        reference_provenance = source_vertex_provenance(source_vertices, reference_vertices)
        reference_landmark_to_roi = landmark_distances(reference_vertices, metadata)
        reference_anatomical_qc = anatomical_landmark_qc(reference_vertices, metadata)
        report["reference"] = {
            "mesh": str(reference_mesh),
            "stats": reference_block,
            "vertex_provenance": reference_provenance,
            "landmark_distance_to_roi_mm": reference_landmark_to_roi,
            "anatomical_landmark_qc": reference_anatomical_qc,
            "verdict": verdict(
                reference_block,
                reference_provenance,
                reference_landmark_to_roi,
                metadata,
                reference_anatomical_qc,
                min_nose_prominence_mm,
            ),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True, type=Path)
    parser.add_argument("--roi-mesh", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--reference-mesh", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--min-nose-prominence-mm",
        type=float,
        default=10.0,
        help="Minimum anterior nose-tip prominence over cheek mean required for an anatomical midface target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        args.source_mesh,
        args.roi_mesh,
        args.metadata,
        args.reference_mesh,
        args.min_nose_prominence_mm,
    )
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
