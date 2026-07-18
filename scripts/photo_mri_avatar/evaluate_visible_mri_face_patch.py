"""Evaluate one-photo avatar meshes against the measurable MRI face patch.

This script is deliberately narrower than the older head-surface comparison:
it uses a face-only MRI patch and excludes the lower third when the MRI does not
contain a reliable chin/mouth surface. The output is an internal diagnostic, not
a validated anatomical accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import KDTree


LANDMARK_NAMES = ["nose_tip", "brow_center", "left_cheek", "right_cheek"]
SOURCE_CROP_NAMES = ["nose_tip", "brow_center", "left_cheek", "right_cheek", "chin"]

MEDIAPIPE_LANDMARKS = {
    "nose_tip": [1],
    "brow_center": [168],
    "left_cheek": [234],
    "right_cheek": [454],
    "chin": [152],
}

TDDFA_68_LANDMARKS = {
    "nose_tip": [30],
    "brow_center": [27],
    "left_cheek": [2],
    "right_cheek": [14],
    "chin": [8],
}


def load_vertices(path: Path) -> np.ndarray:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(tuple(geom.geometry.values()))
    if not hasattr(geom, "vertices"):
        raise ValueError(f"Cannot read vertices from {path}")
    vertices = np.asarray(geom.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    if len(keypoints) != 204:
        raise ValueError(f"Expected 204 flattened 68-landmark entries, got {len(keypoints)}")
    return keypoints[0::3] // 3


def extract_source_landmarks(
    vertices: np.ndarray,
    method: str,
    tddfa_keypoint_ids: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if method == "mediapipe":
        mapping = MEDIAPIPE_LANDMARKS
        dense_vertices = vertices
    elif method == "3ddfa_v2":
        if tddfa_keypoint_ids is None:
            raise ValueError("--bfm-pkl is required for 3ddfa_v2")
        mapping = TDDFA_68_LANDMARKS
        dense_vertices = vertices[tddfa_keypoint_ids]
    else:
        raise ValueError(method)

    return {
        name: dense_vertices[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for name, indices in mapping.items()
    }


def maybe_swap_lr(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swapped = dict(landmarks)
    swapped["left_cheek"], swapped["right_cheek"] = landmarks["right_cheek"], landmarks["left_cheek"]
    return swapped


def similarity_from_points(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
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


def similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
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


def apply_similarity_2d(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ points.T)).T + t


def source_to_mri_axis_frame(points: np.ndarray, x_sign: int, depth_sign: int) -> np.ndarray:
    """Map camera/image mesh axes into the MRI face axis convention.

    Current photo baselines store x as image-horizontal, y as image-vertical,
    and z as camera depth where the nose is more anterior than the cheeks.
    MRI world coordinates use x as left-right, y as anterior-posterior, and z
    as inferior-superior.
    """
    return np.column_stack((x_sign * points[:, 0], depth_sign * points[:, 2], points[:, 1]))


def source_to_mri_front_locked(
    points: np.ndarray,
    r_front: np.ndarray,
    scale: float,
    t_front: np.ndarray,
    source_nose: np.ndarray,
    mri_nose: np.ndarray,
    depth_sign: int,
) -> np.ndarray:
    """Use front-plane similarity for x/z and carry source depth with that scale.

    This avoids estimating the global scale from avatar depth, which is not a
    metric quantity for the current one-photo baselines.
    """
    front = apply_similarity_2d(points[:, [0, 1]], r_front, scale, t_front)
    depth = mri_nose[1] + depth_sign * scale * (points[:, 2] - source_nose[2])
    return np.column_stack((front[:, 0], depth, front[:, 1]))


def scale_translation_from_points(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    denom = float(np.sum(src_c * src_c))
    scale = float(np.sum(src_c * dst_c) / max(denom, 1e-12))
    if scale <= 0:
        scale = abs(scale)
    t = dst_mean - scale * src_mean
    return scale, t


def cheek_width(points: dict[str, np.ndarray]) -> float:
    return float(np.linalg.norm(points["right_cheek"] - points["left_cheek"]))


def scale_translation_with_anchor(
    src_landmarks: dict[str, np.ndarray],
    dst_landmarks: dict[str, np.ndarray],
    scale_anchor: str,
) -> tuple[float, np.ndarray, float, float]:
    src = np.vstack([src_landmarks[name] for name in LANDMARK_NAMES])
    dst = np.vstack([dst_landmarks[name] for name in LANDMARK_NAMES])
    if scale_anchor == "least_squares_landmarks":
        scale, t = scale_translation_from_points(src, dst)
        src_anchor = cheek_width(src_landmarks)
        dst_anchor = cheek_width(dst_landmarks)
        return scale, t, src_anchor, dst_anchor
    if scale_anchor == "cheek_width":
        src_anchor = cheek_width(src_landmarks)
        dst_anchor = cheek_width(dst_landmarks)
        scale = dst_anchor / max(src_anchor, 1e-12)
        t = (dst - scale * src).mean(axis=0)
        return float(scale), t, src_anchor, dst_anchor
    raise ValueError(scale_anchor)


def apply_scale_translation(points: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * points + t


def summarize_distances(distances: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_mm": float(np.mean(distances)),
        f"{prefix}_median_mm": float(np.median(distances)),
        f"{prefix}_p90_mm": float(np.percentile(distances, 90)),
        f"{prefix}_p95_mm": float(np.percentile(distances, 95)),
        f"{prefix}_max_mm": float(np.max(distances)),
    }


def metric_block(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    s2t, _ = KDTree(target).query(source)
    t2s, _ = KDTree(source).query(target)
    out = {}
    out.update(summarize_distances(s2t, "s2t"))
    out.update(summarize_distances(t2s, "t2s"))
    out["assd_mean_mm"] = float((np.mean(s2t) + np.mean(t2s)) / 2.0)
    out["hd95_bidir_mm"] = float(max(np.percentile(s2t, 95), np.percentile(t2s, 95)))
    out["chamfer_l2_mm2"] = float(np.mean(s2t**2) + np.mean(t2s**2))
    return out


def crop_to_mri_visible_region(
    aligned_source: np.ndarray,
    target_region: np.ndarray,
    z_min: float,
    margin_mm: float,
) -> np.ndarray:
    x_min, y_min, z_region_min = target_region.min(axis=0) - margin_mm
    x_max, y_max, z_max = target_region.max(axis=0) + margin_mm
    z_floor = max(z_min, z_region_min)
    keep = (
        (aligned_source[:, 0] >= x_min)
        & (aligned_source[:, 0] <= x_max)
        & (aligned_source[:, 1] >= y_min)
        & (aligned_source[:, 1] <= y_max)
        & (aligned_source[:, 2] >= z_floor)
        & (aligned_source[:, 2] <= z_max)
    )
    if keep.sum() < 50:
        keep = (
            (aligned_source[:, 0] >= x_min)
            & (aligned_source[:, 0] <= x_max)
            & (aligned_source[:, 2] >= z_floor)
            & (aligned_source[:, 2] <= z_max)
        )
    return aligned_source[keep]


def crop_source_anatomical(
    source: np.ndarray,
    source_landmarks: dict[str, np.ndarray],
    vertical_fraction_below_nose: float,
    cheek_margin_fraction: float,
) -> np.ndarray:
    """Crop source in its native camera frame before MRI alignment.

    Both current baselines store x as image-horizontal and y as image-vertical
    with larger y roughly superior. We avoid using source z/depth for region
    selection because 3DDFA and MediaPipe use different depth conventions.
    """
    nose = source_landmarks["nose_tip"]
    brow = source_landmarks["brow_center"]
    chin = source_landmarks["chin"]
    left = source_landmarks["left_cheek"]
    right = source_landmarks["right_cheek"]

    face_height = abs(float(brow[1] - chin[1]))
    cheek_width = abs(float(right[0] - left[0]))
    vertical_min = float(nose[1] - vertical_fraction_below_nose * face_height)
    x_min = min(float(left[0]), float(right[0])) - cheek_margin_fraction * cheek_width
    x_max = max(float(left[0]), float(right[0])) + cheek_margin_fraction * cheek_width

    keep = (
        (source[:, 1] >= vertical_min)
        & (source[:, 0] >= x_min)
        & (source[:, 0] <= x_max)
    )
    if keep.sum() < 50:
        keep = source[:, 1] >= vertical_min
    return source[keep]


def write_preview(
    target_region: np.ndarray,
    source_region: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    source_landmarks: dict[str, np.ndarray],
    output: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12, 4), dpi=160)
    views = [
        (0, -90, "front x/z"),
        (0, 0, "side y/z"),
        (90, -90, "top x/y"),
    ]
    mri_lm = np.vstack([mri_landmarks[name] for name in LANDMARK_NAMES])
    src_lm = np.vstack([source_landmarks[name] for name in LANDMARK_NAMES])
    for idx, (elev, azim, view_title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(target_region[:, 0], target_region[:, 1], target_region[:, 2], s=1, c="#2f80ed", alpha=0.65)
        ax.scatter(source_region[:, 0], source_region[:, 1], source_region[:, 2], s=1, c="#d62728", alpha=0.65)
        ax.scatter(mri_lm[:, 0], mri_lm[:, 1], mri_lm[:, 2], s=26, c="#0b3d91", alpha=1.0)
        ax.scatter(src_lm[:, 0], src_lm[:, 1], src_lm[:, 2], s=22, c="#ff8c00", alpha=1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_title)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_locked_2d_preview(
    target_region: np.ndarray,
    source_region: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    source_landmarks: dict[str, np.ndarray],
    output: Path,
    title: str,
    scale: float,
    landmark_rmse_mm: float,
    hd95_mm: float,
) -> None:
    views = [
        ("front x/z", 0, 2),
        ("side y/z", 1, 2),
        ("top x/y", 0, 1),
    ]
    all_points = np.vstack([target_region, source_region])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = maxs - mins
    margin = np.maximum(span * 0.08, 5.0)
    mins -= margin
    maxs += margin

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=170)
    for ax, (view_title, a, b) in zip(axes, views):
        ax.scatter(target_region[:, a], target_region[:, b], s=2, c="#2f80ed", alpha=0.45)
        ax.scatter(source_region[:, a], source_region[:, b], s=3, c="#d62728", alpha=0.55)
        for point in mri_landmarks.values():
            ax.scatter([point[a]], [point[b]], s=24, c="#0b3d91", edgecolors="white", linewidths=0.4, zorder=5)
        for point in source_landmarks.values():
            ax.scatter([point[a]], [point[b]], s=22, c="#ff8c00", edgecolors="white", linewidths=0.4, zorder=6)
        ax.set_xlim(mins[a], maxs[a])
        ax.set_ylim(mins[b], maxs[b])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_title(view_title)
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        x0 = mins[a] + 0.08 * (maxs[a] - mins[a])
        y0 = mins[b] + 0.08 * (maxs[b] - mins[b])
        ax.plot([x0, x0 + 20], [y0, y0], c="black", lw=2)
        ax.text(x0, y0 + 2, "20 mm", fontsize=7)
        text = f"scale={scale:.4f}\nLM RMSE={landmark_rmse_mm:.1f} mm\nHD95={hd95_mm:.1f} mm"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output)
    plt.close(fig)


def evaluate_one(
    mesh_path: Path,
    method: str,
    target_region: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    tddfa_keypoint_ids: np.ndarray | None,
    args: argparse.Namespace,
) -> dict:
    source = load_vertices(mesh_path)
    source_landmarks = extract_source_landmarks(source, method, tddfa_keypoint_ids)
    candidates = [("normal", source_landmarks), ("lr_swapped", maybe_swap_lr(source_landmarks))]

    best = None
    for orientation, candidate_landmarks in candidates:
        dst_lm = np.vstack([mri_landmarks[name] for name in LANDMARK_NAMES])
        source_native_region = crop_source_anatomical(
            source,
            candidate_landmarks,
            vertical_fraction_below_nose=args.source_vertical_fraction_below_nose,
            cheek_margin_fraction=args.source_cheek_margin_fraction,
        )

        axis_candidates: list[tuple[str, np.ndarray, np.ndarray, float, np.ndarray]] = []
        if args.alignment_mode == "similarity":
            src_lm = np.vstack([candidate_landmarks[name] for name in LANDMARK_NAMES])
            r, scale, t = similarity_from_points(src_lm, dst_lm)
            axis_candidates.append(("free_similarity", apply_similarity(source_native_region, r, scale, t), {
                name: apply_similarity(candidate_landmarks[name].reshape(1, 3), r, scale, t).reshape(3)
                for name in LANDMARK_NAMES
            }, scale, t))
        elif args.alignment_mode == "front_locked_similarity_nose_anchor":
            src_lm_front = np.vstack([candidate_landmarks[name][[0, 1]] for name in LANDMARK_NAMES])
            dst_lm_front = np.vstack([mri_landmarks[name][[0, 2]] for name in LANDMARK_NAMES])
            r_front, scale, _t_front = similarity_2d(src_lm_front, dst_lm_front)
            if args.front_scale_override is not None:
                scale = args.front_scale_override
            t_front = (
                mri_landmarks["nose_tip"][[0, 2]]
                - scale * (r_front @ candidate_landmarks["nose_tip"][[0, 1]])
            )
            for depth_sign in (-1, 1):
                aligned_lm = {
                    name: source_to_mri_front_locked(
                        candidate_landmarks[name].reshape(1, 3),
                        r_front,
                        scale,
                        t_front,
                        candidate_landmarks["nose_tip"],
                        mri_landmarks["nose_tip"],
                        depth_sign=depth_sign,
                    ).reshape(3)
                    for name in LANDMARK_NAMES
                }
                axis_candidates.append(
                    (
                        f"front_locked_depth{depth_sign}",
                        source_to_mri_front_locked(
                            source_native_region,
                            r_front,
                            scale,
                            t_front,
                            candidate_landmarks["nose_tip"],
                            mri_landmarks["nose_tip"],
                            depth_sign=depth_sign,
                        ),
                        aligned_lm,
                        scale,
                        np.array([t_front[0], 0.0, t_front[1]], dtype=np.float64),
                    )
                )
        else:
            for x_sign in (-1, 1):
                for depth_sign in (-1, 1):
                    src_lm_axis = source_to_mri_axis_frame(
                        np.vstack([candidate_landmarks[name] for name in LANDMARK_NAMES]),
                        x_sign=x_sign,
                        depth_sign=depth_sign,
                    )
                    scale, t = scale_translation_from_points(src_lm_axis, dst_lm)
                    source_axis = source_to_mri_axis_frame(source_native_region, x_sign=x_sign, depth_sign=depth_sign)
                    aligned_lm = {}
                    for name in LANDMARK_NAMES:
                        point_axis = source_to_mri_axis_frame(
                            candidate_landmarks[name].reshape(1, 3),
                            x_sign=x_sign,
                            depth_sign=depth_sign,
                        )
                        aligned_lm[name] = apply_scale_translation(point_axis, scale, t).reshape(3)
                    axis_candidates.append(
                        (
                            f"axis_x{x_sign}_depth{depth_sign}",
                            apply_scale_translation(source_axis, scale, t),
                            aligned_lm,
                            scale,
                            t,
                        )
                    )

        for frame_label, source_region, aligned_lm, scale, _t in axis_candidates:
            residuals = np.linalg.norm(
                np.vstack([aligned_lm[name] for name in LANDMARK_NAMES]) - dst_lm,
                axis=1,
            )
            # Keep only the broad target x/z range after anatomical source cropping.
            # Do not filter by target y/depth: the MRI patch is a frontmost surface,
            # while the avatar methods have method-specific depth conventions.
            x_min, _y_min, z_min = target_region.min(axis=0) - args.region_margin_mm
            x_max, _y_max, z_max = target_region.max(axis=0) + args.region_margin_mm
            keep = (
                (source_region[:, 0] >= x_min)
                & (source_region[:, 0] <= x_max)
                & (source_region[:, 2] >= z_min)
                & (source_region[:, 2] <= z_max)
            )
            if keep.sum() >= 50:
                source_region = source_region[keep]
            if len(source_region) < 50:
                continue
            metrics = metric_block(source_region, target_region)
            score = float(np.sqrt(np.mean(residuals**2)) + 0.1 * metrics["hd95_bidir_mm"])
            row = {
                "method": method,
                "mesh": str(mesh_path),
                "mesh_name": mesh_path.name,
                "orientation": f"{orientation}:{frame_label}",
                "scale": float(scale),
                "n_source_region": int(len(source_region)),
                "n_target_region": int(len(target_region)),
                "landmark_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
                "landmark_median_mm": float(np.median(residuals)),
                "landmark_max_mm": float(np.max(residuals)),
                "landmark_residuals_json": json.dumps(
                    {name: float(value) for name, value in zip(LANDMARK_NAMES, residuals)}
                ),
                "score": score,
                **metrics,
                "_source_region": source_region,
                "_aligned_landmarks": aligned_lm,
            }
            if best is None or row["score"] < best["score"]:
                best = row

    if best is None:
        raise ValueError(f"No usable source region after alignment for {mesh_path}")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-patch-mesh", required=True, type=Path)
    parser.add_argument("--mri-patch-metadata", required=True, type=Path)
    parser.add_argument("--photo-mesh-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["mediapipe", "3ddfa_v2"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--bfm-pkl", type=Path)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--nose-drop-mm", type=float, default=18.0)
    parser.add_argument("--region-margin-mm", type=float, default=6.0)
    parser.add_argument("--source-vertical-fraction-below-nose", type=float, default=0.25)
    parser.add_argument("--source-cheek-margin-fraction", type=float, default=0.08)
    parser.add_argument("--front-scale-override", type=float, default=None)
    parser.add_argument(
        "--alignment-mode",
        choices=["similarity", "axis_scale_translation", "front_locked_similarity_nose_anchor"],
        default="similarity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mri_points = load_vertices(args.mri_patch_mesh)
    metadata = json.loads(args.mri_patch_metadata.read_text(encoding="utf-8"))
    mri_landmarks = {
        name: np.asarray(metadata["landmarks"][name], dtype=np.float64)
        for name in LANDMARK_NAMES
    }
    nose_z = float(metadata["landmarks"]["nose_tip"][2])
    if args.z_min is None:
        args.z_min = nose_z - args.nose_drop_mm

    target_region = mri_points[mri_points[:, 2] >= args.z_min]
    if len(target_region) < 500:
        raise ValueError(f"Too few MRI target points after z_min={args.z_min}: {len(target_region)}")

    tddfa_keypoint_ids = None
    if args.method == "3ddfa_v2":
        if args.bfm_pkl is None:
            raise ValueError("--bfm-pkl is required for 3ddfa_v2")
        tddfa_keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)

    rows = []
    for index, mesh_path in enumerate(sorted(args.photo_mesh_dir.glob(args.pattern)), start=1):
        row = evaluate_one(mesh_path, args.method, target_region, mri_landmarks, tddfa_keypoint_ids, args)
        preview = args.output_dir / f"{index:02d}_{args.method}_visible_mri_patch.png"
        write_preview(
            target_region,
            row["_source_region"],
            mri_landmarks,
            row["_aligned_landmarks"],
            preview,
            f"{args.method}: MRI-visible upper/mid face only",
        )
        row["preview"] = str(preview)
        row.pop("_source_region")
        row.pop("_aligned_landmarks")
        rows.append(row)

    fieldnames = [
        "method",
        "mesh_name",
        "mesh",
        "orientation",
        "scale",
        "n_source_region",
        "n_target_region",
        "landmark_rmse_mm",
        "landmark_median_mm",
        "landmark_max_mm",
        "score",
        "s2t_mean_mm",
        "s2t_median_mm",
        "s2t_p90_mm",
        "s2t_p95_mm",
        "s2t_max_mm",
        "t2s_mean_mm",
        "t2s_median_mm",
        "t2s_p90_mm",
        "t2s_p95_mm",
        "t2s_max_mm",
        "assd_mean_mm",
        "hd95_bidir_mm",
        "chamfer_l2_mm2",
        "landmark_residuals_json",
        "preview",
    ]
    csv_path = args.output_dir / "visible_mri_patch_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    json_path = args.output_dir / "visible_mri_patch_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    run_info = {
        "mri_patch_mesh": str(args.mri_patch_mesh),
        "mri_patch_metadata": str(args.mri_patch_metadata),
        "method": args.method,
        "pattern": args.pattern,
        "z_min": args.z_min,
        "nose_drop_mm": args.nose_drop_mm,
        "region_margin_mm": args.region_margin_mm,
        "source_vertical_fraction_below_nose": args.source_vertical_fraction_below_nose,
        "source_cheek_margin_fraction": args.source_cheek_margin_fraction,
        "front_scale_override": args.front_scale_override,
        "alignment_mode": args.alignment_mode,
        "n_target_region": int(len(target_region)),
        "excluded_region": "lower third / unreliable chin-mouth MRI area",
        "csv": str(csv_path),
        "json": str(json_path),
        "n": len(rows),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
