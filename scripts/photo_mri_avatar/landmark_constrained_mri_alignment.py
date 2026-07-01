"""Landmark-constrained alignment of photo face meshes to an MRI head surface.

This is stricter than unconstrained ICP: a similarity transform is estimated
from sparse semantic landmarks first, then source-to-MRI distances are measured
without letting ICP freely slide the face over the MRI cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import KDTree


LANDMARK_NAMES = ["nose_tip", "chin", "brow_center", "left_cheek", "right_cheek"]

MEDIAPIPE_LANDMARKS = {
    # MediaPipe Face Landmarker vertex ids.
    "nose_tip": [1],
    "chin": [152],
    "brow_center": [168],
    "left_cheek": [234],
    "right_cheek": [454],
}

TDDFA_68_LANDMARKS = {
    # 0-based iBUG-68 ids reconstructed by 3DDFA_V2.
    "nose_tip": [30],
    "chin": [8],
    "brow_center": [27],
    "left_cheek": [2],
    "right_cheek": [14],
}


def load_vertices(path: Path, max_points: int | None = None, seed: int = 42) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if max_points and len(vertices) > max_points:
        rng = np.random.default_rng(seed)
        vertices = vertices[rng.choice(len(vertices), max_points, replace=False)]
    return vertices


def summarize_distances(distances: np.ndarray) -> dict[str, float]:
    return {
        "mean_mm": float(np.mean(distances)),
        "median_mm": float(np.median(distances)),
        "p75_mm": float(np.percentile(distances, 75)),
        "p90_mm": float(np.percentile(distances, 90)),
        "max_mm": float(np.max(distances)),
    }


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


def robust_similarity(
    source_landmarks: dict[str, np.ndarray],
    target_landmarks: dict[str, np.ndarray],
    min_points: int = 4,
) -> tuple[np.ndarray, float, np.ndarray, list[str], list[str], np.ndarray]:
    """Fit on the landmark subset with the lowest leave-one-out residual."""
    names = [name for name in LANDMARK_NAMES if name in source_landmarks and name in target_landmarks]
    if len(names) < min_points:
        raise ValueError(f"Need at least {min_points} paired landmarks, got {names}")

    best = None
    subsets = [names]
    if len(names) > min_points:
        subsets.extend(list(combinations(names, len(names) - 1)))
    for subset in subsets:
        subset = list(subset)
        src = np.vstack([source_landmarks[name] for name in subset])
        dst = np.vstack([target_landmarks[name] for name in subset])
        r, scale, t = similarity_from_points(src, dst)
        all_src = np.vstack([source_landmarks[name] for name in names])
        all_dst = np.vstack([target_landmarks[name] for name in names])
        residuals = np.linalg.norm(apply_similarity(all_src, r, scale, t) - all_dst, axis=1)
        score = float(np.median(residuals) + 0.25 * np.percentile(residuals, 90))
        candidate = (score, r, scale, t, subset, residuals)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    _score, r, scale, t, subset, residuals = best
    return r, scale, t, subset, names, residuals


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    if len(keypoints) != 204:
        raise ValueError(f"Expected 204 flattened 68-landmark entries, got {len(keypoints)}")
    return keypoints[0::3] // 3


def extract_source_landmarks(vertices: np.ndarray, method: str, tddfa_keypoint_ids: np.ndarray | None) -> dict[str, np.ndarray]:
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

    landmarks = {}
    for name, indices in mapping.items():
        landmarks[name] = dense_vertices[np.asarray(indices, dtype=np.int64)].mean(axis=0)
    return landmarks


def mri_proxy_landmarks(mri_points: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate rough face landmarks from a LAS-oriented MRI outer-head mesh."""
    x, y, z = mri_points[:, 0], mri_points[:, 1], mri_points[:, 2]
    x_center = np.median(x)
    x_span = np.percentile(x, 99) - np.percentile(x, 1)
    z_percentiles = np.percentile(z, [10, 20, 35, 50, 65, 80, 90])

    def pick_max_y(mask: np.ndarray, fallback_name: str) -> np.ndarray:
        indices = np.where(mask)[0]
        if len(indices) < 20:
            raise ValueError(f"Too few candidates for MRI landmark {fallback_name}: {len(indices)}")
        return mri_points[indices[np.argmax(y[indices])]]

    central = np.abs(x - x_center) <= 0.14 * x_span
    near_left = x <= x_center - 0.18 * x_span
    near_right = x >= x_center + 0.18 * x_span
    side_band = (z >= z_percentiles[2]) & (z <= z_percentiles[4])

    landmarks = {
        "nose_tip": pick_max_y(
            central & (z >= z_percentiles[2]) & (z <= z_percentiles[5]),
            "nose_tip",
        ),
        "chin": pick_max_y(
            central & (z >= z_percentiles[0]) & (z <= z_percentiles[2]) & (y >= np.percentile(y, 65)),
            "chin",
        ),
        "brow_center": pick_max_y(
            central & (z >= z_percentiles[4]) & (z <= z_percentiles[6]),
            "brow_center",
        ),
        "left_cheek": pick_max_y(
            near_left & side_band & (y >= np.percentile(y, 60)),
            "left_cheek",
        ),
        "right_cheek": pick_max_y(
            near_right & side_band & (y >= np.percentile(y, 60)),
            "right_cheek",
        ),
    }
    return landmarks


def maybe_swap_lr(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swapped = dict(landmarks)
    swapped["left_cheek"], swapped["right_cheek"] = landmarks["right_cheek"], landmarks["left_cheek"]
    return swapped


def write_landmark_preview(
    target: np.ndarray,
    source_aligned: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    source_landmarks_aligned: dict[str, np.ndarray],
    output: Path,
) -> None:
    if len(target) > 20000:
        rng = np.random.default_rng(42)
        target = target[rng.choice(len(target), 20000, replace=False)]
    if len(source_aligned) > 7000:
        rng = np.random.default_rng(43)
        source_aligned = source_aligned[rng.choice(len(source_aligned), 7000, replace=False)]

    fig = plt.figure(figsize=(10, 4), dpi=160)
    views = [(0, -90, "xy"), (0, 0, "xz"), (90, -90, "top")]
    mri_lm = np.vstack([mri_landmarks[name] for name in LANDMARK_NAMES])
    src_lm = np.vstack([source_landmarks_aligned[name] for name in LANDMARK_NAMES])
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=0.1, c="#9cc3e6", alpha=0.25)
        ax.scatter(source_aligned[:, 0], source_aligned[:, 1], source_aligned[:, 2], s=0.7, c="#d62728", alpha=0.55)
        ax.scatter(mri_lm[:, 0], mri_lm[:, 1], mri_lm[:, 2], s=24, c="#1f77b4", alpha=1.0)
        ax.scatter(src_lm[:, 0], src_lm[:, 1], src_lm[:, 2], s=18, c="#ff7f0e", alpha=1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def compare_one(
    photo_mesh: Path,
    method: str,
    mri_points: np.ndarray,
    mri_cap: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    tddfa_keypoint_ids: np.ndarray | None,
    args: argparse.Namespace,
) -> dict:
    source = load_vertices(photo_mesh, max_points=args.source_sample)
    source_for_landmarks = load_vertices(photo_mesh)
    source_landmarks = extract_source_landmarks(source_for_landmarks, method, tddfa_keypoint_ids)

    candidates = [("normal", source_landmarks)]
    if args.allow_lr_swap:
        candidates.append(("lr_swapped", maybe_swap_lr(source_landmarks)))

    best = None
    for orientation, oriented_landmarks in candidates:
        r, scale, t, fit_landmarks, residual_names, landmark_residuals = robust_similarity(
            oriented_landmarks,
            mri_landmarks,
        )
        aligned = apply_similarity(source, r, scale, t)
        aligned_landmarks = {
            name: apply_similarity(point.reshape(1, 3), r, scale, t).reshape(3)
            for name, point in oriented_landmarks.items()
        }
        distances, _indices = KDTree(mri_cap).query(aligned)
        stats = summarize_distances(distances)
        score = float(np.median(landmark_residuals) + 0.25 * np.percentile(landmark_residuals, 90) + 0.1 * stats["median_mm"])
        candidate = {
            "method": method,
            "photo_mesh": str(photo_mesh),
            "source_vertices": int(len(source)),
            "orientation": orientation,
            "scale": float(scale),
            "fit_landmarks": fit_landmarks,
            "landmark_residuals_json": json.dumps(
                {name: float(value) for name, value in zip(residual_names, landmark_residuals)}
            ),
            "landmark_rmse_mm": float(np.sqrt(np.mean(landmark_residuals**2))),
            "landmark_median_mm": float(np.median(landmark_residuals)),
            "landmark_p90_mm": float(np.percentile(landmark_residuals, 90)),
            "landmark_max_mm": float(np.max(landmark_residuals)),
            "score": score,
            **stats,
            "_aligned": aligned,
            "_aligned_landmarks": aligned_landmarks,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    assert best is not None
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-mesh", required=True, type=Path)
    parser.add_argument("--photo-mesh-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["mediapipe", "3ddfa_v2"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--bfm-pkl", type=Path, help="3DDFA_V2 bfm_noneck_v3.pkl; required for --method 3ddfa_v2")
    parser.add_argument("--mri-sample", type=int, default=120000)
    parser.add_argument("--source-sample", type=int, default=None)
    parser.add_argument("--front-percentile", type=float, default=62.0)
    parser.add_argument("--allow-lr-swap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mri_points = load_vertices(args.mri_mesh, max_points=args.mri_sample)
    mri_cap = mri_points[mri_points[:, 1] >= np.percentile(mri_points[:, 1], args.front_percentile)]
    mri_landmarks = mri_proxy_landmarks(mri_points)

    tddfa_keypoint_ids = None
    if args.method == "3ddfa_v2":
        if args.bfm_pkl is None:
            raise ValueError("--bfm-pkl is required for --method 3ddfa_v2")
        tddfa_keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl)

    (args.output_dir / "mri_proxy_landmarks.json").write_text(
        json.dumps({name: point.tolist() for name, point in mri_landmarks.items()}, indent=2),
        encoding="utf-8",
    )

    results = []
    for photo_mesh in sorted(args.photo_mesh_dir.glob(args.pattern)):
        result = compare_one(
            photo_mesh,
            args.method,
            mri_points,
            mri_cap,
            mri_landmarks,
            tddfa_keypoint_ids,
            args,
        )
        preview_path = args.output_dir / f"{photo_mesh.stem}_landmark_constrained_alignment.png"
        write_landmark_preview(
            mri_cap,
            result["_aligned"],
            mri_landmarks,
            result["_aligned_landmarks"],
            preview_path,
        )
        result["alignment_preview"] = str(preview_path)
        result.pop("_aligned")
        result.pop("_aligned_landmarks")
        results.append(result)

    fieldnames = [
        "method",
        "photo_mesh",
        "source_vertices",
        "orientation",
        "scale",
        "fit_landmarks",
        "landmark_residuals_json",
        "landmark_rmse_mm",
        "landmark_median_mm",
        "landmark_p90_mm",
        "landmark_max_mm",
        "score",
        "mean_mm",
        "median_mm",
        "p75_mm",
        "p90_mm",
        "max_mm",
        "alignment_preview",
    ]
    csv_path = args.output_dir / "landmark_constrained_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path = args.output_dir / "landmark_constrained_summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "n": len(results)}, indent=2))


if __name__ == "__main__":
    main()
