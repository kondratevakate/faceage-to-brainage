"""Compare photo-derived face meshes against an MRI-derived head surface.

The metric is a first-pass triage score. It performs similarity ICP from the
photo mesh to candidate frontal caps of the MRI mesh and reports nearest-neighbor
distances for the best alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import KDTree


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


def pca_basis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


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


def initial_align_by_pca(src: np.ndarray, dst: np.ndarray, signs: tuple[int, int, int]) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_basis = pca_basis(src)
    dst_basis = pca_basis(dst) @ np.diag(signs)

    src_rms = np.sqrt(np.mean(np.sum((src - src_mean) ** 2, axis=1)))
    dst_rms = np.sqrt(np.mean(np.sum((dst - dst_mean) ** 2, axis=1)))
    scale = dst_rms / max(src_rms, 1e-12)
    r = dst_basis @ src_basis.T
    if np.linalg.det(r) < 0:
        r[:, -1] *= -1
    return apply_similarity(src, r, scale, dst_mean - scale * (r @ src_mean))


def icp_to_target(
    src: np.ndarray,
    target: np.ndarray,
    iterations: int = 40,
    trim_quantile: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(target)
    current = src.copy()
    last_median = np.inf
    for _ in range(iterations):
        distances, indices = tree.query(current)
        keep = distances <= np.quantile(distances, trim_quantile)
        if keep.sum() < 12:
            keep = np.ones_like(distances, dtype=bool)
        r, scale, t = similarity_from_points(current[keep], target[indices[keep]])
        current = apply_similarity(current, r, scale, t)
        median = float(np.median(distances))
        if abs(last_median - median) < 1e-4:
            break
        last_median = median
    distances, _indices = tree.query(current)
    return current, distances


def candidate_caps(
    target: np.ndarray,
    percentile: float,
    front_axis: int | None = None,
    front_sign: int | None = None,
) -> list[dict]:
    caps = []
    axes = [front_axis] if front_axis is not None else list(range(3))
    signs = [front_sign] if front_sign is not None else [-1, 1]
    for axis, sign in product(axes, signs):
        if axis is None or sign is None:
            continue
        score = sign * target[:, axis]
        threshold = np.percentile(score, percentile)
        cap = target[score >= threshold]
        if len(cap) >= 1000:
            caps.append({"axis": axis, "sign": sign, "points": cap})
    return caps


def summarize_distances(distances: np.ndarray) -> dict[str, float]:
    return {
        "mean_mm": float(np.mean(distances)),
        "median_mm": float(np.median(distances)),
        "p75_mm": float(np.percentile(distances, 75)),
        "p90_mm": float(np.percentile(distances, 90)),
        "max_mm": float(np.max(distances)),
    }


def compare_one(photo_mesh: Path, mri_points: np.ndarray, args: argparse.Namespace) -> dict:
    source = load_vertices(photo_mesh, max_points=args.source_sample)
    if len(source) < 12:
        raise ValueError(f"Too few vertices in {photo_mesh}")

    best = None
    for cap in candidate_caps(mri_points, args.front_percentile, args.front_axis, args.front_sign):
        target = cap["points"]
        for signs in product([-1, 1], repeat=3):
            if signs[0] * signs[1] * signs[2] < 0:
                continue
            init = initial_align_by_pca(source, target, signs)
            aligned, distances = icp_to_target(
                init,
                target,
                iterations=args.iterations,
                trim_quantile=args.trim_quantile,
            )
            stats = summarize_distances(distances)
            score = stats["median_mm"] + 0.25 * stats["p90_mm"]
            candidate = {
                "photo_mesh": str(photo_mesh),
                "source_vertices": int(len(source)),
                "target_cap_vertices": int(len(target)),
                "front_axis": int(cap["axis"]),
                "front_sign": int(cap["sign"]),
                "pca_signs": list(signs),
                "score": float(score),
                **stats,
                "_aligned": aligned,
                "_target": target,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    assert best is not None
    return best


def write_alignment_preview(result: dict, output: Path) -> None:
    source = result["_aligned"]
    target = result["_target"]
    if len(target) > 20000:
        rng = np.random.default_rng(42)
        target = target[rng.choice(len(target), 20000, replace=False)]

    fig = plt.figure(figsize=(10, 4), dpi=160)
    views = [(0, -90, "xy"), (0, 0, "xz"), (90, -90, "top")]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=0.1, c="#9cc3e6", alpha=0.25)
        ax.scatter(source[:, 0], source[:, 1], source[:, 2], s=5, c="#d62728", alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-mesh", required=True, type=Path)
    parser.add_argument("--photo-mesh-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--source-sample", type=int, default=None)
    parser.add_argument("--mri-sample", type=int, default=80000)
    parser.add_argument("--front-percentile", type=float, default=62.0)
    parser.add_argument("--front-axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--front-sign", type=int, choices=[-1, 1], default=None)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--trim-quantile", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mri_points = load_vertices(args.mri_mesh, max_points=args.mri_sample)

    results = []
    for photo_mesh in sorted(args.photo_mesh_dir.glob(args.pattern)):
        result = compare_one(photo_mesh, mri_points, args)
        preview_path = args.output_dir / f"{photo_mesh.stem}_to_mri_alignment.png"
        write_alignment_preview(result, preview_path)
        result["alignment_preview"] = str(preview_path)
        result.pop("_aligned")
        result.pop("_target")
        results.append(result)

    csv_path = args.output_dir / "photo_mesh_to_mri_summary.csv"
    json_path = args.output_dir / "photo_mesh_to_mri_summary.json"
    fieldnames = [
        "photo_mesh",
        "source_vertices",
        "target_cap_vertices",
        "front_axis",
        "front_sign",
        "pca_signs",
        "score",
        "mean_mm",
        "median_mm",
        "p75_mm",
        "p90_mm",
        "max_mm",
        "alignment_preview",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "n": len(results)}, indent=2))


if __name__ == "__main__":
    main()
