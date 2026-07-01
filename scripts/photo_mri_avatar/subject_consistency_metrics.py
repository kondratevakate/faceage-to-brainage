"""Subject-aware consistency and separation metrics for avatar meshes.

The script assumes subject labels are known from folder/file prefixes such as
`1_1_*` and `2_1_*`. It does not perform identity recognition.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np


MEDIAPIPE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "brow_center": 168,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

TDDFA_68_LANDMARKS = {
    "nose_tip": 30,
    "chin": 8,
    "brow_center": 27,
    "left_cheek": 2,
    "right_cheek": 14,
    "left_eye_outer": 36,
    "right_eye_outer": 45,
    "mouth_left": 48,
    "mouth_right": 54,
}


def infer_subject_id(path: Path) -> str:
    match = re.match(r"^(\d+_\d+)_", path.name)
    if match:
        return match.group(1)
    for part in path.parts:
        if re.match(r"^\d+_\d+$", part):
            return part
    return "unknown"


def load_ascii_ply_vertices(path: Path) -> np.ndarray | None:
    vertex_count = None
    is_ascii = False
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "format ascii 1.0":
                is_ascii = True
            elif line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
        if not is_ascii or vertex_count is None:
            return None
        vertices = np.loadtxt(f, dtype=np.float64, max_rows=vertex_count, usecols=(0, 1, 2))
    if vertices.ndim == 1:
        vertices = vertices.reshape(1, 3)
    return vertices


def load_mesh_vertices(path: Path) -> np.ndarray:
    vertices = load_ascii_ply_vertices(path)
    if vertices is None:
        import trimesh

        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def bbox_diag(points: np.ndarray) -> float:
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extents))


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


def maybe_sample_correspondence(left: np.ndarray, right: np.ndarray, max_vertices: int | None) -> tuple[np.ndarray, np.ndarray]:
    if not max_vertices or max_vertices <= 0 or len(left) <= max_vertices:
        return left, right
    indices = np.linspace(0, len(left) - 1, max_vertices, dtype=np.int64)
    return left[indices], right[indices]


def mesh_correspondence_distance(left: np.ndarray, right: np.ndarray, max_vertices: int | None = None) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError(f"Vertex count mismatch: {len(left)} != {len(right)}")
    left, right = maybe_sample_correspondence(left, right, max_vertices)
    r, scale, t = similarity_from_correspondence(left, right)
    aligned = apply_similarity(left, r, scale, t)
    distances = np.linalg.norm(aligned - right, axis=1)
    norm = bbox_diag(right)
    return {
        "vertices_used": int(len(left)),
        "mesh_median_pct_bbox": float(100 * np.median(distances) / max(norm, 1e-12)),
        "mesh_p90_pct_bbox": float(100 * np.percentile(distances, 90) / max(norm, 1e-12)),
        "mesh_mean_pct_bbox": float(100 * np.mean(distances) / max(norm, 1e-12)),
        "scale_left_to_right": float(scale),
    }


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    if len(keypoints) != 204:
        raise ValueError(f"Expected 204 flattened 68-landmark entries, got {len(keypoints)}")
    return keypoints[0::3] // 3


def extract_landmarks(vertices: np.ndarray, method: str, tddfa_keypoint_ids: np.ndarray | None) -> dict[str, np.ndarray]:
    if method == "mediapipe":
        return {name: vertices[idx] for name, idx in MEDIAPIPE_LANDMARKS.items()}
    if method == "3ddfa_v2":
        if tddfa_keypoint_ids is None:
            raise ValueError("3DDFA landmarks require --bfm-pkl")
        sparse = vertices[tddfa_keypoint_ids]
        return {name: sparse[idx] for name, idx in TDDFA_68_LANDMARKS.items()}
    raise ValueError(method)


def landmark_descriptor(landmarks: dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(landmarks)
    points = np.vstack([landmarks[name] for name in names])
    norm = bbox_diag(points)
    values = []
    for i, j in combinations(range(len(names)), 2):
        values.append(np.linalg.norm(points[i] - points[j]) / max(norm, 1e-12))
    return np.asarray(values, dtype=np.float64)


def load_mediapipe_rows(output_dir: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(output_dir.glob("*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = Path(meta["input"]).resolve()
        mesh_path = Path(meta["output_mesh"]).resolve()
        rows.append(
            {
                "method": "mediapipe",
                "subject_id": infer_subject_id(source),
                "source_image": str(source),
                "source_name": source.name,
                "mesh_path": str(mesh_path),
                "mesh_name": mesh_path.name,
            }
        )
    return rows


def load_3ddfa_rows(output_dir: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(output_dir.glob("*_3ddfa_v2_face*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = Path(meta["source_image"]).resolve()
        mesh_path = Path(str(metadata_path).replace("_metadata.json", ".ply")).resolve()
        if not mesh_path.exists():
            continue
        rows.append(
            {
                "method": "3ddfa_v2",
                "subject_id": infer_subject_id(source),
                "source_image": str(source),
                "source_name": source.name,
                "mesh_path": str(mesh_path),
                "mesh_name": mesh_path.name,
            }
        )
    return rows


def auc_smaller_is_genuine(genuine: list[float], impostor: list[float]) -> float:
    if not genuine or not impostor:
        return float("nan")
    wins = 0.0
    total = len(genuine) * len(impostor)
    for g in genuine:
        for imp in impostor:
            if imp > g:
                wins += 1.0
            elif imp == g:
                wins += 0.5
    return float(wins / total)


def eer_threshold(genuine: list[float], impostor: list[float]) -> tuple[float, float, float, float]:
    if not genuine or not impostor:
        return float("nan"), float("nan"), float("nan"), float("nan")
    thresholds = sorted(set(genuine + impostor))
    best = None
    for th in thresholds:
        far = sum(imp <= th for imp in impostor) / len(impostor)
        frr = sum(g > th for g in genuine) / len(genuine)
        eer = (far + frr) / 2
        candidate = (abs(far - frr), eer, th, far, frr)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _diff, eer, th, far, frr = best
    return float(eer), float(th), float(far), float(frr)


def summarize_metric(rows: list[dict], metric: str) -> dict:
    genuine = [float(row[metric]) for row in rows if row["pair_label"] == "genuine"]
    impostor = [float(row[metric]) for row in rows if row["pair_label"] == "impostor"]
    eer, threshold, far, frr = eer_threshold(genuine, impostor)
    genuine_p90 = float(np.percentile(genuine, 90)) if genuine else float("nan")
    impostor_p10 = float(np.percentile(impostor, 10)) if impostor else float("nan")
    return {
        "metric": metric,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "genuine_median": float(np.median(genuine)) if genuine else float("nan"),
        "genuine_p90": genuine_p90,
        "impostor_p10": impostor_p10,
        "impostor_median": float(np.median(impostor)) if impostor else float("nan"),
        "median_margin": (float(np.median(impostor)) - float(np.median(genuine))) if genuine and impostor else float("nan"),
        "strict_gap_p10_minus_p90": impostor_p10 - genuine_p90,
        "auc": auc_smaller_is_genuine(genuine, impostor),
        "eer": eer,
        "eer_threshold": threshold,
        "far_at_eer_threshold": far,
        "frr_at_eer_threshold": frr,
        "passes_genuine_p90_lt_impostor_p10": bool(genuine and impostor and genuine_p90 < impostor_p10),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediapipe-dir", type=Path)
    parser.add_argument("--3ddfa-dir", dest="dddfa_dir", type=Path)
    parser.add_argument("--bfm-pkl", type=Path)
    parser.add_argument("--vertex-sample", type=int, default=0, help="Deterministic max vertex count per mesh pair; 0 uses all vertices.")
    parser.add_argument("--include-source-name", action="append", default=[], help="Exact source image filename to include; repeatable.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    tddfa_keypoint_ids = load_3ddfa_keypoint_vertex_ids(args.bfm_pkl) if args.bfm_pkl else None

    input_rows = []
    if args.mediapipe_dir:
        input_rows.extend(load_mediapipe_rows(args.mediapipe_dir))
    if args.dddfa_dir:
        input_rows.extend(load_3ddfa_rows(args.dddfa_dir))
    if args.include_source_name:
        include = set(args.include_source_name)
        input_rows = [row for row in input_rows if row["source_name"] in include]

    vertices_cache = {}
    descriptor_cache = {}
    pair_rows = []
    for method in sorted(set(row["method"] for row in input_rows)):
        method_rows = [row for row in input_rows if row["method"] == method]
        for left, right in combinations(method_rows, 2):
            left_vertices = vertices_cache.setdefault(left["mesh_path"], load_mesh_vertices(Path(left["mesh_path"])))
            right_vertices = vertices_cache.setdefault(right["mesh_path"], load_mesh_vertices(Path(right["mesh_path"])))
            mesh_stats = mesh_correspondence_distance(left_vertices, right_vertices, args.vertex_sample)

            for row, vertices in [(left, left_vertices), (right, right_vertices)]:
                if row["mesh_path"] not in descriptor_cache:
                    landmarks = extract_landmarks(vertices, method, tddfa_keypoint_ids)
                    descriptor_cache[row["mesh_path"]] = landmark_descriptor(landmarks)
            landmark_distance = float(np.linalg.norm(descriptor_cache[left["mesh_path"]] - descriptor_cache[right["mesh_path"]]))

            pair_rows.append(
                {
                    "method": method,
                    "left_subject_id": left["subject_id"],
                    "right_subject_id": right["subject_id"],
                    "pair_label": "genuine" if left["subject_id"] == right["subject_id"] else "impostor",
                    "left_source_name": left["source_name"],
                    "right_source_name": right["source_name"],
                    "left_mesh_name": left["mesh_name"],
                    "right_mesh_name": right["mesh_name"],
                    "landmark_descriptor_distance": landmark_distance,
                    **mesh_stats,
                }
            )

    summary_rows = []
    for method in sorted(set(row["method"] for row in pair_rows)):
        method_pairs = [row for row in pair_rows if row["method"] == method]
        for metric in ["mesh_median_pct_bbox", "mesh_p90_pct_bbox", "landmark_descriptor_distance"]:
            summary = summarize_metric(method_pairs, metric)
            summary_rows.append({"method": method, **summary})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "subject_consistency_pairs.csv",
        pair_rows,
        [
            "method",
            "left_subject_id",
            "right_subject_id",
            "pair_label",
            "left_source_name",
            "right_source_name",
            "left_mesh_name",
            "right_mesh_name",
            "vertices_used",
            "mesh_median_pct_bbox",
            "mesh_p90_pct_bbox",
            "mesh_mean_pct_bbox",
            "scale_left_to_right",
            "landmark_descriptor_distance",
        ],
    )
    write_csv(
        args.output_dir / "subject_consistency_summary.csv",
        summary_rows,
        [
            "method",
            "metric",
            "n_genuine",
            "n_impostor",
            "genuine_median",
            "genuine_p90",
            "impostor_p10",
            "impostor_median",
            "median_margin",
            "strict_gap_p10_minus_p90",
            "auc",
            "eer",
            "eer_threshold",
            "far_at_eer_threshold",
            "frr_at_eer_threshold",
            "passes_genuine_p90_lt_impostor_p10",
        ],
    )
    (args.output_dir / "subject_consistency_summary.json").write_text(
        json.dumps({"pairs": pair_rows, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"pairs": len(pair_rows), "summary_rows": len(summary_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
