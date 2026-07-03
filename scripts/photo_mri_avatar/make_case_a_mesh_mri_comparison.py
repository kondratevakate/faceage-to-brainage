"""Build a Case A mesh-level MRI-vs-photo comparison figure.

This figure avoids projecting MRI geometry onto the 2D photograph. Instead it
renders the MRI face-region surface points and the photo-derived 3DDFA mesh in
the same 3D coordinate frame after the current landmark-constrained alignment.
The MRI crop in this script is for visual QC only; metrics remain defined by the
separate evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[2]
LANDMARKS = ["nose_tip", "chin", "brow_center", "left_cheek", "right_cheek"]
FIT_LANDMARKS = ["nose_tip", "brow_center", "left_cheek", "right_cheek"]
TDDFA_68_LANDMARKS = {
    "nose_tip": [30],
    "chin": [8],
    "brow_center": [27],
    "left_cheek": [2],
    "right_cheek": [14],
}


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_13 = font(13)
FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18)
FONT_20_B = font(20, True)
FONT_28_B = font(28, True)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def finite_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.isfinite(vertices).all(axis=1)]


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), max_points, replace=False)]


def resolve_bfm_pkl(path: Path | None) -> Path:
    if path:
        return path
    candidates = [
        REPO.parent / "_external" / "avatars" / "3DDFA_V2" / "configs" / "bfm_noneck_v3.pkl",
        REPO / "_external" / "avatars" / "3DDFA_V2" / "configs" / "bfm_noneck_v3.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Pass --bfm-pkl pointing to 3DDFA_V2/configs/bfm_noneck_v3.pkl")


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    return keypoints[0::3] // 3


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


def extract_3ddfa_landmarks(vertices: np.ndarray, bfm_ids: np.ndarray) -> dict[str, np.ndarray]:
    dense = vertices[bfm_ids]
    return {
        name: dense[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for name, indices in TDDFA_68_LANDMARKS.items()
    }


def maybe_swap_lr(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swapped = dict(landmarks)
    swapped["left_cheek"], swapped["right_cheek"] = landmarks["right_cheek"], landmarks["left_cheek"]
    return swapped


def mri_proxy_landmarks(mri_points: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = mri_points[:, 0], mri_points[:, 1], mri_points[:, 2]
    x_center = np.median(x)
    x_span = np.percentile(x, 99) - np.percentile(x, 1)
    z_percentiles = np.percentile(z, [10, 20, 35, 50, 65, 80, 90])

    def pick_max_y(mask: np.ndarray) -> np.ndarray:
        indices = np.where(mask)[0]
        if len(indices) < 20:
            return mri_points[int(np.argmax(y))]
        return mri_points[indices[np.argmax(y[indices])]]

    central = np.abs(x - x_center) <= 0.14 * x_span
    near_left = x <= x_center - 0.18 * x_span
    near_right = x >= x_center + 0.18 * x_span
    side_band = (z >= z_percentiles[2]) & (z <= z_percentiles[4])
    return {
        "nose_tip": pick_max_y(central & (z >= z_percentiles[2]) & (z <= z_percentiles[5])),
        "chin": pick_max_y(central & (z >= z_percentiles[0]) & (z <= z_percentiles[2]) & (y >= np.percentile(y, 65))),
        "brow_center": pick_max_y(central & (z >= z_percentiles[4]) & (z <= z_percentiles[6])),
        "left_cheek": pick_max_y(near_left & side_band & (y >= np.percentile(y, 60))),
        "right_cheek": pick_max_y(near_right & side_band & (y >= np.percentile(y, 60))),
    }


def align_photo_to_mri(
    photo_vertices: np.ndarray,
    bfm_ids: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    source_landmarks = extract_3ddfa_landmarks(photo_vertices, bfm_ids)
    candidates = [source_landmarks, maybe_swap_lr(source_landmarks)]
    best = None
    for candidate in candidates:
        src = np.vstack([candidate[name] for name in FIT_LANDMARKS])
        dst = np.vstack([mri_landmarks[name] for name in FIT_LANDMARKS])
        r, scale, t = similarity_from_points(src, dst)
        aligned_lm = {name: apply_similarity(point.reshape(1, 3), r, scale, t)[0] for name, point in candidate.items()}
        residuals = np.linalg.norm(
            np.vstack([aligned_lm[name] for name in LANDMARKS]) - np.vstack([mri_landmarks[name] for name in LANDMARKS]),
            axis=1,
        )
        score = float(np.median(residuals) + 0.25 * np.percentile(residuals, 90))
        aligned = apply_similarity(photo_vertices, r, scale, t)
        item = (score, aligned, aligned_lm)
        if best is None or item[0] < best[0]:
            best = item
    assert best is not None
    return best[1], best[2], best[0]


def mri_face_roi_from_photo_extent(mri_points: np.ndarray, aligned_photo: np.ndarray) -> np.ndarray:
    x_min, y_min, z_min = np.percentile(aligned_photo, 2, axis=0)
    x_max, y_max, z_max = np.percentile(aligned_photo, 98, axis=0)
    x_span = max(x_max - x_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    cx = (x_min + x_max) / 2
    cz = (z_min + z_max) / 2
    rx = 0.64 * x_span
    rz = 0.58 * z_span
    ellipse = ((mri_points[:, 0] - cx) / rx) ** 2 + ((mri_points[:, 2] - cz) / rz) ** 2 <= 1.0
    z_band = (mri_points[:, 2] >= z_min - 0.08 * z_span) & (mri_points[:, 2] <= z_max + 0.10 * z_span)
    candidate = ellipse & z_band
    if candidate.sum() < 1000:
        candidate = np.ones(len(mri_points), dtype=bool)
    y_thr = np.percentile(mri_points[candidate, 1], 62)
    roi = candidate & (mri_points[:, 1] >= y_thr)
    return mri_points[roi]


def project_points(points: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    if view == "front":
        xy = np.column_stack([points[:, 0], points[:, 2]])
        depth = points[:, 1]
    elif view == "side":
        xy = np.column_stack([points[:, 1], points[:, 2]])
        depth = points[:, 0]
    elif view == "threequarter":
        xy = np.column_stack([0.88 * points[:, 0] - 0.48 * points[:, 1], points[:, 2] + 0.08 * points[:, 1]])
        depth = 0.35 * points[:, 0] + 0.94 * points[:, 1]
    else:
        raise ValueError(view)
    return xy, depth


def draw_point_layer(
    image: Image.Image,
    points: np.ndarray,
    view: str,
    bounds_points: np.ndarray,
    color: tuple[int, int, int, int],
    max_points: int,
    seed: int,
    radius: int = 1,
) -> None:
    if len(points) == 0:
        return
    points = sample_points(points, max_points, seed)
    xy, depth = project_points(points, view)
    bounds_xy, _ = project_points(bounds_points, view)
    min_xy = bounds_xy.min(axis=0)
    max_xy = bounds_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, np.array([1.0, 1.0]))
    pad = 28
    scale = min((image.width - 2 * pad) / span[0], (image.height - 2 * pad) / span[1])
    px = pad + (xy[:, 0] - min_xy[0]) * scale
    py = image.height - (pad + (xy[:, 1] - min_xy[1]) * scale)
    order = np.argsort(depth)
    layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for x, y in zip(px[order], py[order]):
        if radius <= 1:
            draw.point((float(x), float(y)), fill=color)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    image.alpha_composite(layer)


def draw_landmarks(
    image: Image.Image,
    landmarks: dict[str, np.ndarray],
    view: str,
    bounds_points: np.ndarray,
    color: tuple[int, int, int, int],
) -> None:
    points = np.vstack([landmarks[name] for name in LANDMARKS if name in landmarks])
    xy, _ = project_points(points, view)
    bounds_xy, _ = project_points(bounds_points, view)
    min_xy = bounds_xy.min(axis=0)
    max_xy = bounds_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, np.array([1.0, 1.0]))
    pad = 28
    scale = min((image.width - 2 * pad) / span[0], (image.height - 2 * pad) / span[1])
    px = pad + (xy[:, 0] - min_xy[0]) * scale
    py = image.height - (pad + (xy[:, 1] - min_xy[1]) * scale)
    draw = ImageDraw.Draw(image, "RGBA")
    for x, y in zip(px, py):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255, 235), width=2)


def render_panel(
    title: str,
    subtitle: str,
    mri_points: np.ndarray | None,
    photo_points: np.ndarray | None,
    mri_landmarks: dict[str, np.ndarray] | None,
    photo_landmarks: dict[str, np.ndarray] | None,
    view: str,
    size: tuple[int, int],
    bounds_points: np.ndarray,
) -> Image.Image:
    panel = Image.new("RGBA", size, "#ffffff")
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=10, outline="#cbd5e1", width=2)
    draw.text((18, 14), title, fill="#0f172a", font=FONT_20_B)
    if subtitle:
        draw.text((18, 42), subtitle, fill="#64748b", font=FONT_14)
    plot_area = Image.new("RGBA", (size[0] - 20, size[1] - 76), "#ffffff")
    plot_bounds = bounds_points
    if mri_points is not None:
        draw_point_layer(plot_area, mri_points, view, plot_bounds, (37, 99, 235, 108), 26000, 11, radius=1)
    if photo_points is not None:
        draw_point_layer(plot_area, photo_points, view, plot_bounds, (234, 88, 12, 150), 16000, 23, radius=1)
    if mri_landmarks is not None:
        draw_landmarks(plot_area, mri_landmarks, view, plot_bounds, (37, 99, 235, 230))
    if photo_landmarks is not None:
        draw_landmarks(plot_area, photo_landmarks, view, plot_bounds, (234, 88, 12, 230))
    panel.alpha_composite(plot_area, (10, 66))
    return panel.convert("RGB")


def fit_image(path: Path, size: tuple[int, int], fill: str = "#ffffff") -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def crop_for_mesh(mesh_path: Path, crops_dir: Path) -> Path | None:
    match = re.search(r"(1_1_photo_.+?_facecrop)_3ddfa_v2_face1$", mesh_path.stem)
    if not match:
        return None
    crop = crops_dir / f"{match.group(1)}.jpg"
    return crop if crop.exists() else None


def crop_key(stem: str) -> str:
    match = re.search(r"(1_1_photo_.+?_facecrop)", stem)
    return match.group(1) if match else stem


def read_case_metrics(path: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = crop_key(Path(row["photo_mesh"]).stem)
            out[key] = (float(row["landmark_rmse_mm"]), float(row["median_mm"]), float(row["p90_mm"]))
    return out


def build(args: argparse.Namespace) -> None:
    work = args.workbench.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bfm_ids = load_3ddfa_keypoint_vertex_ids(resolve_bfm_pkl(args.bfm_pkl))
    mri_mesh = load_mesh(work / "mri_surfaces" / "kate_2018_outer_head.ply")
    mri_points = finite_vertices(mri_mesh)
    mri_landmarks = mri_proxy_landmarks(mri_points)
    mesh_paths = sorted((work / "photo_avatar_crops_3subjects_3ddfa_v2").glob("faceage3_crops_3subjects_1024_1_1*_3ddfa_v2_face1.ply"))[:4]
    if not mesh_paths:
        raise FileNotFoundError("No Case A 3DDFA meshes found")

    aligned_cases = []
    for mesh_path in mesh_paths:
        vertices = finite_vertices(load_mesh(mesh_path))
        aligned, aligned_lm, _score = align_photo_to_mri(vertices, bfm_ids, mri_landmarks)
        aligned_cases.append((mesh_path, aligned, aligned_lm))

    mri_face = mri_face_roi_from_photo_extent(mri_points, aligned_cases[0][1])
    bounds = np.vstack([mri_face, aligned_cases[0][1]])
    metrics = read_case_metrics(work / "landmark_alignment" / "crops_3ddfa_v2" / "landmark_constrained_summary.csv")

    w = 1380
    h = 1560
    pad = 22
    canvas = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 22), "Case A: MRI face mesh vs photo-derived mesh", fill="#0f172a", font=FONT_28_B)
    draw.text(
        (pad, 60),
        "Blue = MRI face-region surface points. Orange = 3DDFA mesh from one photo, aligned by proxy landmarks.",
        fill="#475569",
        font=FONT_18,
    )
    draw.text(
        (pad, 86),
        "This is a 3D QC view. It avoids the misleading 2D MRI-on-photo projection.",
        fill="#991b1b",
        font=FONT_14,
    )

    first_mesh, first_aligned, first_lm = aligned_cases[0]
    panel_size = (322, 330)
    panels = [
        render_panel("MRI face ROI", "visual crop from MRI surface", mri_face, None, mri_landmarks, None, "front", panel_size, bounds),
        render_panel("Photo mesh", "3DDFA face surface", None, first_aligned, None, first_lm, "front", panel_size, bounds),
        render_panel("Overlay / front", "same 3D frame", mri_face, first_aligned, mri_landmarks, first_lm, "front", panel_size, bounds),
        render_panel("Overlay / side", "depth mismatch is visible here", mri_face, first_aligned, mri_landmarks, first_lm, "side", panel_size, bounds),
    ]
    y0 = 120
    for i, panel in enumerate(panels):
        canvas.paste(panel, (pad + i * (panel_size[0] + 14), y0))

    legend_y = y0 + panel_size[1] + 18
    draw.rounded_rectangle((pad, legend_y, w - pad, legend_y + 86), radius=8, outline="#cbd5e1", width=2)
    draw.ellipse((pad + 22, legend_y + 24, pad + 42, legend_y + 44), fill="#2563eb")
    draw.text((pad + 52, legend_y + 22), "MRI face-region surface points", fill="#0f172a", font=FONT_16)
    draw.ellipse((pad + 352, legend_y + 24, pad + 372, legend_y + 44), fill="#ea580c")
    draw.text((pad + 382, legend_y + 22), "photo-derived 3DDFA mesh points", fill="#0f172a", font=FONT_16)
    draw.text(
        (pad + 22, legend_y + 54),
        "The MRI crop is only for visualization; final metrics must use the fixed region/alignment contract.",
        fill="#64748b",
        font=FONT_14,
    )

    strip_y = legend_y + 112
    draw.text((pad, strip_y), "Repeated Case A photos against the same MRI face ROI", fill="#0f172a", font=FONT_20_B)
    strip_y += 36
    small_w, small_h = 322, 460
    crops_dir = work / "photo_crops_3subjects_3ddfa_1024"
    for i, (mesh_path, aligned, aligned_lm) in enumerate(aligned_cases):
        x = pad + i * (small_w + 14)
        panel = Image.new("RGB", (small_w, small_h), "#ffffff")
        pdraw = ImageDraw.Draw(panel)
        pdraw.rounded_rectangle((0, 0, small_w - 1, small_h - 1), radius=10, outline="#cbd5e1", width=2)
        pdraw.text((14, 12), f"A.{i + 1}", fill="#0f172a", font=FONT_20_B)
        crop = crop_for_mesh(mesh_path, crops_dir)
        if crop:
            panel.paste(fit_image(crop, (112, 112), "#f8fafc"), (14, 44))
        local_bounds = np.vstack([mri_face, aligned])
        overlay = render_panel("", "", mri_face, aligned, None, None, "front", (small_w - 24, 230), local_bounds)
        panel.paste(overlay.crop((0, 54, small_w - 24, 230)), (12, 166))
        rmse, med, p90 = metrics.get(crop_key(mesh_path.stem), (float("nan"), float("nan"), float("nan")))
        pdraw.text((14, 406), f"LM RMSE {rmse:.1f} mm", fill="#334155", font=FONT_14)
        pdraw.text((154, 406), f"surf med {med:.1f}", fill="#334155", font=FONT_14)
        pdraw.text((154, 428), f"p90 {p90:.1f}", fill="#334155", font=FONT_14)
        canvas.paste(panel, (x, strip_y))

    note_y = strip_y + small_h + 22
    draw.rounded_rectangle((pad, note_y, w - pad, note_y + 104), radius=8, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text((pad + 18, note_y + 18), "Interpretation", fill="#0f172a", font=FONT_20_B)
    draw.text(
        (pad + 18, note_y + 50),
        "This view is more honest than MRI-on-photo overlay: it shows the current 3D agreement and the remaining landmark/shape mismatch directly.",
        fill="#475569",
        font=FONT_16,
    )
    draw.text(
        (pad + 18, note_y + 76),
        "The MRI face ROI still needs better anatomical masking and manual/semi-manual landmarks before accuracy claims.",
        fill="#475569",
        font=FONT_16,
    )

    path = out / "case_a_mesh_mri_comparison.jpg"
    canvas.save(path, quality=94)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, default=Path("data/avatar_2026_work"))
    parser.add_argument("--output", type=Path, default=Path("project_page/assets"))
    parser.add_argument("--bfm-pkl", type=Path, default=None)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
