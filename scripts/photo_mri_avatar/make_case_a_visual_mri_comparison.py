"""Build visual Case A photo-vs-MRI comparison assets.

The figure is intentionally a visual QC/proxy layer, not camera-calibrated MRI
projection. It maps MRI proxy landmarks into 2D photo coordinates using the
current 3DDFA photo landmarks, then overlays the MRI anterior face cap on the
same crop. This makes the current geometry claim inspectable on the photo.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


LANDMARKS = ["nose_tip", "chin", "brow_center", "left_cheek", "right_cheek"]
FIT_LANDMARKS = ["nose_tip", "brow_center", "left_cheek", "right_cheek"]
REPO = Path(__file__).resolve().parents[2]
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


FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18)
FONT_20_B = font(20, True)
FONT_26_B = font(26, True)


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


def load_3ddfa_keypoint_vertex_ids(bfm_pkl: Path) -> np.ndarray:
    with bfm_pkl.open("rb") as f:
        bfm = pickle.load(f)
    keypoints = np.asarray(bfm["keypoints"], dtype=np.int64)
    return keypoints[0::3] // 3


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


def apply_2d(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ points.T)).T + t


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


def photo_landmarks_2d(ply_path: Path, bfm_ids: np.ndarray, image_h: int) -> dict[str, np.ndarray]:
    verts = load_vertices(ply_path)
    dense = verts[bfm_ids]
    out = {}
    for name, indices in TDDFA_68_LANDMARKS.items():
        point = dense[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        out[name] = np.array([point[0], image_h - point[1]], dtype=np.float64)
    return out


def mri_front_cap_2d(mri_points: np.ndarray) -> np.ndarray:
    cap = mri_points[mri_points[:, 1] >= np.percentile(mri_points[:, 1], 62)]
    if len(cap) > 12000:
        rng = np.random.default_rng(7)
        cap = cap[rng.choice(len(cap), 12000, replace=False)]
    return np.column_stack([cap[:, 0], -cap[:, 2]])


def mri_landmarks_2d(mri_landmarks: dict[str, np.ndarray], swap_lr: bool = True) -> dict[str, np.ndarray]:
    lm = dict(mri_landmarks)
    if swap_lr:
        lm["left_cheek"], lm["right_cheek"] = lm["right_cheek"], lm["left_cheek"]
    return {name: np.array([p[0], -p[2]], dtype=np.float64) for name, p in lm.items()}


def crop_fit(path: Path, size: tuple[int, int], fill: str = "#f8fafc") -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def draw_mri_on_photo(
    crop_path: Path,
    ply_path: Path,
    mri_points: np.ndarray,
    mri_landmarks: dict[str, np.ndarray],
    bfm_ids: np.ndarray,
    output_size: tuple[int, int],
) -> Image.Image:
    base = ImageOps.exif_transpose(Image.open(crop_path)).convert("RGB")
    image_w, image_h = base.size
    photo_lm = photo_landmarks_2d(ply_path, bfm_ids, image_h)
    mri_lm = mri_landmarks_2d(mri_landmarks, swap_lr=True)
    r, scale, t = similarity_2d(
        np.vstack([mri_lm[name] for name in FIT_LANDMARKS]),
        np.vstack([photo_lm[name] for name in FIT_LANDMARKS]),
    )

    cap_photo = apply_2d(mri_front_cap_2d(mri_points), r, scale, t)
    lm_bounds = np.vstack([photo_lm[name] for name in LANDMARKS])
    lm_min = lm_bounds.min(axis=0)
    lm_max = lm_bounds.max(axis=0)
    lm_span = np.maximum(lm_max - lm_min, np.array([1.0, 1.0]))
    face_bounds = (
        lm_min[0] - 0.42 * lm_span[0],
        lm_max[0] + 0.42 * lm_span[0],
        lm_min[1] - 0.55 * lm_span[1],
        lm_max[1] + 0.35 * lm_span[1],
    )
    cap_photo = cap_photo[
        (cap_photo[:, 0] >= max(-40, face_bounds[0]))
        & (cap_photo[:, 0] <= min(image_w + 40, face_bounds[1]))
        & (cap_photo[:, 1] >= max(-40, face_bounds[2]))
        & (cap_photo[:, 1] <= min(image_h + 40, face_bounds[3]))
    ]
    lm_photo = {name: apply_2d(mri_lm[name].reshape(1, 2), r, scale, t)[0] for name in LANDMARKS}

    mask = Image.new("L", base.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    for x, y in cap_photo:
        draw_mask.ellipse((x - 3, y - 3, x + 3, y + 3), fill=42)
    mask = mask.filter(ImageFilter.GaussianBlur(4))
    mask = mask.point(lambda v: min(135, int(v * 2.6)))

    color = Image.new("RGB", base.size, "#ef4444")
    blended = Image.composite(color, base, mask)
    draw = ImageDraw.Draw(blended, "RGBA")

    # Draw target landmark points and their fitted photo counterparts.
    for name in FIT_LANDMARKS:
        x, y = lm_photo[name]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(239, 68, 68, 225), outline=(255, 255, 255, 230), width=2)
    for name in FIT_LANDMARKS:
        x, y = photo_lm[name]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(37, 99, 235, 230))

    draw.rectangle((16, 16, 364, 66), fill=(255, 255, 255, 218), outline=(203, 213, 225, 235))
    draw.text((28, 24), "MRI face-cap proxy over photo", fill=(15, 23, 42, 255), font=FONT_18)
    draw.text((28, 46), "red = MRI anterior cap, blue = photo landmarks", fill=(71, 85, 105, 255), font=FONT_14)

    return crop_fit_from_image(blended, output_size)


def crop_fit_from_image(img: Image.Image, size: tuple[int, int], fill: str = "#f8fafc") -> Image.Image:
    img = img.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def copy_or_placeholder(path: Path, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        return crop_fit(path, size)
    img = Image.new("RGB", size, "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((18, size[1] // 2 - 8), f"missing\n{path.name}", fill="#b91c1c", font=FONT_16)
    return img


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else float("nan")


def metric_summary(work: Path) -> list[tuple[str, str, str, str]]:
    mp = read_csv_rows(work / "landmark_alignment" / "crops_mediapipe" / "landmark_constrained_summary.csv")
    d3 = read_csv_rows(work / "landmark_alignment" / "crops_3ddfa_v2" / "landmark_constrained_summary.csv")
    d3_case = [r for r in d3 if "_1_1_" in r["photo_mesh"]]
    mp_case = [r for r in mp if "_1_1_" in r["photo_mesh"]]
    return [
        ("MediaPipe", f"{mean(mp_case, 'landmark_rmse_mm'):.1f}", f"{mean(mp_case, 'median_mm'):.1f}", f"{mean(mp_case, 'p90_mm'):.1f}"),
        ("3DDFA_V2", f"{mean(d3_case, 'landmark_rmse_mm'):.1f}", f"{mean(d3_case, 'median_mm'):.1f}", f"{mean(d3_case, 'p90_mm'):.1f}"),
        ("DECA/MICA", "blocked", "no FLAME", "no mesh"),
    ]


def make_metric_card(rows: list[tuple[str, str, str, str]], size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), "Current metric readout", fill="#0f172a", font=FONT_20_B)
    draw.text((20, 46), "Case A / landmark-constrained MRI proxy", fill="#64748b", font=FONT_16)
    x = [20, 182, 326, 468]
    y = 90
    headers = ["method", "LM RMSE", "surf med", "surf p90"]
    for idx, header in enumerate(headers):
        draw.text((x[idx], y), header, fill="#475569", font=FONT_14)
    y += 28
    for method, lm, med, p90 in rows:
        fill = "#111827" if method != "DECA/MICA" else "#991b1b"
        for idx, text in enumerate([method, lm, med, p90]):
            draw.text((x[idx], y), text, fill=fill, font=FONT_16)
        y += 34
    draw.line((20, y + 8, size[0] - 20, y + 8), fill="#cbd5e1", width=1)
    draw.text((20, y + 28), "Read this as QC, not validated anatomical accuracy.", fill="#64748b", font=FONT_14)
    draw.text((20, y + 50), "Free ICP ~2-3 mm was over-optimistic; constrained fit is stricter.", fill="#64748b", font=FONT_14)
    return img


def build(args: argparse.Namespace) -> None:
    work = args.workbench.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bfm_ids = load_3ddfa_keypoint_vertex_ids(resolve_bfm_pkl(args.bfm_pkl))
    mri_points = load_vertices(work / "mri_surfaces" / "kate_2018_outer_head.ply", max_points=140000)
    mri_lm = mri_proxy_landmarks(mri_points)

    crops = sorted((work / "photo_crops_3subjects_3ddfa_1024").glob("1_1*_facecrop.jpg"))[:4]
    if not crops:
        raise FileNotFoundError("No Case A 1_1 crops found")

    thumb = (250, 250)
    mri_thumb = (250, 250)
    pad = 16
    label_w = 162
    header_h = 96
    rows = [
        ("photo", "crop"),
        ("3DDFA", "3ddfa"),
        ("MediaPipe", "mediapipe"),
        ("MRI on photo", "mri_photo"),
    ]
    width = label_w + pad * (len(crops) + 2) + thumb[0] * len(crops)
    metric_h = 230
    height = header_h + pad * (len(rows) + 2) + thumb[1] * len(rows) + metric_h + 28
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "Case A: photo baselines against MRI proxy", fill="#0f172a", font=FONT_26_B)
    draw.text(
        (pad, 52),
        "Rows show the same photos, baseline overlays, and a proxy MRI face-cap projection onto each photo.",
        fill="#64748b",
        font=FONT_18,
    )
    draw.text((pad, 76), "Only Case A is displayed. DECA/MICA are prepared but blocked until licensed FLAME is installed.", fill="#991b1b", font=FONT_14)

    copied_projection_paths: list[str] = []
    for row_idx, (label, kind) in enumerate(rows):
        y = header_h + pad + row_idx * (thumb[1] + pad)
        draw.text((pad, y + 94), label, fill="#111827", font=FONT_20_B)
        for col_idx, crop in enumerate(crops):
            x = label_w + pad + col_idx * (thumb[0] + pad)
            if kind == "crop":
                img = crop_fit(crop, thumb)
            elif kind == "3ddfa":
                img = copy_or_placeholder(
                    work / "photo_avatar_crops_3subjects_3ddfa_v2" / f"faceage3_crops_3subjects_1024_{crop.stem}_3ddfa_v2_face1_overlay.jpg",
                    thumb,
                )
            elif kind == "mediapipe":
                safe = crop.stem.replace("-", "_")
                img = copy_or_placeholder(
                    work / "photo_avatar_crops_3subjects_mediapipe" / f"faceage3_{safe}_landmarks_overlay.jpg",
                    thumb,
                )
            else:
                ply = work / "photo_avatar_crops_3subjects_3ddfa_v2" / f"faceage3_crops_3subjects_1024_{crop.stem}_3ddfa_v2_face1.ply"
                img = draw_mri_on_photo(crop, ply, mri_points, mri_lm, bfm_ids, mri_thumb)
                if args.save_intermediates:
                    projection_path = out / f"case_a_{crop.stem}_mri_on_photo_proxy.jpg"
                    img.save(projection_path, quality=94)
                    copied_projection_paths.append(projection_path.name)
            canvas.paste(img, (x, y))
            draw.rounded_rectangle((x, y, x + thumb[0], y + thumb[1]), radius=8, outline="#cbd5e1", width=2)
            if row_idx == 0:
                badge = f"A.{col_idx + 1}"
                draw.rounded_rectangle((x + 8, y + 8, x + 58, y + 34), radius=6, fill="#ffffff", outline="#cbd5e1")
                draw.text((x + 18, y + 11), badge, fill="#111827", font=FONT_14)

    metric_card = make_metric_card(metric_summary(work), (width - pad * 2, metric_h))
    y = header_h + pad + len(rows) * (thumb[1] + pad) + 12
    canvas.paste(metric_card, (pad, y))
    draw.rounded_rectangle((pad, y, width - pad, y + metric_h), radius=8, outline="#cbd5e1", width=2)

    comparison_path = out / "case_a_photo_mri_visual_comparison.jpg"
    canvas.save(comparison_path, quality=94)
    if args.write_manifest:
        (out / "case_a_photo_mri_visual_comparison_manifest.json").write_text(
            json.dumps(
                {
                    "comparison": comparison_path.name,
                    "mri_on_photo_proxy_images": copied_projection_paths,
                    "note": "2D proxy projection from MRI proxy landmarks to 3DDFA photo landmarks; not camera calibration.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(comparison_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, default=Path("data/avatar_2026_work"))
    parser.add_argument("--output", type=Path, default=Path("project_page/assets"))
    parser.add_argument("--bfm-pkl", type=Path, default=None)
    parser.add_argument("--save-intermediates", action="store_true", help="Also save per-photo MRI-on-photo proxy images.")
    parser.add_argument("--write-manifest", action="store_true", help="Write a small JSON manifest for generated assets.")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
