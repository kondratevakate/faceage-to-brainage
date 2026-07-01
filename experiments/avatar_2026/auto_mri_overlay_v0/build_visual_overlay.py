from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\faceage_brainage\avatar_2026")
OUT_DIR = ROOT / "auto_mri_overlay_v0"


def get_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_B = get_font(22, True)
FONT = get_font(18)
SMALL = get_font(14)


def crop_light_margins(img: Image.Image, threshold: int = 246, pad: int = 16) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.width)
    y1 = min(int(ys.max()) + pad + 1, img.height)
    return img.crop((x0, y0, x1, y1))


def fit(path: Path, size: tuple[int, int], crop_margins: bool = False) -> Image.Image:
    if not path.exists():
        img = Image.new("RGB", size, "#f8fafc")
        draw = ImageDraw.Draw(img)
        draw.text((14, size[1] // 2 - 8), "missing", fill="#b91c1c", font=FONT)
        return img
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop_margins:
        img = crop_light_margins(img)
    img.thumbnail(size, Image.Resampling.LANCZOS)
    bg = Image.new("RGB", size, "#f8fafc")
    bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return bg


def build() -> None:
    assets = {
        "crops": ROOT / "photo_crops_3subjects_3ddfa_1024",
        "3ddfa": ROOT / "photo_avatar_crops_3subjects_3ddfa_v2",
        "mp": ROOT / "photo_avatar_crops_3subjects_mediapipe",
        "align3": OUT_DIR / "align_3ddfa_selected",
        "alignmp": OUT_DIR / "align_mediapipe_selected",
    }
    rows = [
        (
            "1_1",
            "1_1_photo_2026-04-09_19-38-04_facecrop.jpg",
            "faceage3_crops_3subjects_1024_1_1_photo_2026-04-09_19-38-04_facecrop_3ddfa_v2_face1",
            "faceage3_1_1_photo_2026_04_09_19_38_04_facecrop_mediapipe_facemesh",
        ),
        (
            "2_1",
            "2_1_photo_2_2026-04-09_20-38-32_facecrop.jpg",
            "faceage3_crops_3subjects_1024_2_1_photo_2_2026-04-09_20-38-32_facecrop_3ddfa_v2_face1",
            "faceage3_2_1_photo_2_2026_04_09_20_38_32_facecrop_mediapipe_facemesh",
        ),
        (
            "3_1",
            "3_1_IMG_20190814_110411_facecrop.jpg",
            "faceage3_crops_3subjects_1024_3_1_IMG_20190814_110411_facecrop_3ddfa_v2_face1",
            "faceage3_3_1_IMG_20190814_110411_facecrop_mediapipe_facemesh",
        ),
    ]
    cols = [
        ("photo crop", lambda r: assets["crops"] / r[1], False),
        ("3DDFA overlay", lambda r: assets["3ddfa"] / f"{r[2]}_overlay.jpg", False),
        ("MediaPipe overlay", lambda r: assets["mp"] / f"{r[3].replace('_mediapipe_facemesh', '_landmarks_overlay')}.jpg", False),
        ("3DDFA x MRI", lambda r: assets["align3"] / f"{r[2]}_landmark_constrained_alignment.png", True),
        ("MediaPipe x MRI", lambda r: assets["alignmp"] / f"{r[3]}_landmark_constrained_alignment.png", True),
    ]

    thumb_w, thumb_h = 250, 200
    pad = 16
    label_w = 76
    header_h = 74
    row_h = thumb_h + 36
    width = label_w + pad * (len(cols) + 2) + thumb_w * len(cols)
    height = header_h + pad + row_h * len(rows) + 42
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), "Automatic MRI alignment visual QC v0", fill="#111827", font=FONT_B)
    draw.text((pad, 42), "Proxy MRI mask/surface alignment for artifact review, not final anatomical ground truth.", fill="#64748b", font=SMALL)
    draw.text((pad, height - 28), "Rows are known folder labels. No identity inference is performed from face appearance.", fill="#64748b", font=SMALL)

    for col_idx, (title, _getter, _crop) in enumerate(cols):
        x = label_w + pad * (col_idx + 2) + thumb_w * col_idx
        draw.text((x, header_h - 24), title, fill="#334155", font=SMALL)

    for row_idx, row in enumerate(rows):
        y = header_h + pad + row_idx * row_h
        draw.text((pad, y + 80), row[0], fill="#111827", font=FONT_B)
        for col_idx, (_title, getter, crop_margins) in enumerate(cols):
            x = label_w + pad * (col_idx + 2) + thumb_w * col_idx
            img = fit(getter(row), (thumb_w, thumb_h), crop_margins=crop_margins)
            canvas.paste(img, (x, y))
            draw.rounded_rectangle((x, y, x + thumb_w, y + thumb_h), radius=8, outline="#cbd5e1", width=2)

    out = OUT_DIR / "auto_mri_overlay_v0_contact_sheet.jpg"
    canvas.save(out, quality=92)
    print(out)


if __name__ == "__main__":
    build()
