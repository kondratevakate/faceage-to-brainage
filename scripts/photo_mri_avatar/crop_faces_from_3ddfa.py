"""Create standardized face crops from 3DDFA_V2 detector metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps


def infer_group(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if "1_1" in parts:
        return "1_1"
    if "2_1" in parts:
        return "2_1"
    if "3_1" in parts:
        return "3_1"
    return "unknown"


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem).strip("_")


def crop_square(img: Image.Image, box: list[float], padding_scale: float, fill=(255, 255, 255)) -> tuple[Image.Image, dict]:
    x0, y0, x1, y1 = box[:4]
    width = x1 - x0
    height = y1 - y0
    side = max(width, height) * padding_scale
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    crop_x0 = int(round(cx - side / 2))
    crop_y0 = int(round(cy - side / 2))
    crop_x1 = int(round(cx + side / 2))
    crop_y1 = int(round(cy + side / 2))

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0
    canvas = Image.new("RGB", (crop_w, crop_h), fill)

    src_x0 = max(crop_x0, 0)
    src_y0 = max(crop_y0, 0)
    src_x1 = min(crop_x1, img.width)
    src_y1 = min(crop_y1, img.height)
    if src_x1 > src_x0 and src_y1 > src_y0:
        region = img.crop((src_x0, src_y0, src_x1, src_y1))
        canvas.paste(region, (src_x0 - crop_x0, src_y0 - crop_y0))

    return canvas, {
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_side_px": crop_w,
        "det_width_px": width,
        "det_height_px": height,
        "det_area_pct": 100 * width * height / max(img.width * img.height, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--padding-scale", type=float, default=2.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metadata_path in sorted(args.metadata_dir.glob("*_3ddfa_v2_face*_metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = Path(meta["source_image"]).resolve()
        det_box = meta.get("detected_box_xyxy_score")
        if not det_box or len(det_box) < 5:
            continue
        img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        crop, crop_meta = crop_square(img, det_box[:4], args.padding_scale)
        crop = crop.resize((args.crop_size, args.crop_size), Image.Resampling.LANCZOS)

        group = infer_group(source)
        crop_name = f"{group}_{safe_stem(source)}_facecrop.jpg"
        crop_path = args.output_dir / crop_name
        crop.save(crop_path, quality=94)

        rows.append(
            {
                "group": group,
                "source_image": str(source),
                "crop_image": str(crop_path.resolve()),
                "crop_name": crop_name,
                "source_width": img.width,
                "source_height": img.height,
                "detector_score": float(det_box[4]),
                **crop_meta,
                "crop_size": args.crop_size,
                "padding_scale": args.padding_scale,
                "source_metadata": str(metadata_path.resolve()),
            }
        )

    csv_path = args.output_dir / "face_crops_manifest.csv"
    fieldnames = [
        "group",
        "source_image",
        "crop_image",
        "crop_name",
        "source_width",
        "source_height",
        "detector_score",
        "det_width_px",
        "det_height_px",
        "det_area_pct",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_side_px",
        "crop_size",
        "padding_scale",
        "source_metadata",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(json.dumps({"crops": len(rows), "manifest": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
