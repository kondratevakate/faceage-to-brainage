"""Create a portable input bundle for cloud avatar baselines.

Default behavior is privacy-minimal: bundle only the primary case subject crops
(`1_1`) and no MRI surfaces or internal controls. The resulting zip can be
uploaded to Colab, AWS, or another private GPU machine and the outputs can be
copied back into `data/avatar_2026_work/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BundleItem:
    source: str
    archive_path: str
    bytes: int
    subject_prefix: str


def subject_prefix(path: Path) -> str:
    parts = path.name.split("_", 2)
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return "unknown"


def write_readme(out_dir: Path, subject: str, include_mri: bool, include_controls: bool) -> None:
    text = f"""# Cloud Avatar Bundle

Purpose: portable input bundle for one-photo avatar baselines.

Default privacy scope:

- subject prefix: `{subject}`
- internal controls included: `{include_controls}`
- MRI surfaces included: `{include_mri}`

Expected cloud workflow:

1. Upload this zip to a private GPU runtime.
2. Unpack it.
3. Run the selected baseline into `outputs/<method>/`.
4. Download the output zip back to the local workbench.

Do not upload this bundle to public demos or public buckets.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbench", type=Path, default=Path("data/avatar_2026_work"))
    parser.add_argument("--crop-dir", type=Path, default=None)
    parser.add_argument("--subject-prefix", default="1_1")
    parser.add_argument("--include-internal-controls", action="store_true")
    parser.add_argument("--include-mri-surface", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bundle-name", default=None)
    args = parser.parse_args()

    workbench = args.workbench.resolve()
    crop_dir = (args.crop_dir or workbench / "photo_crops_3subjects_3ddfa_1024").resolve()
    output_dir = (args.output_dir or workbench / "cloud_bundles").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = args.bundle_name or f"avatar_case_{args.subject_prefix}_{timestamp}"
    staging = output_dir / bundle_name
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "inputs" / "crops").mkdir(parents=True)
    (staging / "outputs").mkdir()

    items: list[BundleItem] = []
    selected: list[Path] = []
    for crop in sorted(crop_dir.glob("*_facecrop.jpg")):
        prefix = subject_prefix(crop)
        if prefix == args.subject_prefix or args.include_internal_controls:
            selected.append(crop)

    if not selected:
        raise FileNotFoundError(f"No crops selected from {crop_dir} with subject prefix {args.subject_prefix}")

    for crop in selected:
        prefix = subject_prefix(crop)
        rel = Path("inputs") / "crops" / crop.name
        dst = staging / rel
        shutil.copy2(crop, dst)
        items.append(BundleItem(str(crop), rel.as_posix(), dst.stat().st_size, prefix))

    manifest_path = staging / "inputs" / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["archive_path", "subject_prefix", "bytes", "source"])
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))

    if args.include_mri_surface:
        mri_dir = workbench / "mri_surfaces"
        target = staging / "inputs" / "mri_surfaces"
        target.mkdir()
        for path in sorted(mri_dir.glob("*")):
            if path.is_file():
                rel = Path("inputs") / "mri_surfaces" / path.name
                dst = staging / rel
                shutil.copy2(path, dst)
                items.append(BundleItem(str(path), rel.as_posix(), dst.stat().st_size, "mri"))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject_prefix": args.subject_prefix,
        "include_internal_controls": args.include_internal_controls,
        "include_mri_surface": args.include_mri_surface,
        "item_count": len(items),
        "items": [asdict(item) for item in items],
    }
    (staging / "bundle_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(staging, args.subject_prefix, args.include_mri_surface, args.include_internal_controls)

    zip_path = output_dir / f"{bundle_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    print(zip_path)
    print(staging)
    print(f"items={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
