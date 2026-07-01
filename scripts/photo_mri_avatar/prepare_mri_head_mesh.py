"""Extract a private outer-head surface mesh from a non-defaced T1 MRI.

The output is face-reconstructable. Keep it in an ignored/private folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


def _as_float_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, img.affine


def _robust_threshold(data: np.ndarray) -> float:
    nonzero = data[data > 0]
    if nonzero.size < 1024:
        raise ValueError("Input volume has too few nonzero voxels to segment.")
    lo, hi = np.percentile(nonzero, [1, 99.5])
    clipped = np.clip(nonzero, lo, hi)
    return float(filters.threshold_otsu(clipped))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        raise ValueError("Threshold produced an empty mask.")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def build_head_mask(data: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Create a conservative whole-head mask from a structural MRI volume."""

    level = _robust_threshold(data) if threshold is None else threshold
    mask = data > level
    mask = _largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.ball(2))
    mask = morphology.remove_small_holes(mask, area_threshold=2048)
    mask = morphology.remove_small_objects(mask, min_size=4096)
    return mask.astype(bool)


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _normals, _values = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        step_size=1,
        allow_degenerate=False,
    )
    verts_h = np.c_[verts, np.ones(len(verts))]
    world = (affine @ verts_h.T).T[:, :3]
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def write_preview(mask: np.ndarray, mesh: trimesh.Trimesh, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7), dpi=160)
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]

    centers = [s // 2 for s in mask.shape]
    slices = [
        mask[centers[0], :, :],
        mask[:, centers[1], :],
        mask[:, :, centers[2]],
    ]
    titles = ["mask sagittal", "mask coronal", "mask axial"]
    for ax, slc, title in zip(axes[:3], slices, titles):
        ax.imshow(np.rot90(slc), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    verts = mesh.vertices
    if len(verts) > 30000:
        rng = np.random.default_rng(42)
        verts = verts[rng.choice(len(verts), 30000, replace=False)]

    projections = [(0, 1, "xy vertices"), (0, 2, "xz vertices"), (1, 2, "yz vertices")]
    for ax, (i, j, title) in zip(axes[3:], projections):
        ax.scatter(verts[:, i], verts[:, j], s=0.05, alpha=0.35)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Non-defaced MRI NIfTI.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", default="subject")
    parser.add_argument("--session", default="session")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, affine = _as_float_volume(args.input)
    threshold = _robust_threshold(data) if args.threshold is None else args.threshold
    mask = build_head_mask(data, threshold=threshold)
    mesh = mask_to_mesh(mask, affine)

    stem = f"{args.subject}_{args.session}"
    mesh_path = args.output_dir / f"{stem}_outer_head.ply"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    mesh.export(mesh_path)

    preview_path = None
    if args.preview:
        preview_path = args.output_dir / f"{stem}_qc.png"
        write_preview(mask, mesh, preview_path)

    metadata = {
        "input": str(args.input),
        "output_mesh": str(mesh_path),
        "preview": str(preview_path) if preview_path else None,
        "subject": args.subject,
        "session": args.session,
        "threshold": threshold,
        "shape": list(data.shape),
        "voxel_count_mask": int(mask.sum()),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "warning": "Face-reconstructable private output. Do not commit or publish without explicit review.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
