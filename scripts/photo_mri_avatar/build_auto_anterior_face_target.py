"""Build an automatic broad anterior face target from a structural MRI.

This is a no-manual-landmark fallback for scans where proxy landmarks are
unstable. It builds a whole-head mask, extracts the outer surface, then crops a
broad anterior x/z face sheet. The output is a candidate target that must pass
automated QC before any avatar metric is interpreted.
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


def load_volume(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return img, data, img.affine


def robust_threshold(data: np.ndarray) -> float:
    nonzero = data[data > 0]
    if nonzero.size < 1024:
        raise ValueError("Input volume has too few nonzero voxels.")
    lo, hi = np.percentile(nonzero, [1, 99.5])
    return float(filters.threshold_otsu(np.clip(nonzero, lo, hi)))


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        raise ValueError("Mask is empty.")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def build_head_mask(data: np.ndarray, threshold: float) -> np.ndarray:
    mask = data > threshold
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.ball(1))
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
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def voxel_world_coordinates(mask_shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ijk = np.indices(mask_shape, dtype=np.float32).reshape(3, -1).T
    ijk_h = np.c_[ijk, np.ones(len(ijk), dtype=np.float32)]
    return (affine @ ijk_h.T).T[:, :3]


def submesh_from_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces)
    keep_faces = vertex_mask[faces].all(axis=1)
    if keep_faces.sum() == 0:
        raise ValueError("Automatic anterior face crop produced no faces.")
    sub = mesh.submesh([keep_faces], append=True, repair=False)
    sub.remove_unreferenced_vertices()
    return sub


def component_summary(mesh: trimesh.Trimesh) -> dict:
    parts = mesh.split(only_watertight=False)
    sizes = sorted((len(part.vertices) for part in parts), reverse=True)
    return {
        "component_sizes": sizes[:10],
        "largest_component_fraction": float(sizes[0] / len(mesh.vertices)) if sizes else 0.0,
    }


def crop_anterior_face(
    head_mesh: trimesh.Trimesh,
    head_mask: np.ndarray,
    affine: np.ndarray,
    x_half_fraction: float,
    z_low_fraction: float,
    z_high_fraction: float,
    front_percentile: float,
) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    vertices = np.asarray(head_mesh.vertices, dtype=np.float64)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    x_center = float(np.median(x))
    x_span = float(np.percentile(x, 99) - np.percentile(x, 1))
    z_p01, z_p99 = np.percentile(z, [1, 99])
    z_span = float(z_p99 - z_p01)
    x_min = x_center - x_half_fraction * x_span
    x_max = x_center + x_half_fraction * x_span
    z_min = float(z_p01 + z_low_fraction * z_span)
    z_max = float(z_p01 + z_high_fraction * z_span)

    xz_candidate = (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)
    if int(xz_candidate.sum()) < 1000:
        raise ValueError(f"Face x/z candidate too small: {int(xz_candidate.sum())}")
    y_threshold = float(np.percentile(y[xz_candidate], front_percentile))
    vertex_mask = xz_candidate & (y >= y_threshold)
    face_mesh = submesh_from_vertex_mask(head_mesh, vertex_mask)

    coords = voxel_world_coordinates(head_mask.shape, affine)
    vx, vy, vz = coords[:, 0], coords[:, 1], coords[:, 2]
    voxel_xz = (vx >= x_min) & (vx <= x_max) & (vz >= z_min) & (vz <= z_max)
    voxel_front = voxel_xz & (vy >= y_threshold)
    surface_shell = head_mask & ~ndi.binary_erosion(head_mask, iterations=2)
    face_mask = voxel_front.reshape(head_mask.shape) & surface_shell
    face_mask = morphology.binary_dilation(face_mask, morphology.ball(1))
    face_mask = morphology.remove_small_objects(face_mask, min_size=256)

    params = {
        "mode": "broad_anterior_face_sheet_no_manual_landmarks",
        "x_center": x_center,
        "x_span_p01_p99": x_span,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_p01": float(z_p01),
        "z_p99": float(z_p99),
        "z_min": z_min,
        "z_max": z_max,
        "z_low_fraction": float(z_low_fraction),
        "z_high_fraction": float(z_high_fraction),
        "x_half_fraction": float(x_half_fraction),
        "front_percentile": float(front_percentile),
        "y_threshold": y_threshold,
        "xz_candidate_vertices": int(xz_candidate.sum()),
        "face_vertices_pre_submesh": int(vertex_mask.sum()),
    }
    return face_mesh, face_mask.astype(bool), params


def write_qc(data: np.ndarray, face_mask: np.ndarray, face_mesh: trimesh.Trimesh, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8), dpi=160)
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]
    centers = [int(np.round(np.mean(np.where(face_mask)[axis]))) for axis in range(3)]
    slices = [
        (data[centers[0], :, :], face_mask[centers[0], :, :], "sagittal face mask"),
        (data[:, centers[1], :], face_mask[:, centers[1], :], "coronal face mask"),
        (data[:, :, centers[2]], face_mask[:, :, centers[2]], "axial face mask"),
    ]
    for ax, (img, mask, title) in zip(axes[:3], slices):
        ax.imshow(np.rot90(img), cmap="gray")
        overlay = np.ma.masked_where(np.rot90(mask) == 0, np.rot90(mask))
        ax.imshow(overlay, cmap="autumn", alpha=0.55)
        ax.set_title(title)
        ax.axis("off")

    verts = np.asarray(face_mesh.vertices)
    if len(verts) > 40000:
        rng = np.random.default_rng(42)
        verts = verts[rng.choice(len(verts), 40000, replace=False)]
    for ax, (i, j, title) in zip(axes[3:], [(0, 1, "xy surface"), (0, 2, "xz surface"), (1, 2, "yz surface")]):
        ax.scatter(verts[:, i], verts[:, j], s=0.08, alpha=0.45, c="#2563eb")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def build(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    img, data, affine = load_volume(args.input)
    threshold = robust_threshold(data) if args.threshold is None else float(args.threshold)
    head_mask = build_head_mask(data, threshold)
    head_mesh = mask_to_mesh(head_mask, affine)
    face_mesh, face_mask, crop_params = crop_anterior_face(
        head_mesh,
        head_mask,
        affine,
        args.x_half_fraction,
        args.z_low_fraction,
        args.z_high_fraction,
        args.front_percentile,
    )

    stem = f"{args.subject}_{args.session}"
    mask_path = args.output_dir / f"{stem}_auto_face_mask.nii.gz"
    mesh_path = args.output_dir / f"{stem}_auto_face_surface.ply"
    qc_path = args.output_dir / f"{stem}_auto_face_qc.png"
    metadata_path = args.output_dir / f"{stem}_auto_face_metadata.json"

    out_img = nib.Nifti1Image(face_mask.astype(np.uint8), affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(mask_path))
    face_mesh.export(mesh_path)
    write_qc(data, face_mask, face_mesh, qc_path)

    metadata = {
        "input": str(args.input),
        "subject": args.subject,
        "session": args.session,
        "threshold": threshold,
        "head_voxels": int(head_mask.sum()),
        "face_voxels": int(face_mask.sum()),
        "head_mesh_vertices": int(len(head_mesh.vertices)),
        "head_mesh_faces": int(len(head_mesh.faces)),
        "face_mesh_vertices": int(len(face_mesh.vertices)),
        "face_mesh_faces": int(len(face_mesh.faces)),
        "face_mesh_components": component_summary(face_mesh),
        "crop_params": crop_params,
        "outputs": {"mask": str(mask_path), "mesh": str(mesh_path), "qc": str(qc_path)},
        "warning": "Automatic broad anterior target; no manual landmarks. Requires automated and visual QC before metric use.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", default="subject")
    parser.add_argument("--session", default="session")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--x-half-fraction", type=float, default=0.38)
    parser.add_argument("--z-low-fraction", type=float, default=0.10)
    parser.add_argument("--z-high-fraction", type=float, default=0.75)
    parser.add_argument("--front-percentile", type=float, default=60.0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
