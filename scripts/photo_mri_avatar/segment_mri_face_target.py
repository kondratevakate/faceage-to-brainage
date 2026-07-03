"""Create a coarse face-region target from a non-defaced structural MRI.

This is a target-building utility, not a validated clinical segmentation. It
extracts a whole-head mask, estimates proxy face landmarks on the outer-head
surface, clips a face-only anterior region, and writes:

- a face ROI NIfTI mask;
- a face-region surface PLY;
- a QC PNG with MRI slices and surface views;
- metadata documenting thresholds and limitations.
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


LANDMARKS = ["nose_tip", "chin", "brow_center", "left_cheek", "right_cheek"]


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
    clipped = np.clip(nonzero, lo, hi)
    return float(filters.threshold_otsu(clipped))


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
    mask = morphology.binary_closing(mask, morphology.ball(2))
    mask = morphology.remove_small_holes(mask, area_threshold=2048)
    mask = morphology.remove_small_objects(mask, min_size=4096)
    return mask.astype(bool)


def robust_normalize(data: np.ndarray) -> np.ndarray:
    nonzero = data[data > 0]
    if nonzero.size < 1024:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(nonzero, [1, 99.5])
    norm = (np.clip(data, lo, hi) - lo) / max(float(hi - lo), 1e-6)
    return norm.astype(np.float32)


def validate_same_grid(left: nib.Nifti1Image, right: nib.Nifti1Image) -> None:
    if left.shape != right.shape:
        raise ValueError(f"Original and defaced shapes differ: {left.shape} vs {right.shape}")
    if not np.allclose(left.affine, right.affine, atol=1e-4):
        raise ValueError("Original and defaced affines differ; register/resample before differencing.")


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, step_size: int = 1) -> trimesh.Trimesh:
    verts, faces, _normals, _values = measure.marching_cubes(
        mask.astype(np.float32),
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )
    verts_h = np.c_[verts, np.ones(len(verts))]
    world = (affine @ verts_h.T).T[:, :3]
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def mri_proxy_landmarks(points: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate rough face landmarks from an MRI outer-head surface."""

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_center = np.median(x)
    x_span = np.percentile(x, 99) - np.percentile(x, 1)
    z_percentiles = np.percentile(z, [10, 20, 35, 50, 65, 80, 90])

    def pick_max_y(mask: np.ndarray, name: str) -> np.ndarray:
        indices = np.where(mask)[0]
        if len(indices) < 20:
            raise ValueError(f"Too few candidate vertices for {name}: {len(indices)}")
        return points[indices[np.argmax(y[indices])]]

    central = np.abs(x - x_center) <= 0.14 * x_span
    near_left = x <= x_center - 0.18 * x_span
    near_right = x >= x_center + 0.18 * x_span
    side_band = (z >= z_percentiles[2]) & (z <= z_percentiles[4])

    return {
        "nose_tip": pick_max_y(central & (z >= z_percentiles[2]) & (z <= z_percentiles[5]), "nose_tip"),
        "chin": pick_max_y(
            central & (z >= z_percentiles[0]) & (z <= z_percentiles[2]) & (y >= np.percentile(y, 65)),
            "chin",
        ),
        "brow_center": pick_max_y(central & (z >= z_percentiles[4]) & (z <= z_percentiles[6]), "brow_center"),
        "left_cheek": pick_max_y(near_left & side_band & (y >= np.percentile(y, 60)), "left_cheek"),
        "right_cheek": pick_max_y(near_right & side_band & (y >= np.percentile(y, 60)), "right_cheek"),
    }


def face_roi_world_mask(points: np.ndarray, landmarks: dict[str, np.ndarray], front_percentile: float) -> tuple[np.ndarray, dict]:
    left = landmarks["left_cheek"]
    right = landmarks["right_cheek"]
    chin = landmarks["chin"]
    brow = landmarks["brow_center"]

    x_center = float((left[0] + right[0]) / 2.0)
    z_center = float((chin[2] + brow[2]) / 2.0)
    cheek_width = max(abs(float(right[0] - left[0])), 1.0)
    face_height = max(abs(float(brow[2] - chin[2])), 1.0)
    rx = 0.78 * cheek_width
    rz = 0.56 * face_height
    ellipse = ((points[:, 0] - x_center) / rx) ** 2 + ((points[:, 2] - z_center) / rz) ** 2 <= 1.0
    z_band = (points[:, 2] >= chin[2] - 0.06 * face_height) & (points[:, 2] <= brow[2] + 0.05 * face_height)

    candidate = ellipse & z_band
    if candidate.sum() < 1000:
        raise ValueError(f"Face ROI candidate too small: {int(candidate.sum())}")
    y_threshold = float(np.percentile(points[candidate, 1], front_percentile))
    roi = candidate & (points[:, 1] >= y_threshold)
    params = {
        "x_center": x_center,
        "z_center": z_center,
        "rx": float(rx),
        "rz": float(rz),
        "front_percentile": float(front_percentile),
        "y_threshold": y_threshold,
        "candidate_points": int(candidate.sum()),
        "roi_points": int(roi.sum()),
    }
    return roi, params


def voxel_world_coordinates(mask_shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ijk = np.indices(mask_shape, dtype=np.float32).reshape(3, -1).T
    ijk_h = np.c_[ijk, np.ones(len(ijk), dtype=np.float32)]
    return (affine @ ijk_h.T).T[:, :3]


def build_face_volume_mask(head_mask: np.ndarray, affine: np.ndarray, landmarks: dict[str, np.ndarray], front_percentile: float) -> tuple[np.ndarray, dict]:
    coords = voxel_world_coordinates(head_mask.shape, affine)
    roi, params = face_roi_world_mask(coords, landmarks, front_percentile)
    surface_shell = head_mask & ~ndi.binary_erosion(head_mask, iterations=2)
    face = roi.reshape(head_mask.shape) & surface_shell
    face = morphology.binary_dilation(face, morphology.ball(1))
    face = morphology.remove_small_objects(face, min_size=256)
    return face.astype(bool), params


def build_deface_difference_mask(
    original: np.ndarray,
    defaced: np.ndarray,
    head_mask: np.ndarray,
    diff_threshold: float,
) -> tuple[np.ndarray, dict]:
    original_norm = robust_normalize(original)
    defaced_norm = robust_normalize(defaced)
    removed = head_mask & (original_norm > 0.08) & (
        (defaced_norm < 0.04) | ((original_norm - defaced_norm) >= diff_threshold)
    )
    removed = morphology.binary_closing(removed, morphology.ball(1))
    removed = morphology.remove_small_objects(removed, min_size=256)
    surface_shell = head_mask & ~ndi.binary_erosion(head_mask, iterations=2)
    face = removed & morphology.binary_dilation(surface_shell, morphology.ball(1))
    if face.sum() < 256:
        face = removed
    face = morphology.binary_dilation(face, morphology.ball(1))
    face = morphology.remove_small_objects(face, min_size=256)
    params = {
        "diff_threshold": float(diff_threshold),
        "removed_voxels": int(removed.sum()),
        "face_voxels": int(face.sum()),
        "mode": "defaced_difference",
    }
    return face.astype(bool), params


def submesh_from_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces)
    keep_faces = vertex_mask[faces].all(axis=1)
    if keep_faces.sum() == 0:
        raise ValueError("Face surface crop produced no faces.")
    sub = mesh.submesh([keep_faces], append=True, repair=False)
    sub.remove_unreferenced_vertices()
    return sub


def write_qc(data: np.ndarray, head_mask: np.ndarray, face_mask: np.ndarray, face_mesh: trimesh.Trimesh, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8), dpi=160)
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]

    centers = [int(np.round(np.mean(np.where(face_mask)[axis]))) for axis in range(3)]
    slices = [
        (data[centers[0], :, :], face_mask[centers[0], :, :], "sagittal face mask"),
        (data[:, centers[1], :], face_mask[:, centers[1], :], "coronal face mask"),
        (data[:, :, centers[2]], face_mask[:, :, centers[2]], "axial face mask"),
    ]
    for ax, (img, msk, title) in zip(axes[:3], slices):
        ax.imshow(np.rot90(img), cmap="gray")
        overlay = np.ma.masked_where(np.rot90(msk) == 0, np.rot90(msk))
        ax.imshow(overlay, cmap="autumn", alpha=0.55)
        ax.set_title(title)
        ax.axis("off")

    verts = np.asarray(face_mesh.vertices)
    if len(verts) > 35000:
        rng = np.random.default_rng(42)
        verts = verts[rng.choice(len(verts), 35000, replace=False)]
    views = [(0, 1, "xy face surface"), (0, 2, "xz face surface"), (1, 2, "yz face surface")]
    for ax, (i, j, title) in zip(axes[3:], views):
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
    head_mesh = mask_to_mesh(head_mask, affine, step_size=args.step_size)
    landmarks = mri_proxy_landmarks(np.asarray(head_mesh.vertices))

    if args.defaced:
        defaced_img, defaced_data, _defaced_affine = load_volume(args.defaced)
        validate_same_grid(img, defaced_img)
        face_mask, volume_params = build_deface_difference_mask(data, defaced_data, head_mask, args.diff_threshold)
        face_mesh = mask_to_mesh(face_mask, affine, step_size=args.step_size)
        surface_params = {"mode": "defaced_difference_surface_from_mask"}
    else:
        vertex_roi, surface_params = face_roi_world_mask(np.asarray(head_mesh.vertices), landmarks, args.front_percentile)
        face_mesh = submesh_from_vertex_mask(head_mesh, vertex_roi)
        face_mask, volume_params = build_face_volume_mask(head_mask, affine, landmarks, args.front_percentile)

    stem = f"{args.subject}_{args.session}"
    mask_path = args.output_dir / f"{stem}_face_roi_mask.nii.gz"
    mesh_path = args.output_dir / f"{stem}_face_surface.ply"
    qc_path = args.output_dir / f"{stem}_face_segmentation_qc.png"
    metadata_path = args.output_dir / f"{stem}_face_segmentation_metadata.json"

    out_img = nib.Nifti1Image(face_mask.astype(np.uint8), affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(mask_path))
    face_mesh.export(mesh_path)
    write_qc(data, head_mask, face_mask, face_mesh, qc_path)

    metadata = {
        "input": str(args.input),
        "defaced": str(args.defaced) if args.defaced else None,
        "subject": args.subject,
        "session": args.session,
        "threshold": threshold,
        "front_percentile": args.front_percentile,
        "head_voxels": int(head_mask.sum()),
        "face_voxels": int(face_mask.sum()),
        "head_mesh_vertices": int(len(head_mesh.vertices)),
        "face_mesh_vertices": int(len(face_mesh.vertices)),
        "face_mesh_faces": int(len(face_mesh.faces)),
        "landmarks": {name: point.tolist() for name, point in landmarks.items()},
        "surface_roi_params": surface_params,
        "volume_roi_params": volume_params,
        "outputs": {
            "mask": str(mask_path),
            "mesh": str(mesh_path),
            "qc": str(qc_path),
        },
        "warning": "Coarse MRI face target. Requires visual/manual QC before using as ground truth.",
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
    parser.add_argument("--front-percentile", type=float, default=72.0)
    parser.add_argument("--defaced", type=Path, default=None, help="Optional defaced copy on the same grid; use original-minus-defaced mask.")
    parser.add_argument("--diff-threshold", type=float, default=0.18, help="Normalized intensity drop threshold for --defaced mode.")
    parser.add_argument("--step-size", type=int, default=1)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
