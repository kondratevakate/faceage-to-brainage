"""
MRI volume → 2D frontal face render.

Pipeline:
  1. Load & reorient to RAS+
  2. Marching-cubes isosurface at skin threshold
  3. Keep only the anterior portion (face region)
  4. PyVista offscreen render with Phong shading → RGB PNG

The output PNG is suitable for MTCNN face detection used by FaceAge.
"""
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from .utils import load_vol, reorient_to_ras


# Default skin-intensity threshold for T1-weighted MRI (in scanner units).
# T1 soft tissue/fat ~300-800; background ~0-50.
# A threshold of 40 captures skin surface reliably across IXI/SIMON.
DEFAULT_SKIN_LEVEL = 40

# Fraction of volume depth (anterior) kept for face crop.
# In RAS+ space the face is at high Y values.
FACE_CROP_FRACTION = 0.45


def _extract_surface(data: np.ndarray, level: float) -> tuple:
    """Run marching cubes and return (verts, faces, normals)."""
    verts, faces, normals, _ = marching_cubes(data, level=level, allow_degenerate=False)
    return verts, faces, normals


def _crop_to_face(verts: np.ndarray, faces: np.ndarray, data_shape: tuple) -> tuple:
    """
    Keep only vertices in the anterior portion of the volume (face region).
    In RAS+ orientation, anterior = high Y index.
    """
    ny = data_shape[1]
    y_min = ny * (1.0 - FACE_CROP_FRACTION)
    mask_verts = verts[:, 1] >= y_min

    # Remap faces that have all three vertices in the mask
    old_to_new = np.full(len(verts), -1, dtype=np.int64)
    new_idx = np.where(mask_verts)[0]
    old_to_new[new_idx] = np.arange(len(new_idx))

    valid_faces_mask = np.all(mask_verts[faces], axis=1)
    new_faces = old_to_new[faces[valid_faces_mask]]
    new_verts = verts[new_idx]
    return new_verts, new_faces


def _build_pyvista_mesh(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Convert marching-cubes output to PyVista PolyData."""
    n_faces = len(faces)
    cells = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces]).ravel()
    cell_type = np.full(n_faces, 5, dtype=np.uint8)  # VTK_TRIANGLE
    mesh = pv.UnstructuredGrid(cells, cell_type, verts.astype(float))
    return mesh.extract_surface()


def render_face(
    nifti_path: str | Path,
    out_png: str | Path,
    level: float = DEFAULT_SKIN_LEVEL,
    image_size: int = 512,
    crop_to_face: bool = True,
) -> bool:
    """
    Render a frontal 2D face image from a T1 MRI volume.

    Parameters
    ----------
    nifti_path : path to .nii / .nii.gz / .mgz
    out_png    : output PNG path
    level      : marching-cubes isosurface threshold (scanner units)
    image_size : output image resolution (square)
    crop_to_face : if True, keep only the anterior face region before rendering

    Returns
    -------
    True  – render saved successfully
    False – surface extraction failed (e.g. empty mesh, wrong threshold)
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    img = reorient_to_ras(load_vol(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    try:
        verts, faces, _ = _extract_surface(data, level)
    except (ValueError, RuntimeError):
        return False

    if len(verts) == 0 or len(faces) == 0:
        return False

    if crop_to_face:
        verts, faces = _crop_to_face(verts, faces, data.shape)
        if len(verts) < 100 or len(faces) < 50:
            return False

    mesh = _build_pyvista_mesh(verts, faces)

    # Offscreen render — works in Colab / headless environments
    pv.global_theme.allow_empty_mesh = True
    pl = pv.Plotter(off_screen=True, window_size=[image_size, image_size])
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        color=[210 / 255, 180 / 255, 140 / 255],  # tan / skin-like
        smooth_shading=True,
        ambient=0.3,
        diffuse=0.7,
        specular=0.1,
    )

    # Camera: frontal view — looking in -Y direction in RAS space
    pl.camera_position = "yz"
    pl.camera.roll = 0
    pl.reset_camera()

    img_arr = pl.screenshot(None, return_img=True)
    pl.close()

    from PIL import Image

    Image.fromarray(img_arr).save(str(out_png))
    return True


def render_multicontrast(
    t1_path: str | Path,
    t2_path: str | Path,
    pd_path: str | Path,
    out_png: str | Path,
    level: float = DEFAULT_SKIN_LEVEL,
    image_size: int = 512,
) -> bool:
    """
    Render a face image where surface vertex colors encode multi-contrast info:
      R channel ← T1 intensity
      G channel ← T2 intensity
      B channel ← PD intensity

    T2 and PD must already be registered to T1 space (same voxel grid).
    """
    from scipy.ndimage import map_coordinates

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t1 = reorient_to_ras(load_vol(t1_path))
    t1_data = t1.get_fdata(dtype=np.float32)

    t2_data = reorient_to_ras(load_vol(t2_path)).get_fdata(dtype=np.float32)
    pd_data = reorient_to_ras(load_vol(pd_path)).get_fdata(dtype=np.float32)

    try:
        verts, faces, _ = _extract_surface(t1_data, level)
    except (ValueError, RuntimeError):
        return False

    verts, faces = _crop_to_face(verts, faces, t1_data.shape)
    if len(verts) < 100:
        return False

    def _sample(vol: np.ndarray, pts: np.ndarray) -> np.ndarray:
        coords = pts.T  # shape (3, N)
        vals = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)
        # Normalize to [0, 1]
        vmax = np.percentile(vals, 99)
        vmin = vals.min()
        return np.clip((vals - vmin) / (vmax - vmin + 1e-8), 0, 1)

    r = _sample(t1_data, verts)
    g = _sample(t2_data, verts)
    b = _sample(pd_data, verts)
    colors = np.stack([r, g, b], axis=1)  # (N, 3) float in [0,1]

    mesh = _build_pyvista_mesh(verts, faces)
    mesh.point_data["RGB"] = (colors * 255).astype(np.uint8)

    pl = pv.Plotter(off_screen=True, window_size=[image_size, image_size])
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="RGB", rgb=True, smooth_shading=True)
    pl.camera_position = "yz"
    pl.reset_camera()

    img_arr = pl.screenshot(None, return_img=True)
    pl.close()

    from PIL import Image

    Image.fromarray(img_arr).save(str(out_png))
    return True
