"""
T1 MRI → cleaned 3D head mesh, with optional face extraction (BioFace3D pipeline).

extract_head_mesh():
  1. Load & reorient to RAS+          (nibabel)
  2. Histogram-match to IXI template  (SimpleITK)
  3. Predict isosurface threshold      (sklearn linear regression, threshold_model.joblib)
  4. Extract isosurface               (VTK vtkFlyingEdges3D)
  5. Remove disconnected components   (PyMeshLab — bioface3d/python/headCleaning.py)
  6. Hollow interior surfaces via AO  (PyMeshLab — bioface3d/python/emptyHead.py)
  7. Apply NIfTI affine → mm space    (RAS+)
  8. Save full head mesh              (PyVista, format from extension)

center_and_extract_face():
  Takes the full head PLY from extract_head_mesh() and produces a face-only PLY
  suitable for BioFace3D-20 MVCNN landmark detection (paper §2.3):
  1. Center bounding box to Cartesian origin
  2. Cut posterior and inferior regions via PyMeshLab face selection
  3. Remove remaining isolated fragments
"""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

from .utils import load_vol, reorient_to_ras

# ── Resource paths resolved relative to this file ────────────────────────────
_BIOFACE3D_ROOT = Path(__file__).resolve().parents[1] / "bioface3d"
_BIOFACE3D_PY   = _BIOFACE3D_ROOT / "python"
_DEFAULT_TEMPLATE = _BIOFACE3D_ROOT / "resources" / "IXI_template.nii"
_DEFAULT_MODEL    = _BIOFACE3D_PY  / "threshold_model.joblib"

# Add bioface3d/python to sys.path so we can import its helper functions
if str(_BIOFACE3D_PY) not in sys.path:
    sys.path.insert(0, str(_BIOFACE3D_PY))

from headCleaning import apply_transform_filter as _clean_mesh   # noqa: E402
from emptyHead   import apply_transform_filter as _hollow_mesh   # noqa: E402


# ── Private helpers ───────────────────────────────────────────────────────────

def _histogram_match(data: np.ndarray, template_path: Path) -> np.ndarray:
    """
    Match the intensity histogram of `data` (RAS+ float32 array) to the
    IXI template.  Logic from bioface3d/python/sliceEnhancement.py lines 15-25.

    SimpleITK expects arrays in (z, y, x) order; nibabel uses (x, y, z).
    We transpose in and out so the returned array is back in (x, y, z).
    """
    raw_sitk      = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    template_sitk = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    hm            = sitk.HistogramMatching(raw_sitk, template_sitk)
    return sitk.GetArrayFromImage(hm).transpose(2, 1, 0).astype(np.float32)


def _predict_threshold(data: np.ndarray, model_path: Path) -> float:
    """
    Predict the optimal isosurface threshold from scan intensity statistics.
    Logic from bioface3d/python/thresholdPrediction.py lines 20-35.

    Note: threshold is predicted from the ORIGINAL (pre-enhancement) data,
    matching the bioface3d convention (the model was trained on original data).
    """
    import pandas as pd

    model = joblib.load(str(model_path))
    df    = pd.DataFrame({"Int_Max": [float(data.max())],
                          "Int_Media": [float(data.mean())]})
    return float(model.predict(df)[0])


def _vtk_isosurface(data: np.ndarray, threshold: float) -> vtk.vtkPolyData | None:
    """
    Run vtkFlyingEdges3D on a numpy volume and return cleaned vtkPolyData.
    Logic from bioface3d/python/headReconstruction.py lines 14-48.

    The input array is expected in (x, y, z) nibabel order; VTK receives it
    in Fortran (column-major) ravel order so dimension indexing stays correct.
    """
    nx, ny, nz = data.shape
    flat      = data.ravel(order="F")
    vtk_array = numpy_support.numpy_to_vtk(flat, deep=True,
                                           array_type=vtk.VTK_FLOAT)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.GetPointData().SetScalars(vtk_array)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image_data)
    contour.SetNumberOfContours(1)
    contour.SetValue(0, threshold)
    contour.ComputeScalarsOff()
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.Update()

    if contour.GetOutput().GetNumberOfPoints() == 0:
        return None

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(contour.GetOutputPort())
    triangle.PassVertsOff()
    triangle.PassLinesOff()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())

    triangle2 = vtk.vtkTriangleFilter()
    triangle2.SetInputConnection(clean.GetOutputPort())
    triangle2.PassVertsOff()
    triangle2.PassLinesOff()
    triangle2.Update()

    return triangle2.GetOutput()


def _write_vtk_ply(polydata: vtk.vtkPolyData, path: str) -> None:
    """Write a vtkPolyData to an ASCII PLY file (PyMeshLab requires a file path)."""
    writer = vtk.vtkPLYWriter()
    writer.SetFileTypeToASCII()
    writer.SetInputData(polydata)
    writer.SetFileName(path)
    writer.Write()


# ── Public API ────────────────────────────────────────────────────────────────

def extract_head_mesh(
    nifti_path: str | Path,
    out_mesh: str | Path,
    ixi_template: str | Path = _DEFAULT_TEMPLATE,
    threshold_model: str | Path = _DEFAULT_MODEL,
) -> bool:
    """
    Extract a cleaned full-head 3D mesh from a T1 MRI scan.

    The output mesh is in RAS+ millimetre space and covers the complete head
    (no face crop) so that the MVCNN landmarker can operate on it directly.

    Parameters
    ----------
    nifti_path     : path to .nii / .nii.gz / .mgz
    out_mesh       : output mesh path (.ply recommended for Meshlab)
    ixi_template   : path to IXI_template.nii used for histogram matching
    threshold_model: path to threshold_model.joblib (sklearn linear regression)

    Returns
    -------
    True  – mesh saved successfully
    False – isosurface extraction failed (empty mesh or bad threshold)
    """
    out_mesh      = Path(out_mesh)
    ixi_template  = Path(ixi_template)
    threshold_model = Path(threshold_model)
    out_mesh.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & reorient ────────────────────────────────────────────────────
    img  = reorient_to_ras(load_vol(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    # ── 1b. Intensity normalisation ───────────────────────────────────────────
    # The threshold model was trained on IXI scans with ~12-bit range (0–4095).
    # Some scanners store raw float intensities several orders of magnitude
    # higher (e.g. SIMON sessions in the millions).  Normalise to 0–4095 via
    # 99th-percentile rescaling so the threshold prediction stays in range.
    # This is applied to both `data` (for threshold prediction) and feeds
    # naturally into histogram matching, which also expects IXI-like intensities.
    _IXI_MAX = 4095.0
    _p99 = float(np.percentile(data[data > 0], 99)) if (data > 0).any() else 1.0
    if data.max() > _IXI_MAX * 2:          # only rescale if clearly out of range
        data = np.clip(data, 0, _p99) / _p99 * _IXI_MAX

    # 2. Histogram matching
    matched = _histogram_match(data, ixi_template)
    
    # 2b. Bias correction
    processed = _n4_bias_correct(matched)
    
    # 3. Threshold prediction
    threshold = _predict_threshold(processed, threshold_model)
    
    # 4. FlyingEdges
    candidate_thresholds = [
        threshold,
        threshold * 0.9,
        threshold * 1.1,
        threshold * 0.8,
        threshold * 1.2,
    ]
    
    polydata = None
    for th in candidate_thresholds:
        pd = _vtk_isosurface(processed, th)
        if pd is not None and pd.GetNumberOfPoints() > 5000:
            polydata = pd
            threshold = th
            break
    
    if polydata is None:
        return False

    # ── 5 & 6. PyMeshLab cleanup + AO hollowing (require temp files) ──────────
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_ply    = str(Path(tmpdir) / "raw.ply")
        clean_ply  = str(Path(tmpdir) / "clean.ply")
        hollow_ply = str(Path(tmpdir) / "hollow.ply")

        _write_vtk_ply(polydata, raw_ply)
        _clean_mesh(raw_ply, clean_ply)
        _hollow_mesh(clean_ply, hollow_ply)

        mesh = pv.read(hollow_ply)

    # ── 7. Voxel → mm (apply NIfTI affine) ───────────────────────────────────
    affine  = img.affine.astype(np.float64)
    verts   = mesh.points
    ones    = np.ones((len(verts), 1), dtype=np.float64)
    verts_mm = (affine @ np.hstack([verts, ones]).T).T[:, :3]
    mesh.points = verts_mm

    # ── 8. Save ───────────────────────────────────────────────────────────────
    mesh.save(str(out_mesh))
    return True

def _make_n4_mask(data: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    nz = data[data > 0]
    lo = float(np.percentile(nz, 20)) if nz.size else 0.0
    hi = float(data.max()) if data.size else 1.0

    mask = sitk.BinaryThreshold(
        img, lowerThreshold=lo, upperThreshold=hi, insideValue=1, outsideValue=0
    )
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    return mask


def _n4_bias_correct(data: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    mask = _make_n4_mask(data)
    corrected = sitk.N4BiasFieldCorrection(img, mask)
    return sitk.GetArrayFromImage(corrected).transpose(2, 1, 0).astype(np.float32)


def center_and_extract_face(
    head_mesh_path: str | Path,
    out_face_mesh: str | Path,
    threshold_y: float = -10.0,
    threshold_z: float = -80.0,
) -> bool:
    """
    Center the full head mesh and crop to the face region.

    Required before BioFace3D-20 MVCNN landmark detection: the model was trained
    on cropped facial meshes, not full heads (BioFace3D paper §2.3).

    Pipeline:
      1. Load PLY (PyVista), translate bounding-box centroid to origin
      2. Select and remove posterior faces  (RAS+ Y < threshold_y)
         and inferior faces                 (RAS+ Z < threshold_z)
      3. Remove small isolated fragments    (PyMeshLab, diameter < 50%)
      4. Save face-only PLY

    Parameters
    ----------
    head_mesh_path : path to full head PLY produced by extract_head_mesh()
    out_face_mesh  : output face PLY path
    threshold_y    : RAS+ Y cut (mm, after centering). Faces with Y < threshold_y
                     are removed, keeping the anterior (face) half.
                     Default -10 empirically suits IXI; adjust if neck or ears included.
    threshold_z    : RAS+ Z cut (mm, after centering). Faces with Z < threshold_z
                     are removed, discarding the inferior neck region.
                     Default -80 retains chin while cutting the neck.

    Returns
    -------
    True  – face mesh saved successfully
    False – face mesh is empty after cropping (bad thresholds or bad input)
    """
    import pymeshlab as ml

    head_mesh_path = Path(head_mesh_path)
    out_face_mesh  = Path(out_face_mesh)
    out_face_mesh.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and center ────────────────────────────────────────────────────
    mesh = pv.read(str(head_mesh_path))
    bounds = mesh.bounds                        # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = np.array([
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    ])
    mesh.points = mesh.points - center          # translate bounding-box centroid to origin

    # ── 2 & 3. Face selection + cleanup via PyMeshLab ─────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        centered_ply = str(Path(tmpdir) / "centered.ply")
        mesh.save(centered_ply)

        ms = ml.MeshSet()
        ms.load_new_mesh(centered_ply)

        # Select faces in posterior or inferior regions (RAS+: Y=Anterior, Z=Superior)
        cond = f"(y1 < {threshold_y}) || (z1 < {threshold_z})"
        ms.compute_selection_by_condition_per_face(condselect=cond)
        ms.meshing_remove_selected_vertices_and_faces()

        # Remove isolated fragments smaller than 50% of the largest component
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=ml.PercentageValue(50), removeunref=True
        )

        if ms.current_mesh().vertex_number() == 0:
            return False

        # Decimate to ≤ target_faces to keep mesh density in the range the
        # BioFace3D-20 MVCNN depth renderer expects (~90K–220K vertices OK;
        # >300K causes RANSAC to fail on back-projection).
        _TARGET_FACES = 200_000
        if ms.current_mesh().face_number() > _TARGET_FACES:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=_TARGET_FACES,
                preservenormal=True,
                preservetopology=True,
            )

        ms.save_current_mesh(str(out_face_mesh), binary=False,
                             save_vertex_color=False, save_vertex_quality=False)

    # ── 4. Convert RAS+ → LSA (Left-Superior-Anterior, BioFace3D training space) ──
    # BioFace3D-20 MVCNN was trained on meshes in LSA orientation.
    # The render camera in render3d.py sits at (0,0,500) looking toward -Z,
    # so the face must be at +Z (LSA Anterior).
    # Mapping: LSA_X = -RAS_X,  LSA_Y = RAS_Z,  LSA_Z = RAS_Y
    reloaded = pv.read(str(out_face_mesh))
    pts = reloaded.points
    reloaded.points = np.column_stack([-pts[:, 0], pts[:, 2], pts[:, 1]])
    reloaded.save(str(out_face_mesh))

    return True
