#!/usr/bin/env python3
"""
End-to-end test: one IXI T1 scan → two head meshes for side-by-side Meshlab comparison.

Usage (run from repo root):
    python scripts/test_mesh_export.py

Outputs written to results/poc_ixi/:
    face_render.png        — 2-D frontal render (sanity check)
    face_mesh.ply          — baseline: raw marching-cubes face crop (export_mesh)
    head_mesh_bioface.ply  — improved: histogram-matched, adaptive threshold,
                             FlyingEdges3D, PyMeshLab cleanup + AO hollowing,
                             full head for MVCNN (extract_head_mesh)
"""
import sys
from pathlib import Path

import pyvista as pv

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.render          import render_face, export_mesh
from src.head_extraction import extract_head_mesh

REPO_ROOT       = Path(__file__).resolve().parent.parent
SCAN_PATH       = REPO_ROOT.parent / "data/IXI/IXI002-Guys-0828-T1.nii.gz"
OUT_DIR         = REPO_ROOT / "results/poc_ixi"
LEVEL           = 40

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_png         = OUT_DIR / "face_render.png"
out_mesh        = OUT_DIR / "face_mesh.ply"
out_mesh_bio    = OUT_DIR / "head_mesh_bioface.ply"

print(f"Scan : {SCAN_PATH}  exists={SCAN_PATH.exists()}\n")


def _print_mesh_stats(label: str, path: Path) -> None:
    m      = pv.read(str(path))
    bounds = m.bounds
    print(f"  {label}")
    print(f"    Vertices : {m.n_points:,}")
    print(f"    Faces    : {m.n_cells:,}")
    print(f"    Bounds X : {bounds[0]:.1f} – {bounds[1]:.1f} mm")
    print(f"    Bounds Y : {bounds[2]:.1f} – {bounds[3]:.1f} mm")
    print(f"    Bounds Z : {bounds[4]:.1f} – {bounds[5]:.1f} mm")


# ── 1. 2-D render (sanity check) ─────────────────────────────────────────────
print("[1/3] Rendering 2-D face PNG ...")
ok_png = render_face(SCAN_PATH, out_png, level=LEVEL)
print(f"      {'OK  ' + str(out_png) if ok_png else 'FAILED'}\n")

# ── 2. Baseline 3-D mesh (export_mesh) ───────────────────────────────────────
print("[2/3] Baseline mesh  (export_mesh — raw marching cubes, face crop) ...")
ok_base = export_mesh(SCAN_PATH, out_mesh, level=LEVEL)
print(f"      {'OK  ' + str(out_mesh) if ok_base else 'FAILED'}\n")

# ── 3. Improved 3-D head mesh (bioface3d pipeline) ───────────────────────────
print("[3/3] Bioface3D head mesh  (histogram match → FlyingEdges3D → PyMeshLab) ...")
ok_bio = extract_head_mesh(SCAN_PATH, out_mesh_bio)
print(f"      {'OK  ' + str(out_mesh_bio) if ok_bio else 'FAILED'}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 56)
print("Mesh stats")
print("=" * 56)
if ok_base:
    _print_mesh_stats("face_mesh.ply         (baseline)", out_mesh)
if ok_bio:
    _print_mesh_stats("head_mesh_bioface.ply (bioface3d)", out_mesh_bio)

print("\nOpen both in Meshlab to compare:")
if ok_base:
    print(f"  meshlab {out_mesh}")
if ok_bio:
    print(f"  meshlab {out_mesh_bio}")
