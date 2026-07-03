# Project Page Status

Updated: 2026-07-03.

## Public Scope

The public repository now exposes a single-subject case study through the
project page:

- `project_page/index.html`
- `project_page/assets/`
- `project_page/METRICS_AND_LABELS.md`
- `project_page/TWIN_FACEAGE_LITERATURE_CONTEXT.md`

The GitHub Pages version is published from the separate `gh-pages` branch.

## Local-Only Scope

The local workspace may contain additional photos, MRI surfaces, face crops,
meshes, overlays, CSV manifests, and internal control-subject outputs. These are
ignored by git. They are working data, not public project claims.

Current local workbench path: `data/avatar_2026_work/`.

## Current Visible Result

The clearest current result is the Case A photo-avatar preprocessing baseline:

- `project_page/assets/case_a_mask_overlays.jpg`
- `project_page/assets/kate_mesh_turntable.gif`

The avatar side is useful for QC and method development. The MRI face target is
currently not reliable enough for public avatar-to-MRI accuracy figures:
automatic MRI face segmentation is the blocker.

## Current Scientific Position

3DDFA and MediaPipe are calibration baselines. They are useful for testing the
pipeline contract, but they are not enough to claim high-fidelity avatar
reconstruction.

The current MRI comparison is suspended as a public result. It needs better MRI
face masks, anatomical landmarks, and posture-aware interpretation before any
anatomical accuracy claim is made.

MRI face-target utility added on 2026-07-03:
`scripts/photo_mri_avatar/segment_mri_face_target.py`. The preferred mode is
`original - defaced` if a defaced copy exists on the same voxel grid. A coarse
surface-shell fallback was run locally for the 2018 Case A MRI:
`data/avatar_2026_work/mri_face_segmentation/kate_2018_surface_shell_v2/`.
This output is draft QC only, not a ground-truth target.

SOTA avatar preflight was added on 2026-07-02. The current local machine does
not expose an NVIDIA/CUDA runtime, so LAM/GAGAvatar-style Gaussian-avatar
methods are not locally runnable here. A separate CPU-only avatar environment is
available at `D:\projects\02_academia\_external\.venvs\avatar_cpu_py311`.
DECA `deca_model.tar`, MICA `mica.tar`, and MICA InsightFace assets are local
external assets. DECA and MICA now have project runners for CPU geometry-only
FLAME export, but both still require the licensed FLAME `generic_model.pkl`.
MeshLAM is the strongest conceptual target for a mesh+texture baseline, but no
separate runnable MeshLAM checkout/weights are available locally yet.

A privacy-minimal cloud bundle path is available locally:
`data/avatar_2026_work/cloud_bundles/avatar_case_1_1_latest.zip`. It contains
only the primary case crops, no internal controls and no MRI surfaces.

## Next Work

1. Add robust surface metrics: directed Hausdorff, HD95, ASSD, Chamfer, and
   regional masked distances.
2. Add a stronger reconstruction baseline under the same metric contract:
   DECA CPU geometry first after the licensed FLAME model is placed locally,
   then MICA CPU geometry as the metric-FLAME baseline, then
   LAM/GAGAvatar/MeshLAM for perceptual avatar quality on a CUDA machine.
3. Use `scripts/photo_mri_avatar/CLOUD_RUNBOOK.md` for Colab/AWS transfer and
   return outputs into `data/avatar_2026_work/photo_avatar_<method>/`.
4. Keep public visuals case-only; keep internal controls non-public unless
   explicitly consented and curated.
