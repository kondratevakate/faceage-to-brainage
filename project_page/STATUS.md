# Project Page Status

Updated: 2026-07-02.

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

## Current Scientific Position

3DDFA and MediaPipe are calibration baselines. They are useful for testing the
pipeline contract, but they are not enough to claim high-fidelity avatar
reconstruction.

The current MRI comparison is a geometry sanity check. It needs better MRI face
masks, anatomical landmarks, and posture-aware interpretation before any
anatomical accuracy claim is made.

SOTA avatar preflight was added on 2026-07-02. The current local machine does
not expose an NVIDIA/CUDA runtime and the project environment does not yet have
`torch`, so LAM/GAGAvatar-style Gaussian-avatar methods are not locally runnable
here. DECA is the most realistic next geometry baseline once FLAME
`generic_model.pkl` and `deca_model.tar` are provided. MeshLAM is the strongest
conceptual target for a mesh+texture baseline, but no separate runnable MeshLAM
checkout/weights are available locally yet.

## Next Work

1. Add robust surface metrics: directed Hausdorff, HD95, ASSD, Chamfer, and
   regional masked distances.
2. Add a stronger reconstruction baseline under the same metric contract:
   DECA/MICA/EMOCA for MRI geometry first, then LAM/GAGAvatar/MeshLAM for
   perceptual avatar quality on a CUDA machine.
3. Keep public visuals case-only; keep internal controls non-public unless
   explicitly consented and curated.
