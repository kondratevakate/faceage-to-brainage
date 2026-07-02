# Avatar 2026 Status

Updated: 2026-07-02.

## Public Scope

The public repository now exposes a single-subject case study through the
project page:

- `project_page/index.html`
- `project_page/assets/`
- `reports/METRICS_AND_LABELS.md`
- `reports/TWIN_FACEAGE_LITERATURE_CONTEXT.md`

The GitHub Pages version is published from the separate `gh-pages` branch.

## Local-Only Scope

The local workspace may contain additional photos, MRI surfaces, face crops,
meshes, overlays, CSV manifests, and internal control-subject outputs. These are
ignored by git. They are working data, not public project claims.

## Current Scientific Position

3DDFA and MediaPipe are calibration baselines. They are useful for testing the
pipeline contract, but they are not enough to claim high-fidelity avatar
reconstruction.

The current MRI comparison is a geometry sanity check. It needs better MRI face
masks, anatomical landmarks, and posture-aware interpretation before any
anatomical accuracy claim is made.

## Next Work

1. Add robust surface metrics: directed Hausdorff, HD95, ASSD, Chamfer, and
   regional masked distances.
2. Add a stronger reconstruction baseline under the same metric contract.
3. Keep public visuals case-only; keep internal controls non-public unless
   explicitly consented and curated.
