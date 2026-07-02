# Avatar 2026 Case Study

This folder contains the public, claim-facing part of the 2026
FaceAge-to-BrainAge avatar workstream.

The scientific question is deliberately narrow:

> Can a one-photo facial avatar be evaluated against the same subject's
> MRI-derived face surface instead of being judged only by visual plausibility?

## What Belongs In Git

- `project_page/` - the public visual article page and its generated assets.
- `reports/METRICS_AND_LABELS.md` - metric definitions and interpretation rules.
- `reports/TWIN_FACEAGE_LITERATURE_CONTEXT.md` - FaceAge/twin-study context.
- `README.md` and `STATUS.md` - current scope and limitations.

## What Stays Local/Ignored

Raw or generated face/MRI artifacts are intentionally not tracked:

- face crops and photo inventories;
- MediaPipe/3DDFA meshes and overlays;
- MRI-derived PLY surfaces and proxy landmarks;
- per-photo alignment dumps;
- CSV manifests and batch/stability outputs;
- internal multi-subject validation artifacts.

Those files are useful for local reproduction, but they are not the public
scientific surface of the project.

## Interpretation

Current outputs are calibration baselines. They establish detection, alignment,
masking, and reporting conventions. They should not be described as
identity-grade avatars or validated biological-age measurements.

The project page separates four claims that must not be collapsed:

1. geometric agreement with MRI-derived surface;
2. perceptual/avatar plausibility;
3. identity consistency under controlled labels;
4. biological-age validity.

That separation is the main methodological contribution of this snapshot.
