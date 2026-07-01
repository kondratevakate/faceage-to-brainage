# Landmark-Constrained MRI Alignment Report

## Purpose

Replace free ICP-only alignment with a stricter test: estimate the photo-to-MRI
similarity transform from semantic landmarks first, then measure distances to
the MRI anterior surface.

This is still a proxy experiment. The MRI landmarks here are automatically
estimated from the 2018 outer-head surface and are not manual anatomical
annotations.

## Inputs

- MRI surface: `../mri_surfaces/kate_2018_outer_head.ply`
- MediaPipe crop meshes: `../photo_avatar_crops_mediapipe/*.ply`
- 3DDFA_V2 crop meshes: `../photo_avatar_crops_3ddfa_v2/*.ply`

## Outputs

- `crops_mediapipe/landmark_constrained_summary.csv`
- `crops_mediapipe/landmark_constrained_summary.json`
- `crops_mediapipe/mri_proxy_landmarks.json`
- `crops_3ddfa_v2/landmark_constrained_summary.csv`
- `crops_3ddfa_v2/landmark_constrained_summary.json`
- `crops_3ddfa_v2/mri_proxy_landmarks.json`
- `*_landmark_constrained_alignment.png` previews in both method folders.

## Landmark Set

Used landmarks:

- `nose_tip`
- `chin`
- `brow_center`
- `left_cheek`
- `right_cheek`

The fitter tries all five landmarks, then leave-one-out subsets, and chooses the
lowest residual transform. Left/right cheek labels are allowed to swap because
image-space and MRI-space conventions can disagree.

Observed fit behavior:

| Method | Landmarks used for transform | Dropped landmark |
|---|---|---|
| MediaPipe FaceMesh | nose, chin, left cheek, right cheek | brow center |
| 3DDFA_V2 | nose, brow center, left cheek, right cheek | chin |

All current best fits use the left/right swapped convention.

## Result

Approximate ranges on the crop set:

| Method | Landmark RMSE | Surface median after landmark transform | Interpretation |
|---|---:|---:|---|
| MediaPipe FaceMesh | ~18-20 mm | ~8-10 mm | Landmark-level geometry is weak; face surface no longer overfits as easily as free ICP. |
| 3DDFA_V2 | ~21-22 mm | ~2.5 mm | Dense BFM surface sits close to MRI anterior cap, but landmark residuals show semantics are not solved. |

Mean residual diagnostics:

| Method | Main failing proxy landmark | Mean residual |
|---|---|---:|
| MediaPipe FaceMesh | `brow_center` | ~33.6 mm |
| 3DDFA_V2 | `chin` | ~39.8 mm |

## Interpretation

This pass is useful because it breaks the illusion from free ICP. The previous
ICP-only metric could report low distances by sliding generic oval face surfaces
onto the MRI anterior cap. Landmark-constrained alignment is stricter and shows
where the anatomical correspondence is still weak.

The current automatic MRI proxy landmarks are not good enough for final accuracy
claims. They are adequate for sanity checking and debugging, but not for saying
which one-photo avatar method is anatomically closer to MRI.

## Scientific Decision

Do not use this proxy landmark RMSE as the final ranking. Use it to justify the
next step: manual or semi-manual MRI landmark annotation.

Minimum manual landmark set:

1. nose tip;
2. subnasale or nose base;
3. chin / pogonion;
4. glabella or nasion;
5. left and right zygomatic/cheek points.

Once those are marked on the MRI surface, rerun the same constrained alignment
and report:

- landmark RMSE;
- source-to-MRI surface median/p90 on matched facial region;
- method stability across crops;
- visual alignment previews.
