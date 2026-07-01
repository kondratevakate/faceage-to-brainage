# Avatar 2026 Pilot Status

Updated: 2026-06-27.

## Purpose

Private working folder for matching photo-derived face/avatar geometry against
MRI-derived outer-head surfaces from the same personal MRI series.

## Current Inputs

MRI source folders:

- `../1_1/mri/3_fspgr_bravo_10mm_ax.nii.gz` - 2018, best current MRI surface baseline.
- `../1_3/nii/401_t1w_ffe.nii.gz` - 2024 FFE candidate, low slice count.
- `../1_3/nii/601_t1w_ffe.nii.gz` - 2024 FFE candidate, low slice count.

Photo source folders:

- `../1_1/photo/` - included in `dataset_v0_three_people` as known folder label `1_1` (4 photos).
- `../2_1/photos/` - included in `dataset_v0_three_people` as known folder label `2_1` (5 photos).
- `../3_1/photo/` - included in `dataset_v0_three_people` as known folder label `3_1` (5 photos).
- `../1_2/` - deferred; empty locally, not blocking `dataset_v0_three_people`.
- `../1_3/photos/` - deferred; empty locally, not blocking `dataset_v0_three_people`.

Important: folder labels are treated as known supervised labels for testing.
The pipeline does not infer who is who from face appearance.

## Generated Artifacts

MRI surfaces:

- `mri_surfaces/kate_2018_outer_head.ply`
- `mri_surfaces/*_qc.png`
- `mri_surfaces/*_metadata.json`

Note: 2024 FFE meshes were explored in repo-local output, but they are not
currently present in this Yandex private workspace. Use 2018 as the active
baseline until regenerated here.

Photo inventory:

- `photo_manifest/photo_inventory.csv`
- `photo_manifest/photo_inventory.json`
- `photo_manifest/photo_contact_sheet.jpg`
- `photo_manifest/selected/`
- `reports/dataset_v0_three_people_manifest.csv` - active three-folder test manifest; 14 included photos plus deferred non-blocking sources.

Photo-derived lightweight face meshes:

- `photo_avatar/kate_frontal_2026_04_09_mediapipe_facemesh.ply`
- `photo_avatar/kate_frontal_low_2026_04_09_mediapipe_facemesh.ply`
- `photo_avatar/kate_gaze_side_2026_04_09_mediapipe_facemesh.ply`
- matching `*_landmarks_overlay.jpg`, `*_facemesh_preview.png`, and `*_metadata.json`

Photo-derived 3DDFA_V2 dense face meshes:

- `photo_avatar_3ddfa_v2/*_3ddfa_v2_face1.ply`
- matching `*_overlay.jpg`, `*_metadata.json`
- `photo_avatar_3ddfa_v2/kate_selected_2026_04_09_3ddfa_v2_summary.json`

Batch photo sweep:

- `photo_avatar_batch_mediapipe/` - MediaPipe FaceMesh on all 2026 photos where detection succeeded.
- `photo_avatar_batch_3ddfa_v2/` - 3DDFA_V2 dense meshes on all 9 photos.
- `stability/batch_2026_04_09/` - inventory, detection summary, mesh stability, contact sheet, and batch report.

Face-crop sweep:

- `photo_crops_3ddfa_1024/` - 1024 px standardized crops from 3DDFA detector boxes.
- `photo_avatar_crops_mediapipe/` - MediaPipe FaceMesh on crops.
- `photo_avatar_crops_3ddfa_v2/` - 3DDFA_V2 on crops.
- `stability/crops_3ddfa_1024/` - crop stability report and overlay contact sheet.

Landmark-constrained MRI alignment:

- `landmark_alignment/crops_mediapipe/landmark_constrained_summary.csv`
- `landmark_alignment/crops_3ddfa_v2/landmark_constrained_summary.csv`
- `landmark_alignment/LANDMARK_ALIGNMENT_REPORT.md`
- `landmark_alignment/*/*_landmark_constrained_alignment.png`

Subject-aware consistency:

- `subject_consistency/crops_3ddfa_1024/subject_consistency_summary.csv`
- `subject_consistency/crops_3ddfa_1024_quality4/subject_consistency_summary.csv`
- `subject_consistency/subject_consistency_distributions.jpg`
- `subject_consistency/SUBJECT_CONSISTENCY_REPORT.md`
- `subject_consistency/crops_3subjects_3ddfa_1024/subject_consistency_summary.csv` - active `dataset_v0_three_people` supervised same-folder vs different-folder result.

Comparisons:

- `comparisons/mri_anterior_y/photo_mesh_to_mri_summary.csv`
- `comparisons/mri_anterior_y_3ddfa_v2/photo_mesh_to_mri_summary.csv`
- `comparisons/mri_anterior_y_batch_mediapipe/photo_mesh_to_mri_summary.csv`
- `comparisons/mri_anterior_y_batch_3ddfa_v2_sampled/photo_mesh_to_mri_summary.csv`
- `comparisons/mri_anterior_y_crops_mediapipe/photo_mesh_to_mri_summary.csv`
- `comparisons/mri_anterior_y_crops_3ddfa_v2_sampled/photo_mesh_to_mri_summary.csv`

## Interpretation

The MediaPipe meshes are fast single-photo baselines. They are useful for
landmarks, rough face shape, and photo-side sanity checks. They are not metric
3D ground truth and do not reconstruct ears, full skull, hair, or back of head.

The 3DDFA_V2 meshes are dense BFM face-surface baselines. They reconstruct a
plausible face oval and nose/mouth/eye region, but still do not reconstruct
ears, skull, hair, or the back of the head.

Current 2018 MRI anterior-cap comparison:

| Method | Vertices | Best median mm | Best p90 mm | Interpretation |
|---|---:|---:|---:|---|
| MediaPipe FaceMesh | 478 | 1.940 | 5.613 | Fast lower-bound face mesh; sparse landmarks. |
| 3DDFA_V2 | 38365 | 2.035 | 5.757 | Dense face mesh; visually plausible, not yet a better MRI match. |

These distances are not a final avatar-quality ranking. Similarity ICP against
an MRI anterior cap can overfit generic oval face surfaces without anatomical
landmark constraints. Treat the current comparison as a pipeline sanity check:
photo meshes align to the expected MRI front region, but geometry accuracy is
not proven.

Batch 2026-04-09 result:

| Method | Success | Note |
|---|---:|---|
| MediaPipe FaceMesh | 5 / 9 | Selective; fails on distant/small faces. |
| 3DDFA_V2 | 9 / 9 | Robust detector; several outputs are too small-face for avatar ranking. |

Dataset v0 three-people result:

| Source | Count | Status |
|---|---:|---|
| `1_1/photo` | 4 | included |
| `2_1/photos` | 5 | included |
| `3_1/photo` | 5 | included |
| `1_2` | 0 | deferred, not blocking |
| `1_3/photos` | 0 | deferred, not blocking |

Current supervised consistency result for `dataset_v0_three_people`: no
MediaPipe/3DDFA metric passes the strict `genuine_p90 < impostor_p10` criterion.
This makes the current outputs a useful detector/QC baseline, not an
identity-grade avatar baseline.

Crop 2026-04-09 result:

| Method | Success | Note |
|---|---:|---|
| MediaPipe FaceMesh | 9 / 9 | Cropping fixes the detector failures. |
| 3DDFA_V2 | 9 / 9 | Clean rerun on standardized face crops. |

Current input shortlist:

1. `photo_crops_3ddfa_1024/1_1_photo_2026-04-09_19-38-04_facecrop.jpg` - good frontal crop without headphones.
2. `photo_crops_3ddfa_1024/2_1_photo_2_2026-04-09_20-38-32_facecrop.jpg` - stable selfie crop.
3. `photo_crops_3ddfa_1024/2_1_photo_3_2026-04-09_20-38-32_facecrop.jpg` - stable alternate crop.
4. `photo_crops_3ddfa_1024/1_1_photo_2026-04-09_19-36-57_facecrop.jpg` - controlled close frontal, but headphones occlude ears/hair.

Landmark-constrained alignment result:

| Method | Landmark RMSE | Surface median after landmark transform | Interpretation |
|---|---:|---:|---|
| MediaPipe FaceMesh | ~18-20 mm | ~8-10 mm | Useful sanity check; weak metric geometry. |
| 3DDFA_V2 | ~21-22 mm | ~2.5 mm | Dense surface stays close to MRI cap, but landmarks show semantic mismatch. |

The automatic MRI proxy landmarks are not accurate enough for final ranking.
Diagnostics show `brow_center` is the main failing proxy for MediaPipe and
`chin` for 3DDFA_V2. The next accuracy step is manual or semi-manual MRI
landmark annotation.

Subject-aware consistency result:

| Method | Dataset | Identity separation |
|---|---|---|
| MediaPipe FaceMesh | full crops | fail |
| MediaPipe FaceMesh | quality-filtered crops | fail |
| 3DDFA_V2 | full crops | fail |
| 3DDFA_V2 | quality-filtered crops | fail |

Using folder labels as known subject IDs, genuine and impostor pair
distributions overlap. This means current MediaPipe/3DDFA geometry is not
identity-separable enough for a Face-ID-style avatar constraint.

The MRI meshes are face-reconstructable private outputs. Do not publish or
commit them without explicit review.

## Next

1. Add anatomical constraints before claiming geometry quality: manual or
   semi-automatic nose/chin/forehead/ear landmarks on MRI and photo meshes.
2. Add a stronger identity-preserving single-photo avatar baseline: MICA/DECA/EMOCA
   after FLAME/model assets are available, or LAM/MeshLAM on CUDA.
3. Split evaluation into geometry, identity, and perception instead of using one
   point-to-surface number.
