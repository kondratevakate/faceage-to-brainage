# Face-Crop Avatar Report

## Purpose

Test whether 3DDFA-based face crops make the full 2026 photo set usable for
single-photo avatar baselines.

## Inputs and Outputs

Input crops:

- `../../photo_crops_3ddfa_1024/*_facecrop.jpg`
- `../../photo_crops_3ddfa_1024/face_crops_manifest.csv`

Baselines on crops:

| Method | Output folder | Success |
|---|---|---:|
| MediaPipe FaceMesh | `../../photo_avatar_crops_mediapipe` | 9 / 9 |
| 3DDFA_V2 | `../../photo_avatar_crops_3ddfa_v2` | 9 / 9 |

Reports:

- `photo_inventory_batch.csv`
- `baseline_detection_summary.csv`
- `mesh_stability_summary.csv`
- `mesh_stability_pairs.csv`
- `crop_overlay_contact_sheet.jpg`

MRI comparisons:

- `../../comparisons/mri_anterior_y_crops_mediapipe/photo_mesh_to_mri_summary.csv`
- `../../comparisons/mri_anterior_y_crops_3ddfa_v2_sampled/photo_mesh_to_mri_summary.csv`

## Main Result

Face cropping changes the dataset from partially usable to fully runnable:
MediaPipe improves from 5 / 9 successful detections on original photos to 9 / 9
on crops. This makes face cropping a required preprocessing step for future
one-photo avatar experiments on this dataset.

## Updated Shortlist

| Rank | Crop | Why |
|---:|---|---|
| 1 | `1_1_photo_2026-04-09_19-38-04_facecrop.jpg` | Good frontal crop without headphones; best 3DDFA stability among visually usable crops. |
| 2 | `2_1_photo_2_2026-04-09_20-38-32_facecrop.jpg` | Stable selfie crop; good practical avatar input. |
| 3 | `2_1_photo_3_2026-04-09_20-38-32_facecrop.jpg` | Stable alternate crop; useful for pose/gaze robustness. |
| 4 | `1_1_photo_2026-04-09_19-36-57_facecrop.jpg` | Controlled close frontal face; headphones occlude ears/hair, so better for face-only than full-head avatar. |

Use with caution:

- `1_1_photo_2026-04-09_19-37-56_facecrop.jpg`: usable, but tilted and mirror/social-photo context.
- `2_1_photo_2026-04-09_20-37-59_facecrop.jpg`: tilted; weakest 3DDFA crop stability.
- `2_1_photo_5_2026-04-09_20-38-32_facecrop.jpg`: lower face is occluded by hand.

Exclude from primary one-photo avatar quality ranking:

- `1_1_photo_2026-04-09_19-37-48_facecrop.jpg`: eyes closed.
- `2_1_photo_4_2026-04-09_20-38-32_facecrop.jpg`: strong profile view; useful for robustness, not primary frontal reconstruction.

## Interpretation

Cropping improves detector success and makes the baselines comparable across the
photo set. It does not solve the core metric problem: MRI point-to-surface
scores remain a weak ranking signal because similarity ICP can still align
generic face ovals.

The next scientifically useful step is landmark-constrained MRI alignment on
the updated shortlist, using crop-based MediaPipe landmarks and approximate MRI
surface landmarks before any point-to-surface distance is reported.
