# Batch Photo Avatar Report: 2026-04-09

## Inputs

- `../1_1/photo`: 4 images.
- `../2_1/photos`: 5 images.
- Total: 9 images.

## Baselines Run

| Method | Output folder | Success |
|---|---|---:|
| MediaPipe FaceMesh | `../../photo_avatar_batch_mediapipe` | 5 / 9 |
| 3DDFA_V2 | `../../photo_avatar_batch_3ddfa_v2` | 9 / 9 |

MediaPipe failed on four images where the face was too small, distant, or not
well suited to its face-landmarker assumptions. 3DDFA_V2 detected every image,
but several detections are on very small faces and should not be treated as good
avatar inputs.

## Outputs

- `photo_inventory_batch.csv`
- `baseline_detection_summary.csv`
- `mesh_stability_summary.csv`
- `mesh_stability_pairs.csv`
- `overlay_contact_sheet.jpg`

MRI comparison outputs:

- `../../comparisons/mri_anterior_y_batch_mediapipe/photo_mesh_to_mri_summary.csv`
- `../../comparisons/mri_anterior_y_batch_3ddfa_v2_sampled/photo_mesh_to_mri_summary.csv`

The 3DDFA_V2 MRI batch comparison uses sampled source vertices
(`--source-sample 5000`) for triage speed.

## Shortlist

| Rank | Image | Why |
|---:|---|---|
| 1 | `1_1/photo/photo_2026-04-09_19-36-57.jpg` | Largest face area; both methods reconstruct; best close frontal baseline, but headphones occlude ears/hair. |
| 2 | `2_1/photos/photo_2_2026-04-09_20-38-32.jpg` | Both methods reconstruct; stable 3DDFA mesh; usable selfie angle. |
| 3 | `2_1/photos/photo_3_2026-04-09_20-38-32.jpg` | Both methods reconstruct; best MediaPipe stability; useful as alternate pose/gaze. |

Use with caution:

- `2_1/photos/photo_2026-04-09_20-37-59.jpg`: face is large enough, but tilted; MediaPipe stability is worse.
- `2_1/photos/photo_5_2026-04-09_20-38-32.jpg`: face is large enough, but hand occludes lower face.

Exclude for one-photo avatar quality ranking:

- `1_1/photo/photo_2026-04-09_19-37-48.jpg`
- `1_1/photo/photo_2026-04-09_19-37-56.jpg`
- `1_1/photo/photo_2026-04-09_19-38-04.jpg`
- `2_1/photos/photo_4_2026-04-09_20-38-32.jpg`

These images are useful as detector stress tests, not as primary avatar inputs:
the face occupies less than 1% of the frame for 3DDFA detections.

## Interpretation

3DDFA_V2 is more robust for detection across the whole dataset. MediaPipe is
more selective and fails on distant/full-body cases, which is helpful as a
quality filter.

The MRI point-to-surface score should not be used alone for ranking. Small-face
photos can get plausible MRI scores after similarity ICP, even though they are
bad avatar inputs. For the next experiment, rank inputs using:

1. face area and visual overlay quality;
2. within-method mesh stability across photos;
3. MRI comparison only after landmark-constrained alignment.
