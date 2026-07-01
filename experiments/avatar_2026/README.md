# Avatar 2026 Experiment Snapshot

This folder is a private experiment snapshot for the FaceAge-to-BrainAge avatar
workstream. It contains face images, face-derived avatar meshes, MRI-derived
outer-head surfaces, visual QC pages, and metric outputs.

## Active Dataset

`dataset_v0_three_people` uses three known folder labels:

- `1_1/photo`: 4 photos;
- `2_1/photos`: 5 photos;
- `3_1/photo`: 5 photos.

`1_2` and `1_3/photos` are deferred because they were empty locally at the time
of this snapshot.

Folder labels are treated as supervised labels. The pipeline does not infer
identity from face appearance.

## Key Entry Points

- `STATUS.md` - current experiment status.
- `reports/dataset_v0_three_people_manifest.csv` - active manifest.
- `reports/METRICS_AND_LABELS.md` - metric definitions and current results.
- `reports/TWIN_FACEAGE_LITERATURE_CONTEXT.md` - FaceAge/twin-study context.
- `project_page/index.html` - visual project page with GIFs and results.
- `auto_mri_overlay_v0/auto_mri_overlay_v0_contact_sheet.jpg` - automatic
  MRI/avatar overlay visual QC.
- `subject_consistency/crops_3subjects_3ddfa_1024/` - supervised
  same-folder vs different-folder consistency metrics.

## Main Outputs

- `photo_crops_3subjects_3ddfa_1024/` - standardized 1024 px face crops.
- `photo_avatar_crops_3subjects_3ddfa_v2/` - 3DDFA_V2 meshes and overlays.
- `photo_avatar_crops_3subjects_mediapipe/` - MediaPipe meshes and overlays.
- `mri_surfaces/` - MRI-derived outer-head surface and QC image.
- `landmark_alignment/` - automatic landmark-constrained MRI alignment previews.
- `project_page/assets/` - generated visual assets for the project page.

## Interpretation

The current 3DDFA/MediaPipe outputs are detector/QC baselines. They do not pass
the strict identity-separation criterion `genuine_p90 < impostor_p10`, so they
should not be described as identity-grade avatars.

The next experimental step is to add a stronger one-shot avatar method
(MeshLAM/LAM or MICA/DECA/EMOCA), rerun the same metrics, and compare against
this baseline.
