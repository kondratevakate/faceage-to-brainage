# Subject-Aware Avatar Consistency Report

## Purpose

Test whether current one-photo avatar meshes preserve subject identity well
enough that meshes from the same folder are closer than meshes from different
folders.

This is a supervised test: folder labels are treated as known subject labels.
The pipeline does not infer identity from faces.

## Inputs

Subject labels:

- `1_1_*` crop files -> subject label `1_1`
- `2_1_*` crop files -> subject label `2_1`

Methods:

- MediaPipe FaceMesh crop outputs.
- 3DDFA_V2 crop outputs.

Reports:

- `crops_3ddfa_1024/subject_consistency_pairs.csv`
- `crops_3ddfa_1024/subject_consistency_summary.csv`
- `crops_3ddfa_1024_quality4/subject_consistency_pairs.csv`
- `crops_3ddfa_1024_quality4/subject_consistency_summary.csv`
- `subject_consistency_distributions.jpg`

## Metrics

Pair labels:

- `genuine`: same folder/subject label.
- `impostor`: different folder/subject label.

Distance metrics:

- `mesh_median_pct_bbox`: median corresponding-vertex distance after similarity
  Procrustes alignment, normalized by face bounding-box diagonal.
- `mesh_p90_pct_bbox`: p90 version of the same mesh distance.
- `landmark_descriptor_distance`: distance between normalized facial landmark
  shape descriptors.

Pass criterion:

- strict pass if `genuine_p90 < impostor_p10`.

## Full Crop Set Result

Full crop set: 9 crops total, 4 in `1_1`, 5 in `2_1`.

| Method | Metric | Genuine median | Genuine p90 | Impostor p10 | AUC | Pass |
|---|---|---:|---:|---:|---:|---|
| 3DDFA_V2 | mesh median | 0.736 | 1.089 | 0.424 | 0.419 | no |
| 3DDFA_V2 | landmark descriptor | 0.106 | 0.173 | 0.052 | 0.409 | no |
| MediaPipe | mesh median | 1.352 | 2.228 | 0.837 | 0.512 | no |
| MediaPipe | landmark descriptor | 0.191 | 0.371 | 0.094 | 0.428 | no |

Interpretation: genuine and impostor distributions overlap strongly. The current
mesh representations are not identity-separable on the full crop set.

## Quality-Filtered Set Result

Quality set: 4 crops total, 2 in `1_1`, 2 in `2_1`.

Included:

- `1_1_photo_2026-04-09_19-36-57_facecrop.jpg`
- `1_1_photo_2026-04-09_19-38-04_facecrop.jpg`
- `2_1_photo_2_2026-04-09_20-38-32_facecrop.jpg`
- `2_1_photo_3_2026-04-09_20-38-32_facecrop.jpg`

| Method | Metric | Genuine median | Genuine p90 | Impostor p10 | AUC | Pass |
|---|---|---:|---:|---:|---:|---|
| 3DDFA_V2 | mesh median | 0.716 | 0.917 | 0.387 | 0.375 | no |
| 3DDFA_V2 | landmark descriptor | 0.101 | 0.128 | 0.055 | 0.250 | no |
| MediaPipe | mesh median | 0.991 | 1.262 | 0.821 | 0.500 | no |
| MediaPipe | landmark descriptor | 0.128 | 0.182 | 0.096 | 0.500 | no |

Interpretation: even after removing obvious bad crops, these two baselines do
not create a Face-ID-style geometry separation on this tiny dataset.

## What This Means

The failure is informative. It does not mean the subjects are visually
indistinguishable. It means the current cheap one-photo geometry baselines
collapse too much identity-specific detail:

- MediaPipe is mainly a landmark/rough-face baseline.
- 3DDFA_V2 uses a low-dimensional BFM prior and tends to produce generic dense
  face shapes.
- Pose, crop, expression, occlusion, and model prior variation are comparable to
  or larger than between-subject differences.

## Next Decision

Do not use MediaPipe or 3DDFA_V2 as identity-preserving avatar baselines.

For identity-sensitive evaluation, the next useful methods are:

1. MICA/DECA/EMOCA after FLAME/model assets are available;
2. LAM or MeshLAM on a CUDA-capable machine;
3. a real 3D face scan / phone depth scan / multi-view reconstruction as a
   stronger photo-side reference.

The subject-aware metric itself is ready: rerun it once stronger avatar meshes
are available.
