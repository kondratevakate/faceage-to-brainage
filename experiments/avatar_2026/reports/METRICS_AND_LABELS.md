# Avatar Metrics and Subject Labels

## What the current 2.5 mm means

The current `~2.5 mm` value is a source-to-MRI anterior-cap nearest-neighbor
surface median after a landmark-seeded similarity transform. It is not a
validated anatomical accuracy claim.

It means:

- the dense 3DDFA face surface can be placed close to the MRI anterior surface;
- the central face oval is geometrically compatible at the surface-distance
  level;
- it does not prove that nose, chin, cheeks, eyelids, or soft-tissue ptosis are
  reconstructed correctly.

Why not final:

- MRI is supine; photos are upright or tilted.
- MRI surface is rough outer-head segmentation, not clinical skin surface.
- Current MRI landmarks are automatic proxy landmarks, not manual anatomical
  landmarks.
- Similarity alignment can hide metric scale and shape errors.

## Working Accuracy Tiers

| Tier | Surface/landmark error | Meaning |
|---|---:|---|
| Clinical scanner-level | <1 mm | Good 3D face scanner / anthropometry territory. |
| Strong avatar geometry | 1-2 mm | Good target for controlled scan-to-avatar comparison. |
| Usable one-photo face shape | 2-4 mm | Plausible geometry, not enough for fine soft-tissue claims. |
| Visual avatar only | 4-8 mm | May look plausible, weak anatomical geometry. |
| Not acceptable for geometry | >8 mm | Use only as visual or detector/debug output. |

For tissue ptosis, use stricter regional metrics:

- eyelid/periorbital: target <1-2 mm;
- cheek/midface sag: target <2 mm;
- jawline/chin/submental region: target <2-3 mm;
- full-head/hair/ears from one photo: do not treat as metric unless validated
  against a scan.

## Supine vs Upright Face

MRI and CT are usually acquired in a horizontal/supine position. Face photos and
3D photography are usually upright. Gravity changes soft tissue differently
across face regions, so MRI-to-photo comparisons should not assume identical
surface geometry.

Protocol implication:

- compare relatively stable landmarks first: nose bridge/tip, glabella/nasion,
  central chin;
- treat cheeks, eyelids, jawline, and submental tissue as posture-sensitive;
- report posture as a covariate: `mri_supine_vs_photo_upright`.

## Avatar Consistency Metric

Consistency should be computed only within the same subject. Do not mix photos
of different people.

Recommended per-subject metrics:

1. Detection success:
   - face found;
   - usable crop;
   - no major occlusion.

2. Shape consistency:
   - align avatar meshes from different photos by semantic landmarks;
   - report pairwise median/p90 surface distance;
   - report coefficient of variation for stable facial distances.

3. Landmark consistency:
   - nose-chin, cheek-cheek, brow-chin, nose-cheek distances after scale policy;
   - leave-one-photo-out mean residual.

4. MRI consistency:
   - landmark RMSE to manually annotated MRI landmarks;
   - median/p90 distance on matched facial regions;
   - separate central face from posture-sensitive soft tissue.

5. Perceptual consistency:
   - human review or blinded pairwise preference;
   - optional face-embedding similarity only if explicitly allowed and handled
     as identity-sensitive data.

## Required Subject Labels

The file `subject_labels_template.csv` has one row per crop. Fill:

- `subject_id`: e.g. `kate`, `sister`, `unknown`;
- `use_for_avatar`: `yes`, `no`, or `qc_only`;
- `notes`: occlusion, closed eyes, profile, hand, headphones.

Once labels are filled, recompute all consistency metrics grouped by
`subject_id`.

## Face-ID-Style Constraint

Use this only as a supervised separation test with known folder labels, not as
automatic identity discovery.

Goal:

- same-subject avatars should be close to each other;
- different-subject avatars should be farther apart;
- the separation margin should remain after pose/crop/expression variation.

Recommended formulation:

1. For every avatar output, compute a geometry descriptor:
   - normalized landmark distances;
   - local shape distances around nose, brow, chin, cheeks;
   - optional mesh PCA coefficients if topology is shared.

2. Compute two distributions:
   - genuine pairs: same `subject_id`;
   - impostor pairs: different `subject_id`.

3. Report:
   - Equal Error Rate-like threshold on avatar descriptors;
   - false match rate at a fixed false non-match tolerance;
   - separation margin: `median(impostor_distance) - median(genuine_distance)`;
   - ROC-AUC for subject separation.

4. Use as a constraint:
   - accept a method only if `genuine_p90 < impostor_p10`;
   - otherwise the avatar representation is not identity-separable enough.

Important: this is not a millimeter-only threshold. Identity separation depends
on which regions are measured. Siblings can be geometrically close, so the
threshold should be estimated from your labeled folders rather than imported as
a universal number.

## 2026-06-26 Three-Folder Baseline

Local photo folders currently visible:

- `1_1/photo`: 4 JPG files.
- `2_1/photos`: 5 JPG files.
- `3_1/photo`: 5 JPG files.
- `1_2`: empty locally.
- `1_3/photos`: empty locally; `1_3/nii` contains NIfTI files.

The three-folder crop baseline uses 14 photos total. Outputs:

- quicklook: `quicklook_3subjects_crops_mediapipe_3ddfa.jpg`;
- crops: `photo_crops_3subjects_3ddfa_1024`;
- MediaPipe meshes: `photo_avatar_crops_3subjects_mediapipe`;
- 3DDFA meshes: `photo_avatar_crops_3subjects_3ddfa_v2`;
- consistency: `subject_consistency/crops_3subjects_3ddfa_1024`.

Subject consistency was computed using folder/file prefixes as known labels.
No face identity inference was performed.

Pair counts per method:

- genuine pairs: 26;
- impostor pairs: 65;
- total pairs across MediaPipe and 3DDFA: 182.

Strict separation criterion: `genuine_p90 < impostor_p10`.

Current result: no metric passes strict separation. This means the current
MediaPipe/3DDFA one-photo geometry is useful as a detector/rough shape baseline,
but should not be described as Face ID-grade or identity-separable avatar
geometry.

Representative summary:

| Method | Metric | Genuine median | Genuine p90 | Impostor p10 | AUC | EER | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| 3DDFA_V2 | mesh median % bbox | 0.682 | 1.066 | 0.367 | 0.564 | 0.462 | no |
| 3DDFA_V2 | mesh p90 % bbox | 1.167 | 1.883 | 0.665 | 0.543 | 0.462 | no |
| 3DDFA_V2 | landmark descriptor | 0.101 | 0.161 | 0.047 | 0.507 | 0.538 | no |
| MediaPipe | mesh median % bbox | 1.246 | 1.850 | 0.921 | 0.682 | 0.427 | no |
| MediaPipe | mesh p90 % bbox | 3.014 | 3.819 | 2.033 | 0.656 | 0.454 | no |
| MediaPipe | landmark descriptor | 0.144 | 0.331 | 0.098 | 0.541 | 0.496 | no |

## FaceAge/Twin Literature Context

The FaceAge biomarker story is stored separately in
`TWIN_FACEAGE_LITERATURE_CONTEXT.md`.

Operational rule for this project:

- avatar geometry accuracy, identity consistency, and biological-age validity
  are separate claims;
- twin literature supports the premise that perceived facial age can be
  biologically meaningful;
- current FaceAge/FAHR-Face models are not yet twin-validated;
- the strongest future-work hook is a twin-controlled validation of AI facial
  age against perceived age, lifestyle discordance, methylation age, telomeres,
  and outcomes.
