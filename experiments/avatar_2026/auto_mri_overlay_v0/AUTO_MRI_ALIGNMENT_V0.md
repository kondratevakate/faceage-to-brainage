# Automatic MRI Alignment Visual QC v0

Updated: 2026-06-27

## Purpose

This folder starts the automatic MRI-to-photo-avatar visual overlay workflow.
It is meant for artifact review and method debugging, not final anatomical
accuracy claims.

Main contact sheet:

- `auto_mri_overlay_v0_contact_sheet.jpg`

Rows are known folder labels:

- `1_1`
- `2_1`
- `3_1`

Columns:

1. photo crop;
2. 3DDFA overlay;
3. MediaPipe overlay;
4. automatic 3DDFA-to-MRI alignment preview;
5. automatic MediaPipe-to-MRI alignment preview.

No face identity inference is performed. Folder labels are treated as known
supervised labels.

## Can MRI Alignment Be Automatic?

Yes. The current automatic proxy alignment is:

1. Build an outer-head/skin-like surface from MRI.
2. Estimate rough MRI facial landmarks from the outer surface:
   - nose tip;
   - chin;
   - brow/glabella region;
   - left/right cheek proxy points.
3. Extract matching landmarks from photo-derived avatar meshes.
4. Fit a robust similarity transform from avatar landmarks to MRI proxy
   landmarks.
5. Measure source-to-MRI anterior-surface distances and render QC previews.

This is useful for:

- detecting gross orientation errors;
- seeing scanner/segmentation artifacts;
- comparing avatar methods under the same MRI proxy;
- building a first automated metric before manual landmark cleanup.

It is not enough for:

- sub-millimeter or clinical accuracy claims;
- eyelid, cheek, jawline, or ptosis claims;
- proving identity-grade reconstruction;
- replacing a 3D face scan.

## MRI Face Without Scanner Artifacts

The next technical problem is not only alignment. It is building a clean
MRI-derived face/head surface:

1. Choose the best anatomical sequence:
   - prefer high-resolution T1/T2 with full head coverage;
   - reject low-slice or partial-head scans for facial surface claims.
2. Separate real skin surface from scanner artifacts:
   - remove table/coil/background components;
   - keep the largest plausible head component;
   - remove floating islands and edge streaks.
3. Smooth without erasing facial landmarks:
   - mild surface smoothing;
   - preserve nose/chin/brow/cheek geometry;
   - avoid over-smoothing ptosis-sensitive regions.
4. Generate QC views:
   - sagittal/coronal/axial surface projections;
   - component labels;
   - anterior face mask;
   - artifact mask.
5. Only then compute automatic avatar-to-MRI distances.

## Current Interpretation

The current visual sheet shows that automatic alignment is feasible as a QC
layer. It should now be used to compare avatar methods visually and to identify
where the MRI mask or proxy landmarks fail.

The next useful improvement is a dedicated `mri_face_surface_v1` step:

- clean outer-head segmentation;
- anterior face mask;
- artifact removal;
- automatic proxy landmarks;
- optional manual landmark override file.
