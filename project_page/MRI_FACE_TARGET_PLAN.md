# MRI Face Target Plan

The current blocker is MRI face segmentation, not the photo-avatar baseline.
The current automatic outer-head surface is not a reliable facial skin target
for avatar accuracy claims.

## Failure Mode

The existing MRI surface is an outer-head extraction. It includes broad scalp
and head regions and does not yet provide a clean facial skin mask. Any
avatar-to-MRI overlay or Hausdorff-style metric computed on this target can look
plausible or terrible for the wrong reason.

## Required Target

The MRI target should be a face-only surface mask with explicit region labels:

- central face;
- nose bridge and nose tip;
- brow/glabella;
- cheeks;
- chin and jawline;
- optional periorbital region, treated separately because MRI/photos differ in
  posture and eyelid state.

Hair, ears, shoulders, scanner artifacts, and non-facial scalp should be
excluded from the primary face target.

## QC Gate

Before reporting any avatar-to-MRI distance, the MRI target must pass:

1. visual slice QC on axial, sagittal, and coronal views;
2. surface render QC from front, side, and three-quarter views;
3. manual or semi-manual landmark review for nose, brow, cheeks, chin;
4. region mask sanity checks: central face vs posture-sensitive regions;
5. no public Hausdorff, ASSD, Chamfer, or overlay claims before the above pass.

## Next Implementation Direction

Use the current photo-avatar outputs only as QC references. Build the MRI face
target first:

1. preferred: if a defaced copy exists in the same voxel grid, estimate the
   face target from `original - defaced` and then keep only the external skin
   shell;
2. fallback: start from the T1 head/skin candidate surface and crop a face ROI
   with anatomical landmarks;
3. remove non-face regions with manual or semi-manual landmark review;
4. add a small editable landmark file for each MRI;
5. export a face-only NIfTI mask, PLY surface, and region labels;
6. rerun metrics only after target QC passes.

Current utility:

```powershell
D:\projects\02_academia\_external\.venvs\avatar_cpu_py311\Scripts\python.exe scripts\photo_mri_avatar\segment_mri_face_target.py `
  --input path\to\original_T1.nii.gz `
  --defaced path\to\defaced_T1.nii.gz `
  --output-dir data\avatar_2026_work\mri_face_segmentation\case_id
```

If `--defaced` is omitted, the script falls back to the current coarse
landmark/anterior-surface ROI and should be treated as a draft target only.

Higher-quality avatar models such as LAM, MeshLAM, MICA, DECA, or EMOCA are
still useful, but they should not be judged against a bad MRI target.
