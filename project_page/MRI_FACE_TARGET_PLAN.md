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

1. start from the T1 head/skin candidate surface;
2. remove non-face regions with anatomical cropping anchored by manual or
   semi-manual landmarks;
3. add a small editable landmark file for each MRI;
4. export a face-only PLY plus region labels;
5. rerun metrics only after target QC passes.

Higher-quality avatar models such as LAM, MeshLAM, MICA, DECA, or EMOCA are
still useful, but they should not be judged against a bad MRI target.
