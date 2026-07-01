# Photo + MRI Avatar Pilot

Purpose: build a private local experiment that compares a photo-derived head
avatar with an MRI-derived outer-head surface from the same person.

This pipeline is intentionally not part of the public gallery. Raw MRI, face
photos, and face-reconstructable meshes must stay in `rendering/output/` or
another ignored/private folder.

## Inputs

Use the same non-defaced MRI sessions from the Kate n=1 series:

| Session | Preferred role | Note |
|---|---|---|
| 2018 `3_fspgr_bravo_10mm_ax.nii.gz` | primary MRI surface | Best anatomical T1-like baseline. |
| 2024 `401_t1w_ffe.nii.gz` / `601_t1w_ffe.nii.gz` | candidate modern surface | More relevant if visual QC beats 2024 `901_3di_mc_hr`. |
| 2022 `4_t1_se_sag.nii.gz` | low-confidence comparison | Thick slices; useful for failure-mode comparison, not avatar quality. |

Photo requirements for the first useful pass:

1. Frontal neutral face, no smile, no glasses if possible.
2. Left and right 45-90 degree profiles.
3. Same-day or age-near photo if the MRI session year matters.
4. Optional but better: a 10-20 second phone video turning head left/right.

## Stage 1: MRI outer-head mesh

Run from the repo root:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\prepare_mri_head_mesh.py `
  --input "D:\path\to\nondefaced_t1.nii.gz" `
  --output-dir rendering\output\photo_mri_avatar_2026\mri_surfaces `
  --subject kate `
  --session 2018 `
  --preview
```

Outputs:

- `*_outer_head.ply`: private face-reconstructable mesh.
- `*_qc.png`: private quick QC image with mask slices and vertex projections.
- `*_metadata.json`: parameters and source path for reproducibility.

## Stage 2: photo-derived avatar

Fast local baseline:

```powershell
.\.venv\Scripts\python.exe -m pip install mediapipe
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\prepare_photo_facemesh.py `
  --input "D:\path\to\face_photo.jpg" `
  --output-dir "D:\path\to\faceage_brainage\avatar_2026\photo_avatar" `
  --model "D:\path\to\face_landmarker.task" `
  --subject kate `
  --session frontal_2026
```

This produces a MediaPipe 468-landmark face mesh with sampled vertex colors.
It is useful as a fast photo-derived shape/landmark baseline, not as metric 3D
ground truth.

Heavier candidate paths:

| Baseline | Input | Output | Why |
|---|---|---|---|
| 3DDFA_V2 | 1 frontal photo | Dense BFM face mesh | Runnable CPU baseline without FLAME; useful before licensed FLAME-family methods. |
| MICA/DECA/EMOCA | 1 frontal photo | FLAME-like mesh | Fast identity/shape baseline. |
| MeshLAM | 1 photo | animatable textured mesh | Good match to mesh + texture comparison. |
| LAM/SEGA/OMEGA-style | 1 photo | Gaussian avatar | Best perceptual realism, weaker geometry guarantees. |
| phone video / photogrammetry | short video | stronger photo-side geometry | Best practical ground truth if available. |

3DDFA_V2 runner:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\run_3ddfa_v2.py `
  --repo "D:\projects\02_academia\_external\avatars\3DDFA_V2" `
  --input-dir "D:\path\to\avatar_2026\photo_manifest\selected" `
  --pattern "*.jpg" `
  --output-dir "D:\path\to\avatar_2026\photo_avatar_3ddfa_v2" `
  --subject kate `
  --session selected_2026_04_09
```

The runner bypasses the upstream renderer and writes PLY directly from TDDFA.
If the optional FaceBoxes Cython NMS extension is missing, it installs a
pure-Python NMS fallback at runtime.

Face-crop preprocessing from 3DDFA detector boxes:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\crop_faces_from_3ddfa.py `
  --metadata-dir "D:\path\to\avatar_2026\photo_avatar_batch_3ddfa_v2" `
  --output-dir "D:\path\to\avatar_2026\photo_crops_3ddfa_1024" `
  --crop-size 1024 `
  --padding-scale 2.2
```

For cropped face folders, pass a restrictive pattern to avoid processing contact
sheets or other QC images:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\run_3ddfa_v2.py `
  --repo "D:\projects\02_academia\_external\avatars\3DDFA_V2" `
  --input-dir "D:\path\to\avatar_2026\photo_crops_3ddfa_1024" `
  --pattern "*_facecrop.jpg" `
  --output-dir "D:\path\to\avatar_2026\photo_avatar_crops_3ddfa_v2" `
  --subject kate `
  --session crops_3ddfa_1024
```

Batch summary:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\summarize_photo_baseline_batch.py `
  --photo-dir "1_1=D:\path\to\1_1\photo" `
  --photo-dir "2_1=D:\path\to\2_1\photos" `
  --photo-pattern "*.jpg" `
  --mediapipe-dir "D:\path\to\avatar_2026\photo_avatar_batch_mediapipe" `
  --3ddfa-dir "D:\path\to\avatar_2026\photo_avatar_batch_3ddfa_v2" `
  --mri-csv "mediapipe=D:\path\to\mediapipe_mri_summary.csv" `
  --mri-csv "3ddfa_v2=D:\path\to\3ddfa_mri_summary.csv" `
  --output-dir "D:\path\to\avatar_2026\stability\batch_2026_04_09"
```

Landmark-constrained MRI alignment:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\landmark_constrained_mri_alignment.py `
  --mri-mesh "D:\path\to\avatar_2026\mri_surfaces\kate_2018_outer_head.ply" `
  --photo-mesh-dir "D:\path\to\avatar_2026\photo_avatar_crops_mediapipe" `
  --method mediapipe `
  --output-dir "D:\path\to\avatar_2026\landmark_alignment\crops_mediapipe"
```

For 3DDFA_V2, pass the BFM file used by the upstream repo:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\landmark_constrained_mri_alignment.py `
  --mri-mesh "D:\path\to\avatar_2026\mri_surfaces\kate_2018_outer_head.ply" `
  --photo-mesh-dir "D:\path\to\avatar_2026\photo_avatar_crops_3ddfa_v2" `
  --method 3ddfa_v2 `
  --bfm-pkl "D:\projects\02_academia\_external\avatars\3DDFA_V2\configs\bfm_noneck_v3.pkl" `
  --output-dir "D:\path\to\avatar_2026\landmark_alignment\crops_3ddfa_v2"
```

Subject-aware consistency/separation:

```powershell
.\.venv\Scripts\python.exe pipeline\photo_mri_avatar\subject_consistency_metrics.py `
  --mediapipe-dir "D:\path\to\avatar_2026\photo_avatar_crops_mediapipe" `
  --3ddfa-dir "D:\path\to\avatar_2026\photo_avatar_crops_3ddfa_v2" `
  --bfm-pkl "D:\projects\02_academia\_external\avatars\3DDFA_V2\configs\bfm_noneck_v3.pkl" `
  --output-dir "D:\path\to\avatar_2026\subject_consistency\crops_3ddfa_1024"
```

The scientific comparison should keep these claims separate:

| Claim | Evidence |
|---|---|
| photo avatar looks realistic | human/QoE or image metrics |
| photo avatar preserves identity | face embedding similarity / landmarks |
| MRI mesh preserves geometry | point-to-surface or landmark distance against photo-side mesh |
| age signal is preserved | age estimate before/after texture/avatar prior |
