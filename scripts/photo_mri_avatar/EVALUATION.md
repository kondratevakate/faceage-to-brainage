# Evaluation Protocol: One-Photo Avatar vs MRI-Derived Face Surface

Purpose: rank single-photo avatar/reconstruction baselines by how well their
recovered face geometry agrees with the same person's MRI-derived head surface.

This is a staged protocol. Early metrics are weak but runnable. Stronger claims
require better ground truth, such as photogrammetry, multi-view reconstruction,
or manual craniofacial landmarks.

## Claims and Evidence

| Claim | Current evidence | Strength |
|---|---|---|
| The method detects and reconstructs a plausible face from one photo. | Landmark overlay, mesh preview, face bounding box. | Strong for detection, weak for metric 3D. |
| The method is stable across same-session photos. | Mesh-to-mesh similarity after normalization/alignment. | Moderate if photos differ in pose/lighting. |
| The method agrees with MRI-derived face/head geometry. | Similarity alignment to MRI surface plus point-to-surface distances. | Weak until MRI skin segmentation and photo-side metric scale improve. |
| The method preserves identity/age signal. | Not yet measured. Needs face embeddings and age-model runs. | Missing. |

## First-Pass Metrics

For each photo-derived mesh:

1. **Detection QC**
   - face detected: yes/no
   - landmarks/vertices count
   - overlay image for visual inspection

2. **MRI Alignment Distance**
   - normalize/alignment is estimated from source mesh to the anterior/frontal
     cap of the MRI surface;
   - for LAS-oriented MRI, force `front_axis=1` and `front_sign=+1` rather than
     selecting the lowest-distance cap automatically. Automatic cap selection can
     incorrectly match a face mesh to the superior/top head cap because both are
     oval surfaces;
   - evaluate source-to-MRI nearest-neighbor distances after similarity ICP;
   - report median, mean, p75, p90, max in millimeters.

3. **Caveats**
   - MediaPipe FaceMesh is not a full-head avatar and has arbitrary scale;
   - MRI threshold mesh is a rough outer-head surface, not tissue-aware skin;
   - without manually paired landmarks, ICP can find a plausible but not
     anatomically guaranteed alignment;
   - therefore this metric is for baseline triage, not publication-level proof.

## Baseline Queue

| Baseline | Runnable now | Role |
|---|---:|---|
| MediaPipe FaceMesh | yes | Fast lower-bound, landmarks/rough face surface. |
| 3DDFA_V2 | yes | Dense BFM face mesh baseline; runnable without FLAME. |
| MICA / DECA / EMOCA | blocked on FLAME/assets | Stronger FLAME-like mesh baseline. |
| MeshLAM | pending | One-photo textured mesh avatar, likely best mesh/render candidate. |
| LAM / Gaussian avatar | pending | Perceptual realism candidate; weaker for geometry without ground truth. |

## Current Pilot Result

Using the 2018 MRI outer-head mesh and the forced LAS anterior cap
(`front_axis=1`, `front_sign=+1`), MediaPipe and 3DDFA_V2 both land around a
median source-to-MRI distance of about 2 mm after unconstrained similarity ICP.
This should not be read as proof that the methods recover MRI-matched geometry.
The visual previews show that the anterior cap plus ICP can overfit oval face
surfaces without enforcing anatomical correspondences.

Current meaning of the metric:

- valid: the photo mesh is detected, can be aligned, and roughly occupies the
  expected anterior-head region;
- not valid yet: ranking fine-grained avatar quality or claiming one-photo
  reconstruction is anatomically accurate against MRI.

## Decision Rule

Use MediaPipe only to validate the pipeline. The first serious comparison should
rank at least one stronger face/avatar method against MediaPipe and 3DDFA_V2
with additional constraints. A Gaussian avatar should be evaluated separately as
a perceptual/rendering method, not treated as metric geometry unless it exposes a
mesh or dense correspondence.
