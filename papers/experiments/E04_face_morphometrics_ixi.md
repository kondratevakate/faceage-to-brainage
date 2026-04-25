# E04 — Face morphometrics (BioFace3D-20 + GPA + EDMA + Ridge) on IXI

- **Author**: GB (Gleb Bobrovskikh, `BobrG`)
- **Date**: 2026-04-16
- **Status**: data-saved
- **Commit**: `c3773f0` (face age via face biomarkers)

## Question
Forget photos. The MRI face surface is already a 3D mesh. If we extract
geometric landmarks from it directly and fit a regressor on
shape-features only (no skin colour, no texture), how well can we
predict chronological age?

## Method
1. Extract **BioFace3D-20** — 20 anatomical landmarks on the face
   surface — using the BioFace3D MATLAB GUI (`bioface3d/`).
2. Apply **Generalized Procrustes Analysis (GPA)** to remove
   translation, rotation, scale across subjects.
3. Compute **EDMA (Euclidean Distance Matrix Analysis)** features —
   pairwise inter-landmark distances.
4. Train **Ridge regression** (`alpha` chosen by CV) on the EDMA
   feature vector to predict chronological age.
5. Evaluate with subject-disjoint k-fold CV.

Code:
- [`src/face_age_morphometrics/scripts/extract_features.py`](../../src/face_age_morphometrics/scripts/extract_features.py)
- [`src/face_age_morphometrics/scripts/train_age_regressor.py`](../../src/face_age_morphometrics/scripts/train_age_regressor.py)
- [`src/face_age_morphometrics/scripts/benchmark_regressors.py`](../../src/face_age_morphometrics/scripts/benchmark_regressors.py)

## Result
| metric | value |
|---|---|
| N | 72 (subset of IXI with successful landmark extraction) |
| MAE | **7.81 yr** |
| RMSE | — |
| Pearson r | 0.78 |
| bias | −1.83 yr |

Output table on SIMON: [`papers/tables/simon_faceage_morphometrics.csv`](../tables/simon_faceage_morphometrics.csv).

## Interpretation
The geometric branch outperforms FaceAge on raw MRI renders (7.81 vs
11.34 MAE) and is much less biased (−1.83 vs +10.04). This is
consistent with the project's working hypothesis: **MRI carries face
geometry but not surface appearance, so a geometry-only model has the
right input distribution while FaceAge does not.**

## What's left undone
1. **Subject-disjoint CV is implicit** but the exact split protocol
   needs to be reported (Paplhám 2024).
2. **Training data leakage check**: was the regressor trained on a
   superset of the test subjects? If yes, the 7.81 yr MAE is biased
   optimistic. Needs explicit train/val/test split disclosure.
3. **Only 72 / 581** IXI subjects make it through landmark extraction.
   Why does landmarking fail on the rest? Diagnose.
4. **MATLAB dependency** for landmark extraction blocks reproducibility
   on a Linux/Colab pipeline.

## Pointers
- IXI output: not currently in `papers/tables/` — owner GB to export
- SIMON output: `papers/tables/simon_faceage_morphometrics.csv`

## Next
- [E12](E12_simon_slope.md) tested whether morphometrics tracks
  longitudinal aging on SIMON. Result: **no detectable slope**
  (`+0.038`/yr, `p = 0.84`). Geometry-only does well cross-sectionally
  but does not pick up within-subject aging in the SIMON window
  (29 → 46 yr).
