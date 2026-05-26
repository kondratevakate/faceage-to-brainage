# E02 — Linear calibration of FaceAge on IXI

- **Author**: KK (Kate Kondrateva)
- **Date**: 2026-04-15
- **Status**: data-saved (calibrated on 20, applied to 80)
- **Builds on**: [E01](E01_faceage_ixi_multiview.md)

## Question
The raw FaceAge output from E01 has a `+10` yr positive bias. Does a
single linear calibration on a small held-out subset substantially
reduce MAE without overfitting?

## Method
1. Take E01 multi-view averaged predictions.
2. Subject-disjoint split: 20 calibration / 80 test.
3. Fit
   `y_chron = a · y_face_pred + b`
   on the 20 calibration subjects (least squares).
4. Apply the fitted line to the 80 test subjects.
5. Report MAE on the 80 held-out subjects.

## Result
| metric | value |
|---|---|
| Calibration formula | `y_chron = 1.010 · y_face_pred − 17.049` |
| Test N | 80 |
| MAE on test | **10.24 yr** |
| Pearson r | 0.83 |
| bias on test | **−2.12 yr** |

Output: [`papers/tables/ixi_faceage_splits.csv`](../tables/ixi_faceage_splits.csv).

## Interpretation
- Linear calibration removes ~80% of the bias (+10.04 → −2.12).
- MAE only drops marginally (11.34 → 10.24): the underlying scatter is
  not bias, it's per-subject noise that calibration cannot fix.

## What's left undone
1. **Calibration set N = 20 is tiny.** A 5-fold CV with proper subject-
   disjoint folds (Paplhám 2024 standard) is the minimum acceptable
   protocol now.
2. The calibration is post-hoc on the same dataset; no test on a
   genuinely external cohort (CamCAN, OpenBHB, etc.).
3. Non-linear (polynomial / GAM) calibration was not tried.

## Pointers
- Output: `papers/tables/ixi_faceage_splits.csv` (with `split` column)
- Caller: `scripts/batch_face_age.py` + post-hoc calibration in notebook

## Next
[E03](E03_photorealistic_renders.md) — independently tested whether
photorealistic stable-diffusion augmentation closes the domain gap (it
made things worse).
