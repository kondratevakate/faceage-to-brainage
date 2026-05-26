# E01 — FaceAge on IXI multi-view renders

- **Author**: RK (Ramil Khafizov, `smileyenot983`)
- **Date**: 2026-04-15 / 2026-04-16
- **Status**: data-saved (`406 / 581` IXI subjects)
- **Commit**: `1521d6f` (add faceage)

## Question
Does the FaceAge model (Bontempi et al., *Lancet Digital Health* 2025),
trained on photographs, return any sensible age signal when applied to
faces rendered from MRI iso-surfaces?

## Method
For each IXI T1 NIfTI:

1. Marching cubes on intensity threshold `t = 30` → triangle mesh of the
   skin surface (PyVista).
2. Render 9 frontal views per subject with ±20° yaw/pitch and randomised
   lighting (`run_multiview.py`).
3. Per render: MTCNN face detect → crop → FaceAge ResNet-50 regression.
4. Average the 9 predictions per subject.

Code:
- [`src/faceage/faceage_mri/FaceAge/run_multiview.py`](../../src/faceage/faceage_mri/FaceAge/run_multiview.py)
- [`src/faceage/faceage_mri/FaceAge/run_multiview_multiproc.py`](../../src/faceage/faceage_mri/FaceAge/run_multiview_multiproc.py)

## Result
| metric | value |
|---|---|
| N | 406 (out of ~581 IXI) |
| MAE | **11.34 yr** |
| RMSE | 13.83 yr |
| bias (pred − chron) | **+10.04 yr** |
| Pearson r | 0.83 |

Output: [`papers/tables/ixi_faceage_multiview_raw.csv`](../tables/ixi_faceage_multiview_raw.csv).

## Interpretation
Detectable monotonic age signal (`r=0.83`), but a near-MAE positive bias
indicates that FaceAge consistently sees MRI renders as ~10 years older
than they are — this is the classical photo-vs-render domain gap.

## What's left undone
1. **IOP site nearly absent** (2 / ~70). The marching-cubes threshold is
   probably wrong for IOP intensities. See [E13](E13_data_audit.md).
2. **80 IXI subjects** have planned splits but were never rendered.
3. No subject-disjoint k-fold CV (Paplhám CVPR 2024 standard).
4. Photo-trained MTCNN occasionally fails on MRI renders; bypass-MTCNN
   was added but not formally compared.

## Pointers
- Output table: `papers/tables/ixi_faceage_multiview_raw.csv`
- Raw renders: `papers/figures/ixi_renders/` (gitignored)
- Caller: `scripts/batch_face_age.py`

## Next
[E02](E02_faceage_ixi_calibration.md) — linear calibration on top of
this output.
