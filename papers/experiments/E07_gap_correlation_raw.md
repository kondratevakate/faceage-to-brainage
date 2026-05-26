# E07 — Raw gap correlation (face vs brain) on IXI

- **Author**: KK
- **Date**: 2026-04-16
- **Status**: data-saved, but **finding was overturned in [E11](E11_gap_confound_check.md)**

## Question
On the same scan, does the face-age gap correlate with the brain-age
gap? If yes, the two modalities share an aging signal.

## Method
Intersect E02 calibrated face-age outputs with E05 SynthBA outputs.
Compute `face_age_gap = pred_face − chron`,
`brain_age_gap = pred_brain − chron`. Pearson and Spearman correlations.

## Result (raw, headline number that went into early paper drafts)
| metric | value |
|---|---|
| N | **93** |
| Pearson r | **+0.308** (p = 2.7 × 10⁻³) |
| Spearman ρ | **+0.318** (p = 1.9 × 10⁻³) |

Output: [`papers/tables/ixi_gap_correlation.csv`](../tables/ixi_gap_correlation.csv).

## ⚠ This number is misleading on its own
[E11](E11_gap_confound_check.md) shows the entire correlation is
explained by shared age-bias: partial r controlling for chronological
age = **−0.015**. The paper has been rewritten. Keep the raw number
here for completeness only.

## What's left undone
- N = 93 is severely under-powered (need ≈194 for r=0.2 at 80% power).
- Site-stratified analysis impossible (IOP missing, see [E13](E13_data_audit.md)).
- Bias-corrected gaps were not used at the time.
