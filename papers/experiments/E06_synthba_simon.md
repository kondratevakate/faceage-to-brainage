# E06 — SynthBA on SIMON (scanner reproducibility)

- **Author**: KK
- **Date**: 2026-04-14
- **Status**: data-saved
- **Notebook**: [`notebooks/08_synthba_simon_colab.ipynb`](../../notebooks/08_synthba_simon_colab.ipynb)

## Question
SIMON is one healthy male scanned across 36 different scanners and 73
sessions while aging from 29.6 to 46.4 years. Two questions in one:
1. Is SynthBA reproducible across scanners (low SD across scans)?
2. Does it follow this single subject's longitudinal aging?

## Method
For each of the 99 SIMON T1 scans: SynthStrip → SynthBA. Tabulate
predictions per scan with chronological age and scanner ID.

## Result
| metric | value |
|---|---|
| N | 99 scans (1 subject, 36 scanners, 73 sessions) |
| Mean predicted age | **27.4 yr** |
| Prediction SD across scans | **1.21 yr** ✅ excellent |
| Chronological age range | 29.6 – 46.4 yr |
| MAE | **16.2 yr** ❌ |
| Bias (pred − chron) | **−16.2 yr** |

Output: [`papers/tables/simon_brainage_synthba.csv`](../tables/simon_brainage_synthba.csv).

## Interpretation (the headline finding of the paper)
The model is **highly reproducible across scanners** but **almost
totally wrong** for this subject. SD = 1.21 yr is a publishable-grade
test–retest number; bias = −16 yr is a complete OOD failure of
calibration. **Scanner reproducibility is not validity.**

## What's left undone
- No comparison against SFCN / MIDIBrainAge / BrainIAC on the same
  scans (notebooks 09, 10 were run but did not export CSVs — see [E09](E09_no_card.md), [E10](E10_no_card.md)).
- ICC(2,1) and Bland-Altman LoA not yet computed.
- No external SIMON-like cohort (SRPBS Travelling Subjects, MR-ART) for
  cross-validation of the reproducibility number.

## Next
- [E12](E12_simon_slope.md) — formal slope test: does any method track
  this subject's aging? (SynthBA does, weakly; face methods do not.)
