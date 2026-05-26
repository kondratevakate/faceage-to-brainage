# E12 — SIMON longitudinal slope test

- **Author**: CL (overnight)
- **Date**: 2026-04-24
- **Status**: done — paper rewritten

## Question
SIMON spans 29.6 → 46.4 yr in one subject (~17 yr of real aging). For
each method, does the regression
`predicted_age = slope · chron_age + intercept` show a slope > 0?

A slope of 1 means perfect tracking. A slope of 0 means the model is
"frozen" regardless of true age.

## Method
For each method, ordinary least-squares regression on the SIMON scans
(99 paired with chronological age). Code: same overnight script.

## Result
| method | N | slope | SE | p | R² | bias |
|---|---|---|---|---|---|---|
| **SynthBA (brain)** | 102 | **+0.148** | 0.036 | 6.9 × 10⁻⁵ | 0.147 | −16.2 |
| Face morphometrics | 82 | +0.038 | 0.185 | 0.84 | 0.001 | +5.1 |
| FaceAge multiview | 90 | −0.140 | 0.182 | 0.44 | 0.007 | +8.5 |

## Interpretation
- **SynthBA**, despite its visually flat predictions and 16 yr bias,
  recovers ~15 % of the true aging rate at high statistical
  significance. Brain parenchyma changes are detectable even when the
  model's baseline is wrong by 16 years.
- **Both face methods are statistically flat**: their visible scatter
  in Figure 1 of the paper is *scanner / session noise at fixed
  time-points*, not within-subject aging. The face branch as currently
  implemented carries no longitudinal aging signal in this subject.

This **reverses an earlier visual-inspection claim** in the paper that
"face morphometrics show a positive trend with chronological age". The
real test is `p = 0.84`.

## What's left undone
- Replicate on SRPBS Travelling Subjects (multiple subjects, multiple
  scanners) so the conclusion does not rest on a single individual.
- Repeat with SFCN, MIDIBrainAge, BrainIAC on SIMON to see whether
  better brain-age models track the subject's slope better than 15 %.
- Mixed-effects model with random scanner intercept to disentangle
  scanner effect from time effect.

## Paper consequence
Abstract, Results §SIMON, Figure caption updated in commit `cd9b74a`.

## Pointers
- Code: [`papers/midl2026/overnight_analysis.py`](../midl2026/overnight_analysis.py)
- Full log: [`papers/midl2026/RESULTS.tsv`](../midl2026/RESULTS.tsv)
- Findings: [`papers/midl2026/FINDINGS.md`](../midl2026/FINDINGS.md)
