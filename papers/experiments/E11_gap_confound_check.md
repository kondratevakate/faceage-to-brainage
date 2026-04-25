# E11 — IXI gap correlation: age-bias confound check

- **Author**: CL (Claude, autoresearch overnight session)
- **Date**: 2026-04-24
- **Status**: done — **overturned [E07](E07_gap_correlation_raw.md), paper rewritten**
- **Branch**: `overnight-gap-confound-hypothesis`

## Hypothesis (pre-registered)
**H₁**: the IXI face-gap × brain-gap correlation is predominantly
shared age-bias, not shared biology.

**Decision rule**:
- accept H₁ if partial Pearson r controlling for chronological age ≤ 0.15
- reject if partial r ≥ 0.20
- ambiguous in between

## Method
Standard Smith et al. 2019 brain-age bias check:
1. Reproduce raw r.
2. Regress each gap on `true_age` and subtract the linear fit.
3. Compute Pearson r on the residuals (= partial correlation).
4. Permutation null over 10 000 shuffles.

Code: [`papers/midl2026/overnight_analysis.py`](../midl2026/overnight_analysis.py)

## Result
| quantity | value |
|---|---|
| Raw Pearson r | +0.308 (p = 2.7 × 10⁻³) |
| face_gap vs age | r = **−0.641** (p = 5 × 10⁻¹²) |
| brain_gap vs age | r = **−0.496** (p = 4 × 10⁻⁷) |
| **Partial r \| age** | **−0.015** (p = 0.89) |
| Bias-corrected r | −0.015 (p = 0.89) |

**Verdict: ACCEPT H₁.**

## Interpretation
Both predictors are over-estimating young subjects and under-estimating
old subjects (the canonical brain-age bias). This shared nuisance
structure produces a spurious positive correlation between the two
gaps. Once the bias is regressed out, **no residual association remains.**

This reproduces, from a single non-defaced T1, the negative result that
Cole et al. 2020 *NeuroImage* obtained in UK Biobank with subjective
facial age ratings. Same conclusion via a different (and cleaner)
method.

## What's left undone
- Repeat with non-linear (polynomial / GAM) bias correction to confirm
  robustness — not yet done because the data audit ([E13](E13_data_audit.md))
  showed N = 93 is too small to support any further sub-analysis until
  the cohort is restored.
- Replicate on a SIMON-disjoint cohort (CamCAN, MR-ART, OpenBHB).

## Pointers
- Full numbers: [`papers/midl2026/RESULTS.tsv`](../midl2026/RESULTS.tsv)
- Findings doc: [`papers/midl2026/FINDINGS.md`](../midl2026/FINDINGS.md)
- Pre-registration: [`papers/midl2026/HYPOTHESIS.md`](../midl2026/HYPOTHESIS.md)

## Paper consequence
Abstract, Results §gap correlation, Discussion bullet rewritten in
commit `cd9b74a` on this branch.
