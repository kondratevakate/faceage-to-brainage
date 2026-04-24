# Overnight Hypothesis — Karpathy-style autoresearch

**Date**: 2026-04-24 · **Branch**: `overnight-gap-confound-hypothesis`

## One hypothesis

**H₀ (what the paper currently claims)**: face-age gap and brain-age gap
measured from the same T1 MRI scan share a common biological aging signal,
reflected in Pearson `r = 0.31` (`p < 0.01`) on `N = 93` IXI subjects.

**H₁ (the confound claim we test)**: The `r = 0.31` correlation is
*predominantly* driven by **shared age-bias** in both predictors
(regression-to-the-mean against chronological age), not by shared biology.
Both FaceAge and SynthBA are known to over-predict for young subjects and
under-predict for old subjects. If true, both gaps will be correlated with
chronological age in the *same direction*, and the apparent inter-gap
correlation is partly mediated through that shared nuisance variable.

## Pre-registered prediction

We test the canonical correction (Smith et al. 2019, *Hum. Brain Mapp.*):
**partial Pearson correlation of the two gaps controlling for
chronological age**.

- If partial `r | age` ≥ 0.20 → H₁ is **rejected**. The shared-aging claim
  is robust to the canonical confound. Paper survives unchanged.
- If partial `r | age` ≤ 0.15 → H₁ is **accepted**. At least half of the
  raw correlation is age-bias. Paper needs substantive rewrite.
- If 0.15 < partial `r | age` < 0.20 → **ambiguous**. Report both. Expand
  with a linear-mixed-model sensitivity analysis.

## Why this matters

This is the load-bearing sentence of the MIDL paper's Discussion:
> "The two aging gaps computed from the same scan show a statistically
> significant positive correlation (r = 0.31, ρ = 0.32, p < 0.01),
> suggesting partial but incomplete overlap between facial and cerebral
> aging signals."

If H₁ is accepted, this sentence has to go. The experiment takes about a
minute on data already in the repo. High stake, low cost.

## Method

File: [`overnight_analysis.py`](overnight_analysis.py)

1. Load `papers/tables/ixi_gap_correlation.csv` (N = 93 paired subjects).
2. Compute raw Pearson and Spearman correlations between
   `face_age_gap` and `brain_age_gap` — reproduce the paper's number.
3. Regress each gap on `true_age` and subtract the linear fit (Smith et al.
   bias-correction). Recompute correlation on residuals.
4. Compute partial Pearson correlation of `face_age_gap` and
   `brain_age_gap` controlling for `true_age`.
5. Decision rule above.

## Secondary ratchet — SIMON longitudinal slope

While at it, answer a related scientific question for free:
**does any of the three methods track the subject's longitudinal aging
on SIMON?** (29 → 46 years, 99 scans)

For each method fit `predicted ~ a * chron_age + b`. Report:
- `slope`, `slope SE`, `p`
- `slope ≈ 1`: method tracks aging correctly (even if biased in mean)
- `slope ≈ 0`: method is frozen — the OOD failure case

## Tertiary ratchet — stratified IXI

If H₁ is accepted, is there any age band where the correlation survives
cleanly? Stratify IXI into three bins (20–45, 45–65, 65–86 y) and report
raw and partial correlations per bin.

## Log

Results will be appended to [`RESULTS.tsv`](RESULTS.tsv) as the script runs.
