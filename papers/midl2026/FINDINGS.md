# Overnight Findings — 2026-04-24

## TL;DR

**The paper's main claim is statistically confounded.** The `r = 0.31`
correlation between face-age gap and brain-age gap on IXI — the single
result advertised as evidence of "shared facial-cerebral aging signal" —
completely disappears after controlling for chronological age.

The paper must be rewritten. Details, numbers, and decision below.

---

## Primary result — IXI gap correlation confound check

| Quantity | Value |
|---|---|
| N paired subjects | 93 |
| Age range | 20.1–74.0 y (mean 35.8 ± 11.9) |
| **Raw Pearson `r(face_gap, brain_gap)`** | **`+0.308`** (p = 2.7×10⁻³) |
| face_gap vs true_age | `r = −0.641` (slope = −0.464, p = 5×10⁻¹²) |
| brain_gap vs true_age | `r = −0.496` (slope = −0.283, p = 4×10⁻⁷) |
| **Bias-corrected `r`** (residuals after removing linear age dep.) | **`−0.015`** (p = 0.888) |
| **Partial `r(face_gap, brain_gap │ true_age)`** | **`−0.015`** |

**Verdict: ACCEPT H₁** (pre-registered: accept if partial r ≤ 0.15).

### Mechanism

Both gaps depend *strongly and in the same direction* on chronological
age: each predictor over-predicts young subjects and under-predicts old
subjects (the classical brain-age bias, Smith et al. 2019 *HBM*). When
two predictions share this nuisance structure, any two-by-two scatter
plot of gaps will automatically show a spurious positive correlation —
this is exactly what `r = 0.31` is. After the bias is regressed out,
the residual association with each other is **essentially zero**.

### Stratification (tertiary)

| Age band | N | raw r | partial r │ age |
|---|---|---|---|
| 20–45 y | 75 | +0.306 | +0.083 |
| 45–65 y | 15 | −0.349 | −0.398 |
| 65–86 y | 3 | skipped (N < 10) | — |

In the young-to-middle band (where 80% of subjects live), the confound
explains ~73% of the raw correlation. In older subjects N is too small
to conclude anything; expanding IXI/ADNI/OASIS cohorts is the only path.

---

## Secondary result — SIMON longitudinal slope

*Does any method actually track the subject's aging across the 29 → 46 y
window on SIMON (one subject, 36 scanners, 99 scans)?*

| Method | N | slope (yr/yr) | SE | p | R² | bias |
|---|---|---|---|---|---|---|
| **SynthBA (brain)** | 102 | **+0.148** | 0.036 | 6.9×10⁻⁵ | 0.147 | −16.23 |
| Face morphometrics | 82 | +0.038 | 0.185 | 0.84 | 0.001 | +5.14 |
| FaceAge multiview | 90 | −0.140 | 0.182 | 0.44 | 0.007 | +8.52 |

**SynthBA weakly but significantly tracks aging** (slope +0.148, p<10⁻⁴),
despite predicting a near-constant ~27 y regardless of the subject's
actual age (huge bias, tiny slope). The brain parenchyma changes enough
within 29→46 y that even a severely out-of-distribution model picks up
a fraction of it.

**Neither face method tracks aging at all.** Face morphometrics has no
slope, FaceAge has negative slope (noise). The visual "positive trend"
in Figure 1 of the draft is driven by scanner / session variance at
fixed time points, not by the subject's aging.

This flips the implicit hypothesis of the project: the MRI brain signal
*does* contain longitudinal aging information even in an OOD setting,
while the MRI face signal — at least as captured by the current render
pipeline — does not.

---

## What this means for the MIDL paper

Three load-bearing paragraphs become false or misleading:

1. **Abstract** — "the two aging gaps are positively correlated
   (`r = 0.31`, `p < 0.01`)" → must be reformulated as "the raw
   correlation is driven by shared age-bias; after partialing out
   chronological age, no residual association is detected".

2. **Results §Gap correlation** — same issue.

3. **Discussion bullet** — "the face-age and brain-age gaps from the
   same scan are modestly but significantly correlated" → false as
   stated. Must be replaced with "no residual correlation after age
   control" and framed as a cleaner null result.

The paper's scientific contribution **becomes stronger**, not weaker,
with this correction:

- The field is full of papers claiming shared aging signal on the basis
  of un-corrected `r` values. Reporting the negative result after
  proper bias control is a **methodologically honest** contribution.
- The SIMON OOD analysis is still intact and arguably the paper's most
  valuable finding (a model with SD 1.26 y can be 16 y off in absolute
  terms — this is the readiness gap the paper argues for).
- The "no residual shared-aging signal" result aligns with Cole et al.
  2020 *NeuroImage* (the UK Biobank null), which used perceived age
  rather than AI-derived face age. We reach the same conclusion from
  same-scan MRI-derived face age, which is a **novel confirmation via
  a new method**.

### Proposed rewrite (one paragraph)

> On 93 overlapping IXI subjects, the raw Pearson correlation between
> face-age gap and brain-age gap was `r = 0.308` (`p = 2.7×10⁻³`). Both
> gaps, however, were strongly associated with chronological age
> (face_gap: `r = −0.64`, brain_gap: `r = −0.50`) — the standard
> brain-age-bias pattern documented by Smith et al. 2019. After partialing
> out chronological age, the between-gap correlation was
> `r = −0.015` (`p = 0.89`): essentially zero. We conclude that the apparent
> shared-aging signal in the raw correlation is an artifact of common
> age-bias in the two predictors, not evidence of a shared biological
> aging trajectory. This reproduces, from a single non-defaced T1 scan,
> the negative result Cole et al. 2020 obtained with subjective facial
> age ratings in UK Biobank.

---

## What to do next (ratchet)

Karpathy-style: each is a single-hypothesis, single-metric, single-commit
follow-up.

1. **Fit a state-of-the-art brain-age model on IXI and rerun the partial
   correlation**. If the confound survives SFCN/BrainIAC too, publish
   the null cleanly. If it disappears with a better brain-age model,
   that would be a new finding about model-driven confound.

2. **Does the confound survive non-linear bias correction?** Current
   correction is linear. Try polynomial (order-2) and GAM. If the partial
   r stays near zero under any reasonable model of the bias, the null is
   robust.

3. **Same test on a longitudinal cohort** (OASIS-3 or ADNI4 subset with
   DUA-compliant use). If gap change within the same subject across time
   still correlates after age control, then longitudinal shared aging may
   exist even though cross-sectional does not.

4. **Face age from MRI without the FaceAge detour**. If FaceAge is
   essentially a biological-age model and we want chronological, fit a
   chronological-age regressor directly on the morphometric features of
   the MRI-derived face surface. Compare its gap against brain gap with
   the partial-correlation test.

---

## Data and reproducibility

- Analysis script: [`overnight_analysis.py`](overnight_analysis.py)
- Full log of experiments: [`RESULTS.tsv`](RESULTS.tsv)
- Bias-corrected gap values per subject: [`ixi_gap_corrected.csv`](ixi_gap_corrected.csv)
- Branch: `overnight-gap-confound-hypothesis`
- Seed: none needed; computation is deterministic

To reproduce:
```bash
git checkout overnight-gap-confound-hypothesis
cd papers/midl2026
PYTHONIOENCODING=utf-8 python overnight_analysis.py
```
