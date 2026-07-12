# Multimodal Health Feature Palette

Date: 2026-07-12

## Purpose

This note defines a research palette for linking MRI, foundation-model outputs,
behavioral traces, sensory data, retina/OCT, vascular markers, labs, and
questionnaires to population-level health patterns.

The goal is not to diagnose disease from structural MRI. The goal is to define
measurable feature families, expected confounders, and valid claim boundaries so
chronological age is not used as the only proxy for biological or functional
state.

## Scientific Claim Boundary

Valid claims:

- a feature is associated with a population-level outcome after explicit
  adjustment for age, sex, scanner/site, preprocessing, and measured
  confounders;
- a feature is useful as QC, domain-shift, or robustness evidence;
- sex modifies a feature-outcome relationship in a specified dataset;
- a model output is a candidate risk or phenotype marker only after labeled
  external validation.

Invalid claims:

- "male brain" or "female brain" is healthier;
- a sex classifier output diagnoses Parkinson's disease, Alzheimer's disease,
  vascular disease, stress, loneliness, or biological age;
- a Kate n=1 prediction validates a disease or aging claim;
- BrainFM/NeuroFM embeddings validate segmentation or morphometry.

## Sex As Moderator, Not Health Score

Sex should be modeled as a moderator, confounder, and QC axis. It should not be
reduced to a health score.

Operational uses:

- evaluate whether a model uses sex or total intracranial volume as a shortcut;
- stratify age/health associations by sex;
- test `sex x age`, `sex x feature`, and `sex x feature x age` interactions;
- track whether predicted sex flips under preprocessing, resampling, rotation,
  or skull-stripping perturbations;
- report brain-age residuals and uncertainty separately by sex where labels are
  available.

Male-associated hypothesis space:

- earlier vascular or metabolic risk burden;
- higher cardiovascular mortality risk in many populations;
- potentially faster or earlier structural atrophy in some MRI cohorts;
- higher burden of Parkinson's disease or Lewy body dementia in several
  epidemiological summaries.

Female-associated hypothesis space:

- longer average lifespan;
- some PET work reports a younger-appearing metabolic brain age in women;
- higher lifetime Alzheimer's disease burden, partly but not only due to
  longevity;
- menopause/estrogen transition as a potential vascular and neurodegenerative
  vulnerability window;
- possible stronger coupling between white-matter hyperintensity burden and
  cognitive or dementia outcomes in late life in some cohorts.

These are hypothesis families for cohort analysis. They are not individual
diagnostic claims.

## Measurable Feature Families

The expanded systematic possibility map is tracked in:

`docs/kate_n1_2026/multimodal_feature_possibility_map.md`

High-priority families:

- structural MRI: TIV, brain volume, GM/WM/CSF, cortical thickness, surface
  area, hippocampal volume, ventricular volume, regional atrophy;
- vascular MRI: WMH, lacunes, enlarged perivascular spaces, microbleeds,
  microinfarct-sensitive sequences, ASL perfusion, MRA/vessel-wall MRI;
- DWI/qMRI: FA, MD, RD, AD, fixel density/cross-section, free-water, myelin or
  water-content proxies;
- sensory reserve: hearing, vision, OCT/fundus, visual and auditory cortical
  markers;
- metabolic/inflammatory: LDL, ApoB, HbA1c, blood pressure, CRP, CBC-derived
  inflammatory indices, BMI/waist;
- mobility/environment: life-space radius, location entropy, home-stay
  fraction, outdoor time, green/blue-space exposure, walkability, pollution;
- psychosocial: loneliness scale, social support, depression/anxiety, sleep,
  stress/allostatic load;
- model robustness: test-retest variance, site variance, preprocessing delta,
  perturbation delta, sex-class flip, brain-age residual stability.

## Analysis Pattern

For population datasets, prefer models shaped like:

```text
outcome ~ age_spline + sex + feature + scanner_site + TIV + preprocessing
        + sex:feature + sex:age_spline + feature:age_spline
```

For repeated scans or travelling subjects:

```text
outcome ~ fixed_effects + (1 | subject) + (1 | site/scanner)
```

Chronological age should remain a covariate, calibration axis, or baseline
comparison. It should not be treated as the sole target standing in for health.

## N=1 Use

For Kate n=1, the palette is limited to:

- application/QC reporting;
- robustness and protocol-sensitivity checks;
- generating hypotheses for cohort analysis;
- identifying which additional data modalities would be needed.

It cannot validate disease risk, vascular age, immune age, loneliness, dementia
risk, or biological age without external labeled population evidence.

## Sources To Track

- Brain sex prediction and TIV/age confounding review:
  https://www.explorationpub.com/Journals/ent/Article/1004141
- Age-dependent functional sex differences:
  https://academic.oup.com/cercor/article/31/6/3021/6104776
- Female metabolic brain-age PET result:
  https://www.pnas.org/doi/10.1073/pnas.1815917116
- Sex differences in dementia and risk-factor literature:
  https://link.springer.com/article/10.1186/s13195-024-01598-2
- Midlife cardiovascular risk factors and dementia in UK Biobank:
  https://www.ukbiobank.ac.uk/publications/sex-differences-in-the-association-between-major-cardiovascular-risk-factors-in-midlife-and-dementia-a-cohort-study-using-data-from-the-uk-biobank/
- Lancet Commission dementia risk framework:
  https://www.thelancet.com/commissions/dementia-prevention-intervention-care
