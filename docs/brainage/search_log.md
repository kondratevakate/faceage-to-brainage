# BrainAge search log

## Scope and date

Search performed 2026-07-18. This is a reproducible rapid search for experiment
design, not a complete systematic review.

## Sources

- PubMed for peer-reviewed neuroimaging studies and reviews.
- DOI landing pages for bibliographic verification.
- Official GitHub, Hugging Face, project, and documentation pages for model
  provenance and input contracts.
- Official SIMON, SRPBS, BNU, and OpenNeuro/OpenfMRI dataset pages.
- medRxiv for current NeuroFM and critical-method preprints, labelled as such.

## Search strings

```text
("brain age" OR "brain-age") AND
  (MRI OR neuroimaging) AND (review OR biomarker)

("brain age delta" OR "predicted age difference") AND
  (bias correction OR age bias OR regression)

("brain age") AND
  (test-retest OR reliability OR longitudinal OR scanner OR site)

("brain age") AND
  (clinical utility OR external validation OR dementia OR mortality)

("foundation model" OR NeuroFM) AND
  (structural MRI OR brain age OR preprocessing)

("travelling subject" OR "traveling subject") AND
  (structural MRI OR harmonization)
```

Exact-title searches were run for SFCN, DeepBrainNet, SynthBA, brainageR,
NeuroFM, SIMON, SRPBS, BNU test-retest, and Maclaren ds000239.

## Inclusion rules

- Structural MRI age prediction or a directly relevant model comparison.
- Independent, longitudinal, test-retest, scanner/site, or outcome validation.
- Bias, calibration, harmonization, or clinical-utility analysis.
- Official model documentation when preprocessing or output schema was needed.

## Exclusion rules

- Internal training accuracy without enough information to interpret the split.
- Feature visualizations without an independent target or stability analysis.
- Disease claims inferred from an age output without an external outcome.
- Secondary reporting when a primary paper or official source was available.

## Verification notes

- The selected foundation model is `https://github.com/rockNroll87q/NeuroFM`,
  locally reviewed at commit `d4e3c46` on 2026-07-18.
- NeuroFM is a 2026 medRxiv preprint at review time, not peer-reviewed evidence.
- The official code requires a skull-stripped T1 and performs its own conforming
  and z-score normalization. Raw-orig inputs do not satisfy this contract.
- SIMON is one individual with 73 sessions and 36 scanners. SRPBS contains nine
  young male travelling subjects. Neither provides population age validation.
- Search results were screened by one reviewer; no PRISMA count or exhaustive
  coverage claim is made.
