# BNU1 BrainAge test-retest protocol

**Protocol version:** 1.0, locked before BNU1 model outputs are generated or
inspected.

## Scope and permitted claim

BNU1 is a paired repeatability cohort for complete brain-age pipelines, not an
age-accuracy cohort for NeuroFM. The official FCP-INDI snapshot contains 57
healthy adults aged 19-30 years, 30 reported male and 27 reported female, with
two sessions approximately six weeks apart. Only 50 participants have a T1 file
at both sessions in the public BIDS tree; seven have baseline T1 only. One of
the 50 nominal pairs has a truncated session-2 T1, leaving 49 primary QC-passed
pairs.

All participants are below NeuroFM's documented 40-90-year range. NeuroFM
outputs may quantify technical repeatability but cannot support MAE,
calibration, biological-age, health, diagnosis, segmentation, or morphometry
claims in this cohort. Stable embeddings remain feature QC only.

## Frozen acquisition and inclusion

- Source: FCP-INDI CoRR `BNU_1`, DOI
  [10.15387/fcp_indi.corr.bnu1](https://doi.org/10.15387/fcp_indi.corr.bnu1).
- Snapshot: anonymous S3 inventory retrieved 2026-07-18.
- Integrity: every selected object must match its single-part S3 ETag/MD5 and a
  local SHA-256 is retained in external provenance.
- Selected source objects: 107 T1 NIfTI files, 57 per-participant session
  tables, `participants.tsv`, and `T1w.json`.
- Header/numeric QC: readable 3D image, finite non-empty data, valid positive
  voxel sizes, unique participant/session key, matching acquisition hash, and
  shape `144 x 256 x 256` as specified by `T1w.json` and observed in 106 of 107
  T1 files.
- Primary paired cohort: the 50 participants with both session-1 and session-2
  T1 files, provided both scans pass the frozen QC. The session-2 T1 for
  `sub-0025913` has only 81 rather than 144 slices and approximately 102 mm
  first-axis nonzero coverage, versus approximately 178 mm in session 1. That
  source pair is excluded before model inference. Seven baseline-only
  participants and the QC-excluded source pair remain in the participant flow
  and are not treated as failures of inference.

The session tables report integer age at session 1 and exact retest duration in
days. Session-2 chronological age is therefore an approximation equal to
baseline integer age plus elapsed days/365.2425. It is not precise enough for a
sub-year age-accuracy claim.

## Model pipelines

Each model must receive its official preprocessing. The NeuroFM route uses the
official `rockNroll87q/NeuroFM` code and weights frozen in the BrainAge metric
contract, with a separately recorded skull-strip method and QC. No convenience
preprocessing is shared across models unless it is part of each released
pipeline.

Raw MRI, skull-stripped NIfTI, masks, model weights, aggregate latent arrays,
and logs remain external. Git receives code, source hashes, compact de-identified
tables, and reports only.

## Primary repeatability endpoints

For each scalar output, compute paired session difference `d = session2 -
session1` and report:

- number attempted, successful pairs, failures, and missing source pairs;
- mean paired bias and its participant-bootstrap 95% interval;
- SD of paired differences and Bland-Altman limits of agreement;
- median, 95th percentile, and maximum absolute paired difference;
- within-subject repeatability SD `SD(d) / sqrt(2)`;
- ICC(2,1) absolute agreement and ICC(3,1) consistency with participant-level
  bootstrap intervals.

For the age head, the predeclared engineering screen is a 95th-percentile
absolute paired difference no greater than five years. This is not a clinical
threshold or an equivalence claim. Fifty pairs can estimate paired dispersion
more credibly than Maclaren's three participants, but transportability remains
limited to this young, single-site, single-scanner cohort.

Volume heads are assessed in absolute and fractional units. The sex output is
an audit variable; report within-person class changes, not a health score.

## Feature endpoints

For embeddings, report paired cosine distance, standardized L2 distance,
nearest-neighbour identity retention, and per-dimension agreement. Compare
within-person pair distances with between-person distances using
participant-level resampling. No downstream label is inferred from feature
stability alone.

## Decision rules

- B0 requires complete provenance, input QC, named output schema, and failure
  accounting.
- Limited B2 evidence requires the frozen 49-pair cohort to complete without
  tuning and all repeatability endpoints to be reported.
- BNU1 cannot establish NeuroFM B1 age accuracy because every participant is
  outside the documented age range.
- BNU1 cannot establish scanner generalization, longitudinal ageing rate,
  health association, or clinical utility.

## References

Lin Q, et al. A connectivity-based test-retest dataset of multi-modal magnetic
resonance imaging in young healthy adults. *Scientific Data*. 2015;2:150056.
[doi:10.1038/sdata.2015.56](https://doi.org/10.1038/sdata.2015.56).

FCP-INDI. BNU 1 - Beijing Normal University (He, Lin).
[Official dataset page](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/bnu_1.html).
