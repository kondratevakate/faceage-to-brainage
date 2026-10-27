# Preregistration draft: external validation of FaceAge

**Template:** OSF Secondary Data Preregistration

**Status:** working draft; bracketed fields block submission

**Protocol owner:** Ekaterina Kondratyeva and `[LOCK CONTRIBUTORS]`

## Title

External validation, calibration and repeatability of a frozen facial-age
measurement pipeline in `[LOCK DATASET AND RELEASE]`.

## Research question

How accurately and consistently does a frozen FaceAge pipeline estimate
chronological or independently rated apparent age in people not used for model
training, pipeline selection or calibration?

This registration evaluates measurement validity. It does not claim that the
output is biological age, causal ageing rate, health status or clinical utility.

## Prior knowledge and data history

Before this registration, the investigators developed code, reviewed published
benchmarks and inspected project development/QC outputs, including Kate/Case A,
SIMON and any IXI analyses listed in the final model-dataset ledger. These are
not confirmatory data. All datasets, images, predictions, aggregate metrics and
manuscript results previously viewed will be enumerated in
`[LOCK PRIOR-DATA APPENDIX URL/HASH]`.

For the target dataset, the investigators have seen only `[LOCK EXACTLY WHAT HAS
BEEN SEEN: PUBLIC PAPER, FEASIBILITY COUNTS, DATA DICTIONARY, UNLABELED QC]`.
No target FaceAge predictions, age-stratified reconstruction outputs or primary
accuracy results have been inspected as of submission. Any exception will be
described and the affected analysis labelled exploratory.

## Dataset and eligibility

- Dataset/release: `[LOCK NAME, HOLDER, RELEASE/EXTRACT DATE]`.
- Access and ethics: `[LOCK DUA, IRB/ETHICS APPROVAL OR DETERMINATION]`.
- Intended population: `[LOCK AGE RANGE, GEOGRAPHY, RECRUITMENT SETTING]`.
- Expected sample: `[LOCK N PEOPLE, N FAMILIES, N VISITS AND AGE DISTRIBUTION]`.
- Required inputs: source face photographs or 3D surfaces, age at acquisition,
  stable participant/visit ID and capture-quality metadata.
- Twin/family data require stable family and zygosity fields; all relatives are
  retained in the same resampling unit.
- Training overlap must be absent or explicitly quantified. Unknown overlap is
  reported as unknown and prevents the word “independent” without qualification.

## Frozen pipeline

Before registration, archive:

- model name/version/checkpoint SHA-256: `[LOCK]`;
- code repository commit and environment/container digest: `[LOCK]`;
- face detection, crop, color normalization, view handling and failure rules:
  `[LOCK]`;
- photo/3D input specification and any camera or scale calibration: `[LOCK]`;
- participant-level aggregation across images/views: `[LOCK]`;
- calibration/bias-correction map and the non-target data on which it was fit:
  `[LOCK OR STATE NONE]`.

No target labels will be used to select a crop, view, model, ensemble,
calibration map or exclusion threshold. If a dedicated calibration partition is
allowed, it will be separated by family before the test partition is opened and
will never be included in final test metrics.

## Hypothesis and primary estimand

**H1:** the frozen FaceAge pipeline has lower participant-level MAE than a
predeclared constant-age baseline fitted without target-test labels.

Define
`delta_MAE = MAE(FaceAge) - MAE(baseline)`. The primary estimand is
`delta_MAE` in all eligible target participants. H1 is supported only if the
family-cluster bootstrap 95% confidence interval has an upper bound below zero.

Baseline definition: `[LOCK TRAINING/DEVELOPMENT MEAN OR OTHER TRANSPARENT
BASELINE AND ITS NUMERIC VALUE]`. The FaceAge prediction target is
`[LOCK CHRONOLOGICAL AGE / APPARENT AGE]`; these are not interchangeable. If
both are present, one is primary and the other is secondary.

Claim-level F2 additionally requires calibration slope and intercept to pass the
predeclared non-catastrophic calibration margins `[LOCK MARGINS AND RATIONALE]`.
Failure of this gate is reported even if H1 is supported.

## Secondary estimands

- MAE, median and 90th-percentile absolute error, RMSE and mean signed error;
- calibration intercept/slope, residual-age association, Pearson/Spearman
  correlation and R-squared;
- failure rate and reason, including all attempted eligible participants;
- repeatability ICC(2,1), within-person SD, MDC95 and Bland-Altman limits when
  repeated captures exist;
- rigid metric geometry error against 3D face/MRI surface only when scale and a
  face mask are frozen in advance;
- performance by predeclared age band, sex, site/camera, image QC and
  ancestry/skin-tone representation where cell size permits.

Subgroup and regional-face analyses are secondary and multiplicity-controlled
within named families using `[LOCK METHOD]`. Small cells are summarized without
unstable inferential claims.

## Exclusions and missingness

Apply exclusions without reference to prediction error:

1. no valid consent/DUA for the planned processing;
2. missing or ambiguous participant/family ID;
3. missing acquisition age or age precision below `[LOCK MINIMUM]`;
4. duplicate/re-exported capture identified by source hash;
5. frozen image-QC failure: `[LOCK OPERATIONAL RULES]`;
6. out-of-scope age or clinical stratum defined before predictions are run.

Model/detector failures remain in the attempted denominator. There is no
complete-case substitution or removal based on large error. Covariate
missingness is tabulated; primary accuracy uses every participant with image and
age, while adjusted secondary models use `[LOCK COMPLETE CASE OR IMPUTATION
PLAN]`.

## Statistical analysis

The participant is the prediction unit. Multiple images/views are aggregated by
the frozen rule. Confidence intervals use at least `[LOCK >=2000]` bootstrap
replicates with family as the cluster for twin/family data and participant as the
cluster otherwise. Site is respected by stratification or a hierarchical model
when more than one site contributes.

Raw and any pre-fitted corrected predictions are reported side by side. No
correction is fitted on test labels. Full participant flow, failures and
predeclared metrics are released regardless of direction.

## Sample size and precision

This is a fixed-availability secondary dataset. Before registration, run a
label-blind precision calculation or simulation using the expected N, family
structure and plausible error distribution from development data. Lock:

- analyzable N and expected attrition: `[LOCK]`;
- target 95% CI half-width for MAE and `delta_MAE`: `[LOCK]`;
- minimum detectable MAE improvement at 80%/90% power: `[LOCK]`;
- rule if access yields fewer participants: `[LOCK ANALYZE WITH PRECISION WARNING
  / DECLARE INSUFFICIENT WITHOUT HYPOTHESIS TEST]`.

## Deviations, outputs and interpretation

All deviations are dated before affected analysis where possible and labelled
confirmatory-preserving or exploratory. A failed model, null result or failed
calibration gate is not replaced by a newly selected pipeline under this
registration.

Required outputs follow the
[FaceAge metric contract](../faceage/metrics_protocol.md): participant flow,
cohort table, provenance, all aggregate metrics with uncertainty, failure table,
subgroup table and a statement of the highest passed claim level F0-F5.
