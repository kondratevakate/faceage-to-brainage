# Preregistration draft: external validation of BrainAge

**Template:** OSF Secondary Data Preregistration

**Status:** working draft; bracketed fields block submission

**Protocol owner:** Ekaterina Kondratyeva and `[LOCK CONTRIBUTORS]`

## Title

External validation, calibration and technical robustness of a frozen BrainAge
pipeline in `[LOCK DATASET AND RELEASE]`.

## Research question

How accurately and reliably does a frozen T1-MRI BrainAge pipeline estimate
chronological age in an unseen cohort within the model's documented input and
age domain?

This is a prediction-measurement validation. Brain-age gap is not interpreted as
a causal rate of biological ageing, disease, prognosis or clinical utility.

## Prior knowledge and data history

Before this registration, the investigators developed and ran BrainAge code on
development or robustness resources including Kate, SIMON, Maclaren and BNU1,
and may have viewed IXI outputs. The final appendix will enumerate every dataset,
prediction and aggregate result previously inspected:
`[LOCK PRIOR-DATA APPENDIX URL/HASH]`.

Maclaren failed the project's predeclared 5-year 95th-percentile repeat-difference
screen and is below NeuroFM's documented 40-90 year range. BNU1 is also below
that range. These findings remain prior robustness evidence and cannot be
relabeled as confirmatory age-accuracy results.

For the target dataset, investigators have seen only `[LOCK PUBLIC PAPER,
FEASIBILITY COUNTS, DATA DICTIONARY AND ANY UNLABELED IMAGE QC]`. No predictions,
participant age residuals or primary aggregate results have been inspected.

## Dataset and eligibility

- Dataset/release: `[LOCK NAME, HOLDER, RELEASE/EXTRACT DATE]`.
- Access and ethics: `[LOCK DUA, IRB/ETHICS APPROVAL OR DETERMINATION]`.
- Expected sample: `[LOCK N PEOPLE, N FAMILIES, N VISITS, AGE/SITE DISTRIBUTION]`.
- Required inputs: original T1 MRI, age at scan, participant/visit ID, scanner
  metadata and enough provenance to reproduce preprocessing.
- Primary model's documented age and input domain: `[LOCK]`. If NeuroFM is
  primary, confirmatory accuracy is restricted to age 40-90 and skull-stripped
  T1 processed by the official pipeline.
- Known model-training sources and target overlap: `[LOCK MODEL-DATASET LEDGER]`.
  A cohort used for training, calibration or model selection is not external.

Twin relatives and repeat visits are grouped by family/participant. Site and
scanner sampling are described before model outputs are opened.

## Frozen pipelines

Primary complete pipeline: `[LOCK ONE MODEL, REPOSITORY, COMMIT, CHECKPOINT
SHA-256, INPUT CONTRACT AND PREPROCESSING]`.

Comparators: `[LOCK SYNTHBA / SFCN / OTHER RELEASED MODEL]` and a transparent
constant-age baseline. Each model receives its official preprocessing; one
shared convenient volume is not substituted unless that is the model's own
specified input.

Archive the environment/container digest, skull-strip and conform commands,
orientation/voxel checks, intensity normalization, QC rubric, failure rules and
any pre-fitted age-bias correction. Model or preprocessing selection will not
use target-test predictions or labels.

## Hypothesis and primary estimand

**H1:** the frozen primary BrainAge pipeline has lower participant-level MAE
than a predeclared constant-age baseline fitted without target-test labels.

Define
`delta_MAE = MAE(BrainAge) - MAE(baseline)`. H1 is supported only if the
family-cluster bootstrap 95% confidence interval has an upper bound below zero.
Baseline definition and numeric value: `[LOCK]`.

Claim-level B1 additionally requires calibration slope and intercept to satisfy
predeclared non-catastrophic margins `[LOCK MARGINS AND RATIONALE]`. A model may
beat the baseline yet fail this calibration gate.

If several learned models are compared, the named primary pipeline alone tests
H1. Comparator ranking is secondary and adjusted using `[LOCK MULTIPLICITY
METHOD]`; a better comparator does not replace the primary after results are
seen.

## Secondary estimands

- attempted/successful N, failure rate and reason;
- MAE, median and 90th-percentile AE, RMSE and mean signed error;
- calibration intercept/slope, BAG-age association, Pearson/Spearman correlation
  and R-squared;
- performance by age band, sex, site/vendor, field strength, protocol, TIV and
  QC stratum where sample size permits;
- test-retest ICC(2,1), within-person SD, MDC95 and site/scanner variance when
  repeated or travelling-participant scans exist;
- raw versus pre-fitted corrected predictions;
- complete-pipeline runtime and resource use.

Embeddings, morphometry, sex output and exploratory downstream probes are
separate secondary families. They do not validate BrainAge and will not be used
to rescue a failed H1.

## Exclusions and missingness

Exclusions are applied without prediction-error information:

1. invalid consent/DUA or missing participant/visit identity;
2. missing or insufficiently precise age at scan;
3. age outside the primary model's documented range;
4. duplicate source image by hash;
5. non-T1 or acquisition incompatible with the frozen input contract;
6. frozen raw/preprocessing QC failure: `[LOCK OPERATIONAL RULES]`;
7. predeclared out-of-scope clinical or scanner stratum: `[LOCK OR NONE]`.

Inference/preprocessing failures remain in the attempted denominator and are not
silently excluded. Primary age accuracy uses participants with valid T1 and age;
covariate-adjusted secondary analyses use `[LOCK COMPLETE CASE OR IMPUTATION
PLAN]`.

## Statistical analysis

The participant is the accuracy unit. For repeated scans, the frozen primary
visit rule is `[LOCK]`; repeat scans are used in a separately named robustness
analysis. Confidence intervals use at least `[LOCK >=2000]` bootstrap replicates,
clustered by family for twins and by participant otherwise. Multi-site results
are reported per site and combined by a stratified bootstrap or hierarchical
model `[LOCK CHOICE]`.

Calibration/bias correction is never fitted to test labels. All predeclared
metrics and all failures are reported, including when H1 or the calibration gate
fails.

## Sample size and precision

Before submission, perform a precision/power simulation using expected N,
family clustering, site structure and development-only error distributions.
Lock:

- analyzable N and expected attrition: `[LOCK]`;
- target CI half-width for MAE and `delta_MAE`: `[LOCK]`;
- minimum detectable MAE improvement at 80%/90% power: `[LOCK]`;
- minimum per-site/subgroup N for inferential reporting: `[LOCK]`;
- rule for shortfall: `[LOCK ANALYZE WITH PRECISION WARNING / DECLARE
  INSUFFICIENT WITHOUT HYPOTHESIS TEST]`.

## Deviations, outputs and interpretation

Any amendment is timestamped and its knowledge state described. A failed model
is reported under this registration; replacement pipelines are exploratory or
receive a new registration and new untouched holdout.

Required outputs follow the
[BrainAge metric contract](../brainage/metrics_protocol.md): manifest and
provenance, complete prediction/failure tables, accuracy and calibration,
test-retest/site robustness where available, subgroup results and the highest
supported claim level B0-B5.
