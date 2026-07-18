# BrainAge and NeuroFM metric contract

**Status:** protocol for model comparison, robustness, and longitudinal
evaluation. Thresholds are engineering research gates, not clinical standards.

## Units, datasets, and leakage control

- The unit for age accuracy is a participant. Repeated scans never cross splits.
- Calibration and age-bias correction are fitted on development data only.
- Model selection uses a validation set; final metrics use locked external data.
- SIMON, SRPBS, BNU test-retest, and Maclaren repeated scans are held-out
  robustness controls. They are not used to tune preprocessing or thresholds.
- Report each cohort and model-preprocessing pair separately before pooling.
- Record model source, commit, checkpoint SHA-256, container/environment, and
  exact preprocessing command for every result.

## Claim ladder

| Level | Evidence | Permitted wording |
|---|---|---|
| B0 execution | Named outputs, provenance, input/output QC | "Inference completed" |
| B1 age prediction | Independent in-range age labels, accuracy and calibration | "The model predicts age in this cohort" |
| B2 robustness | Held-out repeats, scanners, and perturbations | "The output is robust within tested margins" |
| B3 longitudinal | Repeated people and slope/change analysis | "The output tracks change in this cohort" |
| B4 association | Independent phenotype, covariates, external replication | "The output is associated with this outcome" |
| B5 utility | Prospective decision study and comparator | "The output improves the tested decision" |

An embedding-only run remains B0 unless a predeclared, independently evaluated
probe supplies the missing target evidence.

## Cross-sectional age metrics

For true age `y`, predicted age `y_hat`, and raw `BAG = y_hat - y`, report:

- n attempted, n successful, failure rate, and reasons.
- MAE, median AE, RMSE, mean signed error, and 90th percentile AE.
- Pearson r and Spearman rho, but never as the sole accuracy metric.
- Calibration intercept and slope with 95% confidence intervals.
- Correlation and flexible smooth of BAG with chronological age.
- R-squared and performance against a training-mean baseline.
- Metrics by age bin, sex, site/vendor, field strength, protocol, TIV, and QC
  stratum where sample sizes permit.

Participant-stratified bootstrap confidence intervals are required. Report raw
and corrected predictions. Any correction model and its fitting cohort are part
of the released artifact.

## Test-retest and travelling-subject metrics

- ICC(2,1) with 95% CI for absolute agreement across acquisitions.
- Within-subject SD, median absolute deviation, SEM, and MDC95.
- Bland-Altman bias and limits of agreement.
- Variance components for participant, scanner/site, session, and residual from
  a mixed-effects model.
- Scanner/site maximum pairwise shift and 95th percentile absolute shift.
- Rank stability is secondary and cannot replace agreement.

Proposed B2 age-output gate: ICC lower 95% bound above 0.75, absolute mean site
bias no greater than 2 years, and 95th percentile repeat difference no greater
than 5 years. Individual-change use requires ICC lower bound above 0.90 and a
change larger than MDC95. Freeze these margins before opening held-out controls.

For SRPBS, estimate site effects with participant fixed or random effects and
report all nine participants. For SIMON, report every original T1 session with
age, scanner, protocol, QC status, and prediction. SIMON supports a within-case
trajectory and scanner stress test only; do not compute a population confidence
interval from its 73 sessions as if n=73 independent people.

## Longitudinal metrics

With at least three visits per participant, fit a mixed model such as:

```text
predicted_age ~ chronological_age + baseline_age + sex + site +
                (1 + time | participant)
```

Report population slope, 95% CI, difference from 1.0, participant slope
distribution, change-score MAE, negative-slope frequency, and residual
autocorrelation. A slope near 1 is longitudinal consistency, not proof that the
model measures a causal rate of biological ageing.

For a single travelling subject, present slope and scanner-adjusted sensitivity
analysis descriptively. Never fine-tune on that trajectory and then call the
same slope a blind result.

## Controlled robustness experiments

Run perturbations after creating one frozen, QC-passed model-correct input.
Preserve the unperturbed image and transformation matrices.

| Family | Predeclared levels | Primary output |
|---|---|---|
| Rotation | 1, 3, and 5 degrees about each axis | signed and absolute age delta |
| Resolution | simulate 0.8, 1.0, and 1.2 mm, then conform once | delta and failure rate |
| Scale | isotropic 0.95, 1.00, 1.05 within fixed FOV | age, brain-volume, and ventricle-volume delta |
| Translation | 1, 3, and 5 voxels per axis | output delta |
| Intensity | mild bias field, noise, and contrast changes | output delta and QC |

Interpolation order and number of interpolation operations are fixed. Scale
perturbation is a numerical sensitivity test, not a biologically realistic
change in head size. For age, use TOST against a predeclared equivalence margin
of +/-2 years; for volume outputs, use both absolute mm3 and a +/-5% engineering
margin. Report curves and confidence intervals even when equivalence fails.

## NeuroFM-specific contract

Use only the official `rockNroll87q/NeuroFM` repository and record its commit.
At reviewed commit `d4e3c46`:

- input is skull-stripped T1-weighted NIfTI;
- internal target is 256 x 256 x 256 at 1 mm isotropic in LIA;
- resampling uses cubic interpolation when required;
- intensity normalization is a whole-volume z-score;
- official predictor schema in code is `brain_age`, `sex`,
  `ventricle_volume`, `brain_volume`;
- supported age range is documented as 40-90 years;
- feature dimensions are 161, 256, and 512 for S, M, and L.

Before cohort inference, run a schema test that compares the named summary CSV
columns with direct API output. Do not infer positional order from README prose.
Record the skull-strip method and QC. Raw-orig is not a correct NeuroFM input.

### Predictor outputs

- Evaluate brain age under the general age contract.
- Compare total brain and ventricular volume against an independent method on a
  held-out subset; report bias, agreement, and TIV/size dependence.
- Treat sex output as a model audit variable, not a health score. Report AUROC,
  balanced accuracy, calibration, and age-by-sex interaction only with suitable
  labels and sample size.

### Foundation features

For repeated scans report per-dimension ICC distributions, standardized L2
distance, cosine similarity, and nearest-neighbour identity retention. Compare
within-person distances with between-person distances using participant-level
resampling. For cohort probes, use grouped nested cross-validation and
permutation tests; all visits from a person stay in one fold.

Feature stability does not validate age, segmentation, morphometry, dementia,
vascular health, cognition, or any other downstream construct.

## Model comparison

Each model receives its own official preprocessing. Compare complete pipelines,
not isolated networks on a convenient shared volume. Minimum comparison set:

- NeuroFM age head from `rockNroll87q/NeuroFM`.
- SynthBA with its released preprocessing and weights.
- At least one established T1 model such as SFCN/pyment, DeepBrainNet, or
  brainageR, subject to reproducible access and licensing.
- A simple age-mean baseline and, for extracted morphometry, a transparent
  regularized regression baseline.

Rank models on a vector of metrics rather than a single score. Predeclare the
primary endpoint and use Pareto reporting for MAE, calibration, repeatability,
scanner variance, runtime, and failure rate.

## Required outputs

1. Input manifest with random study ID, age, visit, scanner/site, source hash,
   preprocessing status, and QC status.
2. One prediction table per complete model pipeline.
3. Accuracy and calibration table with participant-level confidence intervals.
4. Test-retest/travelling variance table.
5. Longitudinal slope table.
6. Perturbation-response table.
7. NeuroFM feature-stability table when embeddings are produced.
8. Failure table and claim-level conclusion B0-B5.

No raw MRI, NIfTI derivatives, model weights, logs, or identifiable paths are
committed to Git.
