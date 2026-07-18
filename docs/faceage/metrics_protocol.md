# FaceAge metric contract

**Status:** protocol for development and external evaluation. Thresholds below
are project decision gates, not clinical reference standards.

## Units and split policy

- The statistical unit is a participant, not an image, crop, view, or session.
- All images, views, and visits from one participant stay in one split.
- Hyperparameters, calibration, bias correction, thresholds, and region choices
  are fitted on development data only.
- The external test set remains locked until the pipeline and exclusions are
  frozen. SIMON- or Kate-like repeated cases are QC data, not accuracy cohorts.
- Report results per dataset before any pooled estimate. A pooled result must use
  a hierarchical model or dataset-stratified bootstrap.

## Claim ladder

| Level | Required data | Required evidence | Permitted wording |
|---|---|---|---|
| F0 pipeline | At least one case | Complete run, provenance, visual QC | "The pipeline runs" |
| F1 repeatability | Repeated standardized photos or views | Agreement and within-subject error | "The output is repeatable under this protocol" |
| F2 age prediction | Independent people with age labels | Accuracy, calibration, subgroup results | "The model predicts chronological/apparent age" |
| F3 metric geometry | Photo plus 3D scan or facial MRI surface | Error in millimetres under fixed alignment | "The reconstruction has measured geometric error" |
| F4 association | Independent health outcome and covariates | Adjusted external association and replication | "The feature is associated with the outcome" |
| F5 utility | Prospective decision study | Calibration, discrimination, decision benefit | "The model has utility in the tested setting" |

No level inherits the next claim automatically.

## Age endpoints

For chronological or apparent age `y` and prediction `y_hat`:

- MAE: mean absolute error.
- Median absolute error and 90th percentile absolute error.
- RMSE, reported with MAE because it emphasizes large errors.
- Mean signed error (bias) and its 95% confidence interval.
- Calibration intercept and slope from `y = a + b * y_hat`.
- Pearson and Spearman correlation with 95% intervals.
- Residual-age correlation. A non-zero value flags age-dependent bias.
- R-squared, including a comparison with an age-only or train-mean baseline.

Bootstrap confidence intervals resample participants, never individual photos.
MAE is always accompanied by the test age range and age distribution.

### Bias correction

Any linear, isotonic, or other calibration map is fitted inside training folds or
on a dedicated calibration set. Report raw and corrected predictions. Applying a
correction fitted on the test labels invalidates the test result.

## Repeatability endpoints

Repeated photos should include same-session repeats and a 7-14 day repeat where
feasible. Record camera, focal length, distance, illumination, pose, expression,
time, body mass, acute illness, and cosmetic changes.

- ICC(2,1) with 95% CI for absolute agreement across repeated acquisitions.
- ICC(3,1) only when the tested camera/raters are fixed and not generalized.
- Within-subject standard deviation `s_w`.
- Standard error of measurement and `MDC95 = 1.96 * sqrt(2) * s_w`.
- Bland-Altman mean difference and 95% limits of agreement.
- Lin concordance correlation coefficient for paired repeats.
- Failure rate and missingness by capture condition.

Proposed minimum research gate: ICC(2,1) lower 95% bound above 0.75 and median
absolute repeat difference no greater than 2 years for an age output. A claim
about individual change requires the stronger gate of lower bound above 0.90 and
an observed change larger than MDC95. These margins must be frozen before the
locked test and reported even if failed.

## Longitudinal endpoints

For participants with at least three visits, fit a mixed model with participant
random intercept and, when supported, random slope:

```text
predicted_age ~ chronological_age + sex + site/camera + (1 + time | participant)
```

Report the population slope, its distance from 1.0, participant-level slope
distribution, change-score MAE, and the fraction of negative slopes. Two visits
can estimate change but not distinguish linear trend from measurement noise.

## Geometry endpoints

Evaluation requires a predeclared face mask and transformation policy.

1. Report both rigid metrical alignment and, separately, similarity alignment.
2. Similarity-aligned error is scale-free and cannot support a metric-size claim.
3. Do not crop or choose landmarks after viewing test error.
4. Use the same valid regions for every method or report method-specific coverage.

Primary geometry metrics:

- Landmark RMSE in millimetres, with per-landmark errors.
- Symmetric point-to-surface median, mean, and 95th percentile distance.
- ASSD and HD95; maximum Hausdorff distance is secondary because it is unstable.
- Surface-normal angular error.
- Valid-surface coverage and reconstruction failure rate.
- Regional errors for forehead, periorbital, nose, cheeks, perioral, and chin
  masks defined before testing.

Chamfer distance may be reported for comparison with vision literature, but its
units, sampling density, and squared/non-squared definition must be explicit.

## Robustness perturbations

Apply controlled perturbations to the same source photograph: yaw/pitch/roll,
crop margin, image downsampling, JPEG quality, illumination, and detector jitter.
For each output report signed delta, absolute delta, 95th percentile delta,
failure rate, and an equivalence test against a predeclared margin. Perturbation
robustness does not replace real repeated acquisition.

## Subgroups and confounders

At minimum report age bins, sex, ancestry/skin-tone representation, camera/site,
BMI when available, and image-quality strata. Menopause, facial hair, cosmetic
procedures, dental status, and major weight change are candidate modifiers, not
labels to infer from a photograph.

Health-outcome models must include chronological age and a predeclared covariate
set. Report the incremental likelihood ratio or change in validated performance
over the clinical baseline, calibration, and decision-curve analysis where a
decision is actually proposed. Correct families of exploratory regional tests
for multiple comparisons.

## Required output tables

Every released evaluation should contain:

1. A participant flow and exclusion table.
2. Dataset, split, age, sex, and capture summaries.
3. Model/checkpoint hash and preprocessing provenance.
4. Participant-level predictions, with private identifiers replaced by random
   study IDs in controlled storage.
5. Aggregate metrics with confidence intervals and failure counts.
6. Repeatability and perturbation tables.
7. Geometry metrics under rigid and similarity alignment.
8. A claim-level statement identifying the highest passed level F0-F5.

## Statistical references

- Koo TK, Li MY. A Guideline of Selecting and Reporting Intraclass Correlation
  Coefficients for Reliability Research.
  [doi:10.1016/j.jcm.2016.02.012](https://doi.org/10.1016/j.jcm.2016.02.012).
- Moons KGM, et al. PROBAST+AI.
  [doi:10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505).
- The NoW challenge. [Official metric specification](https://now.is.tue.mpg.de/).
