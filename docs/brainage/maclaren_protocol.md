# NeuroFM Maclaren test-retest protocol

**Protocol version:** 1.0, locked before NeuroFM predictions are inspected.

## Question and permitted claim

This experiment asks how repeatable the released NeuroFM-S outputs and latent
representation are across 40 short-interval T1 acquisitions of each of three
people in Maclaren ds000239. It is a technical robustness experiment at claim
level B2, conditional on this scanner, acquisition protocol, skull-strip method,
and three participants.

The participants are aged 26, 31, and 30 years. NeuroFM documents a brain-age
range of 40-90 years. Predicted ages are therefore out-of-range model outputs:
they may be used to quantify repeated-measure dispersion but not MAE,
calibration, biological age, or population accuracy. Stable embeddings do not
validate segmentation, morphometry, health, diagnosis, or any downstream task.

## Frozen dataset

- Dataset: OpenfMRI/OpenNeuro `ds000239`, release R1.0.1.
- Design: three participants, 40 T1w acquisitions each, one GE MR750 3 T
  protocol.
- Source processing: SPM12 defacing by the dataset submitter; images are not
  skull-stripped.
- Demographics: participant-level integer age and reported gender from
  `participants.tsv`; exact scan dates and age in days are unavailable.
- Inclusion: all 120 readable 3D T1w files with finite, non-empty data and the
  expected complete participant-by-run design.
- Exclusions: unreadable, non-3D, empty, non-finite, invalid voxel-size,
  duplicate, or unexpected participant/run files. No scan is excluded from a
  prediction after its model output is viewed.

The committed inclusion table stores paths relative to the controlled dataset
root and SHA-256 for every source image. Raw and derivative NIfTI files remain
outside Git.

## Locked pipeline

1. HD-BET 2.0.1, official release-2.0.0 checkpoint, CPU, TTA disabled. The
   low-memory wrapper uses one preprocessing and one export worker because the
   upstream CLI hard-codes 4 and 8 workers, which exceeds the 8 GB WSL runtime;
   the predictor architecture and checkpoint are unchanged.
2. Inspect success/failure, mask fraction, shape, voxel size, and hashes. A
   deterministic external montage is used for visual skull-strip QC but is not
   committed because it is an MRI derivative.
3. `rockNroll87q/NeuroFM` commit
   `d4e3c463910d939a681d24ebdeb26d44dea6878f`, NeuroFM-S, weight SHA-256
   `8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f`.
4. NeuroFM internal processing: cubic conforming to 1 mm isotropic,
   256 x 256 x 256, LIA, followed by whole-volume z-score normalization.
5. Named output schema is taken from code:
   `brain_age`, `sex`, `ventricle_volume`, `brain_volume`. A direct API versus
   summary-CSV schema smoke test must pass before the cohort run.
6. Outputs requested: `brain_health,latent`, using summary mode to avoid
   per-volume output files.

HD-BET outputs, masks, logs, NeuroFM aggregate arrays, and perturbation NIfTI
files are external artifacts. Git receives only scripts, source hashes, compact
tables, metadata, and the scientific report.

## Primary repeatability endpoint

For the brain-age head, calculate participant-specific SD, MAD, IQR, range, and
the pooled within-participant SD:

```text
SD_within = sqrt(sum_i sum_j (y_ij - mean_i)^2 / (N - n_subjects))
RC95 = 1.96 * sqrt(2) * SD_within
```

`RC95` is the expected 95% absolute difference scale for two repeated outputs
under the homoscedastic-error assumption. It is not an individual clinical
threshold.

The same dispersion measures are reported for ventricular and total-brain
volume. Sex-class consistency is reported as the number and fraction of class
changes within each participant; sex is an audit output, not a health score.

## Secondary endpoints

- ICC(2,1), absolute agreement, for each scalar output.
- ICC(3,1), consistency, as a sensitivity metric.
- Deterministic 2,000-resample session bootstrap intervals for ICC and pooled
  within-person SD, conditional on these three participants. They are not
  population confidence intervals.
- Failure rate and skull-strip mask-fraction dispersion.
- Per-dimension embedding ICC distribution.
- Cosine similarity and standardized L2 distance to each participant centroid.
- Leave-one-scan-out nearest-neighbour identity retention by cosine distance.
- Within-person versus between-person cosine-distance distributions.

Because `n_subjects=3`, ICC estimates and intervals are intrinsically unstable;
within-subject dispersion is the primary result.

## Locked perturbation screen

The perturbation set is created before viewing perturbed predictions from
`run-20` of each participant. These three scans are selected by rule, not by
image quality or baseline model output.

| Family | Levels | Implementation | Margin |
|---|---|---|---|
| Rotation | +/-1, +/-3, +/-5 degrees about each voxel axis | cubic interpolation, fixed grid/FOV | age +/-2 years |
| Resolution | 0.8, 1.0, 1.2 mm isotropic | one cubic resample; NeuroFM then applies its normal conform | age +/-2 years |
| Scale | 0.95 and 1.05 | cubic affine transform in a fixed grid/FOV | age +/-2 years; volumes +/-5% |

Report signed and absolute deltas from the unperturbed scan, failure rate, and
whether each observed delta lies inside the engineering margin. With only three
participants, no TOST equivalence claim is permitted. Scale is a numerical
sensitivity probe, not simulated biological growth or atrophy.

## Decision rules

- The run reaches B0 if provenance and named outputs are complete.
- It reaches limited B2 evidence if all 120 baselines pass preprocessing and
  inference and repeatability metrics are reported without tuning.
- The general B2 gate from the BrainAge metric contract requires ICC lower 95%
  bound above 0.75 and a 95th-percentile repeat difference no greater than five
  years. This dataset cannot establish a population lower bound with only three
  participants, even if the point estimate passes.
- No result from this experiment supports B1 age accuracy, B3 ageing rate, B4
  health association, or B5 clinical utility.

## References

Maclaren J, Han Z, Vos SB, Fischbein N, Bammer R. Reliability of brain volume
measurements: a test-retest dataset. *Scientific Data*. 2014;1:140037.
[doi:10.1038/sdata.2014.37](https://doi.org/10.1038/sdata.2014.37).

Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. SynthStrip: skull-stripping
for any brain image. *NeuroImage*. 2022;260:119474.
[doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474).

The executed pipeline uses HD-BET, not SynthStrip; the SynthStrip citation is
retained only as the alternative reviewed skull-strip route. HD-BET must be
cited from the exact package documentation in any resulting manuscript.
