# Poster vs Kate Brain Results

Date: 2026-06-16

Poster recognized from the user-provided photo/transcription:

**Assessing Site and Software-Related Variability in FastSurfer Brain Measures
Using Identical 3T Scanners**

Authors include Deborah Frueh, Weiyi Zeng, Santiago Estrada, Mohammad Shahid,
Ruediger Stirnberg, Philipp Ehses, Eberhard D. Pracht, Tony Stoecker, Martin
Reuter, and Monique M.B. Breteler.

## What The Poster Tested

The poster estimates non-biological variability in FastSurfer-derived brain
measures under controlled acquisition conditions.

Setup:

- Siemens MAGNETOM Prisma 3T scanners.
- Same/harmonized multi-modal protocol.
- Scanner software version comparison: `VE11C -> XA20`.
- FastSurfer pipeline version `2.4.2`.
- Longitudinal stream.
- Phenotypes:
  - cortical gray matter volumes, DKT;
  - cortical thickness, DKT;
  - subcortical volumes, aseg.
- Metrics:
  - ICC;
  - paired t-test;
  - relative difference percent;
  - linear mixed models.

Cohorts:

- Within-scanner test-retest: 90 participants, age 31-87, mean 55.2 years,
  68.0% women.
- Traveling heads: 16 participants, age 22-69, mean 38 years, 62.5% women.

## Poster Results

### Within-scanner test-retest

| Measure family | ICC | Mean relative difference | Significant regions after FDR |
|---|---:|---:|---:|
| Gray matter volume | 0.99 +/- 0.008, range 0.96-1.00 | 0.24% +/- 0.40%, range -1.13% to 1.57% | 5/62 |
| Cortical thickness | 0.92 +/- 0.058, range 0.71-0.98 | 0.17% +/- 0.37%, range -0.45% to 1.48% | 2/62 |
| Subcortical volume | 0.99 +/- 0.006, range 0.98-1.00 | 0.05% +/- 0.21%, range -0.24% to 0.37% | 0/22 |

### Site effect

| Measure family | ICC | Mean relative difference | Significant regions after FDR |
|---|---:|---:|---:|
| Gray matter volume | 0.98 +/- 0.018, range 0.87-1.00 | 0.70% +/- 0.89%, approximate range -2.24% to 2.26% | 3/62 |
| Cortical thickness | 0.93 +/- 0.053, range 0.64-0.98 | -0.13% +/- 0.47%, approximate range -1.22% to 1.29% | 0/62 |
| Subcortical volume | 0.99 +/- 0.008, range 0.96-1.00 | 1.15% +/- 0.41%, approximate range 0.55% to 1.94% | 15/22 |

### Scanner software upgrade effect

| Measure family | ICC | Mean relative difference | Significant regions after FDR |
|---|---:|---:|---:|
| Gray matter volume | 0.98 +/- 0.016, range 0.88-1.00 | 0.64% +/- 0.88%, range -1.16% to 3.22% | 0/62 |
| Cortical thickness | 0.93 +/- 0.046, range 0.78-0.98 | 0.22% +/- 0.73%, range -1.09% to 2.48% | 0/62 |
| Subcortical volume | 0.99 +/- 0.006, range 0.98-1.00 | 0.05% +/- 0.21%, range -0.24% to 0.37% | 0/22 |

Poster conclusions:

- Within-session variability is small but non-zero.
- Site effects introduce small systematic shifts, especially in some volumetric
  measures, while overall agreement remains high across identical scanner
  hardware/software.
- Scanner software upgrade effects are systematic and primarily affect
  subcortical structures.

## Our Existing Results

### SynthSeg rotation and TTA on Kate 2018

Source:

- `outputs/synthseg_rotation_tta_report.md`
- `reprocessed_2026/summary.md`

We do see the test-time augmentation results:

- rotation sweep: 9 angles from `-12` to `+12` degrees in `3` degree steps;
- median 9-angle TTA CV: `1.24%`;
- +/-3 degree SynthSeg floor: `1.43%`;
- interpolation-only floor: `0.05%`;
- model instability component: `1.36%`;
- cross-scanner median spread: `16.7%`;
- scanner/protocol spread is about `12x` the SynthSeg rotation floor.

This is a stronger stress test than the poster's within-scanner repeat scan,
because it isolates model sensitivity to orientation and interpolation.

### FastSurfer rotation on Kate 2018

Source:

- `outputs/fastsurfer_rotation_stability_report.md`

Same 2018 scan, +/-3 degree pair:

- median SynthSeg floor: `1.43%`;
- median FastSurfer floor: `1.48%`.

Interpretation: ordinary FastSurfer and SynthSeg have nearly the same median
rotation floor in this test, but different structure-level failure locations.

### FastSurfer Long rotation on Kate 2018

Source:

- `outputs/fastsurfer_long_symmetry_report.md`

Same 2018 scan, +/-3 degree pair:

| Metric family | Cross median CV | Long median CV | Improved |
|---|---:|---:|---:|
| Subcortical volumes | 0.24% | 0.12% | 10/12 |
| Cortical DKT parcel volumes | 0.36% | 0.10% | 53/62 |
| HypVINN volumes | 1.53% | 0.80% | 18/24 |

This is closest to the poster's FastSurfer longitudinal-stream framing, but our
test is a synthetic rotation-pair stability test, not site/test-retest.

### FreeSurfer 7.4.1 longitudinal on Kate 2018 vs 2022

Source:

- `outputs/fs_long_consistency_report.md`

2018 GE 3T 1 mm-like vs 2022 Siemens 1.5T thick-slice T1:

- subcortical median cross-sectional CV: `9.18%`;
- subcortical median longitudinal CV: `10.46%`;
- subcortical regions improved: `7/12`;
- regional cortical thickness median CV improved from `10.50%` to `5.83%`;
- cortical thickness regions improved: `43/68`;
- cortex volume CV improved from `21.19%` to `1.88%`;
- 2024 was excluded from FS7 longitudinal because it failed Talairach.

This is not comparable to the poster's identical-3T test-retest setup. It is a
cross-vendor/cross-protocol stress test.

## Direct Comparison

| Axis | Poster | Kate results | Interpretation |
|---|---|---|---|
| Scanner control | Identical Siemens Prisma 3T scanners, harmonized protocol | GE 3T 2018, Siemens 1.5T thick-slice 2022, Philips 1.5T 2024 3DI | Our acquisition variability is much larger and clinically messier. |
| Software | FastSurfer 2.4.2 longitudinal | FastSurfer 2.5.3 Long, FS7.4.1 Long, SynthSeg, FS8.2 running | Our method matrix is broader and includes failure/rescue branches. |
| Within-session floor | Poster: subcortical relative difference about 0.05%, GMV 0.24%, thickness 0.17% | FastSurfer Long rotation: subcortical 0.12%, cortical DKT volume 0.10%; SynthSeg TTA CV 1.24% | Our FastSurfer Long rotation floor is close to the poster's low-noise range; SynthSeg TTA is a harder perturbation. |
| Site effect | Poster: small but systematic; subcortical site effect significant in 15/22 regions | Our cross-scanner median spread by SynthSeg is 16.7% | Our scanner/protocol effect is far larger because scanners and protocols are not identical. |
| Software upgrade effect | Poster: mostly systematic, subcortical emphasis, but small relative differences | Our FS7 vs FS8 comparison is still running; ordinary FastSurfer failed 2024 3DI | We cannot yet claim FS8 upgrade effect; current evidence says contrast/protocol mismatch is a bigger issue than minor software upgrade. |
| Longitudinal processing | Poster uses FastSurfer longitudinal stream | FastSurfer Long improved rotation repeatability; FS7 Long improved cortical measures but mixed subcortical results | Longitudinal templates help when the input quality supports them; they cannot recover missing 2022 slice resolution. |
| Failure modes | Poster focuses on controlled variability, not failure cases | FS7 failed 2024 Talairach; FastSurfer collapsed on 2024 3DI | Our project adds failure mapping, which is absent or secondary in the poster. |

## Bottom Line

The poster establishes a low non-biological variability floor for FastSurfer
when scanners are nearly ideal and matched: identical 3T Siemens Prisma systems,
harmonized protocol, and controlled site/software differences.

Our data show a different regime:

1. Under controlled rotation of the same 2018 scan, our FastSurfer Long numbers
   are compatible with a low variability floor and in some summaries even lower
   than the poster's reported site-level differences.
2. Under real cross-year/cross-vendor/cross-protocol conditions, our variability
   is much larger than the poster's site/software effects.
3. The main problem in Kate's data is not ordinary FastSurfer noise. It is
   acquisition/protocol/contrast mismatch: 2022 thick slices and 2024 3DI
   contrast.
4. TTA is visible and useful in our results. It quantifies orientation/model
   instability, but it does not solve cross-scanner harmonization by itself.
