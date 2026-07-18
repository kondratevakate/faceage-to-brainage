# Dataset acquisition plan

**Version:** 1.0, 2026-07-18.

The machine-readable registry is
[`data/research_dataset_registry.csv`](../data/research_dataset_registry.csv).
It records candidate use, access, overlap risk, local status, and the next
action. Registry inclusion does not mean that access has been granted or that a
dataset is valid for every model.

Public archive provenance is recorded separately in
[`data/public_dataset_checksums.csv`](../data/public_dataset_checksums.csv).
The registry and checksum table are versioned; raw archives and extracted MRI
remain outside Git.

## Dataset roles

| Role | Purpose | Examples | Not sufficient for |
|---|---|---|---|
| Development/QC | Build and debug deterministic pipelines | Kate n=1 | Accuracy or population claims |
| Scanner stress test | Hold identity while scanner/site varies | SIMON, SRPBS | Demographic calibration |
| Short-interval retest | Estimate measurement noise | Maclaren, BNU1 | Long-term ageing rate |
| External age validation | Accuracy and calibration in unseen people | IXI, CamCAN, OpenBHB components | Validity if model trained on the same people/source |
| Longitudinal/outcome | Change and outcome association | OASIS-3, NKI-RS | Same-scan face analysis if data are defaced |
| Face geometry | Photo-to-scan metric validation | NoW, Headspace, FaceScape, FaceBase | Brain-age validation |
| Paired face-brain | Primary cross-modal hypothesis | verified non-defaced MRI cohorts; prospective cohort | Clinical utility without outcomes |

## Acquisition order

### P0: freeze existing local resources

1. Preserve Kate, SIMON, and SRPBS raw data outside Git and create immutable
   file-hash manifests in controlled storage.
2. Record which SIMON scans are original T1, which are derivatives, and why each
   scan is included or excluded.
3. Keep NeuroFM raw-orig results as a documented invalid-input sensitivity
   branch. Official NeuroFM results require skull stripping and model-correct
   conforming.
4. Do not tune on SIMON or SRPBS after viewing the blind robustness endpoints.

### P1: acquire low-friction robustness controls

- **Maclaren ds000239:** 3 participants scanned 40 times, small and openly
  licensed. Release R1.0.1 was downloaded and integrity-checked on 2026-07-18;
  all 130 extracted files are covered by a verified external SHA-256 manifest,
  including 120 T1w scans. The next gate is the subject-session inclusion table
  and locked analysis. Use for within-scanner repeat noise and perturbation
  checks.
- **BNU1 test-retest:** 57 young adults with two T1 sessions about six weeks
  apart. Use for participant-level repeatability.
- **IXI:** nearly 600 healthy multi-site adults with demographics and raw T1.
  Verify face coverage and model-training overlap before assigning a paired or
  external role.

Downloads live in the external data root, not this repository. Each acquisition
gets source URL, access date, license/DUA snapshot, archive hash, extraction
command, and a BIDS/manifest validation report.

### P2: request controlled or licensed resources

- **NoW:** request benchmark access for photo-to-3D metric validation.
- **FaceBase 3D Facial Norms:** request controlled individual meshes and
  metadata; note the 3-40 age range and cross-sectional design.
- **Headspace/LYHM:** request the full-head dataset and verify age distribution,
  redistribution terms, and permitted publications.
- **FaceScape:** institutional application; only approved portrait indices may
  be published and use is non-commercial.
- **CamCAN, OASIS-3, and NKI-RS:** apply for age, longitudinal, cognitive, and
  health-outcome analyses. Determine whether distributed T1 data are defaced
  before assigning a FaceAge role.

### P3: prospective paired cohort

Public data rarely combine standardized photos, metric facial scans, non-defaced
T1, repeated acquisition, age, and health covariates. The confirmatory design
therefore includes a prospective controlled-access cohort if no existing cohort
passes the audit.

Target:

- 480 adults aged 20-79, balanced by sex and 10-year age band;
- at least 60 short-interval repeat participants;
- development, calibration, and locked holdout assigned by participant;
- a second independent cohort or site for replication.

Core acquisition:

- non-defaced 1 mm T1 MRI with acquisition and scanner metadata;
- standardized frontal and oblique photographs with fixed distance, focal
  length, lighting, neutral expression, and calibration target;
- optical 3D facial scan in a validation subset;
- exact age in days, sex at birth, height, weight/BMI, blood pressure, major
  neurological history, recent acute illness, major weight change, dental or
  cosmetic intervention, and image-quality fields;
- consent for biometric processing, linkage, controlled sharing, and withdrawal.

Optional outcome modules should be added only with a predeclared question:
FLAIR/SWI/TOF/DWI for vascular or microstructural endpoints, cognition for
neurodegeneration, laboratory measures for lipids/inflammation, and validated
questionnaires or sensors for behavioural exposures.

## Training-overlap audit

Before evaluation, create a model-dataset ledger with:

- model/version/checkpoint hash;
- all declared pretraining, training, validation, and calibration sources;
- component datasets inside aggregates such as OpenBHB;
- participant or source overlap when identifiers permit;
- age range, sex balance, site/vendor, field strength, and preprocessing domain.

IXI is a component of OpenBHB and has been used by some released brain-age
models. It cannot be called external for those models. The same rule applies to
IMDB-WIKI and UTKFace for facial models trained on them. Unknown overlap is
reported as unknown, not assumed absent.

## Data structure outside Git

```text
<controlled_data_root>/
  raw/<dataset>/<release>/
  sourcedata/<dataset>/<release>/
  derivatives/<pipeline>/<version>/<dataset>/
  manifests/<dataset>/<release>/
  metadata/model_dataset_overlap.csv
  locked_holdout/
```

The manifest contains random study IDs, paths relative to the controlled root,
SHA-256 hashes, modality, visit, age, site/scanner, license, and QC state. The
linkage key and direct identifiers are stored separately with restricted access.

## Acceptance checklist

A dataset enters analysis only when all items are resolved:

1. License/DUA permits the intended processing and reporting.
2. Participant and session units are understood.
3. Chronological age precision and scan date relationship are known.
4. Modality, voxel size, orientation, defacing, and face coverage are audited.
5. Model-training overlap is known or explicitly marked unknown.
6. Repeated observations are groupable by participant.
7. Raw-file hashes and extraction provenance exist.
8. Missingness and QC rules are frozen before outcome analysis.

## Immediate deliverables

1. Freeze the Maclaren subject-session inclusion table and run the locked
   short-interval repeatability protocol.
2. Download and hash BNU1 in external controlled storage.
3. Audit the local IXI release for demographics, face coverage, and overlap with
   each candidate model.
4. Submit NoW, FaceBase, Headspace, FaceScape, CamCAN, OASIS-3, and NKI-RS access
   requests where institutional credentials and approvals are available.
5. Build the prospective ethics/consent package only after the public-data gap
   audit is complete.
