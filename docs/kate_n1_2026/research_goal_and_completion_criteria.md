# Research Goal and Completion Criteria

Date: 2026-06-27

## Rewritten Goal

Build a reproducible brain MRI segmentation benchmark that compares modern and
emerging segmentation pipelines on both Kate's longitudinal personal MRI scans
and external test-retest or travelling-head datasets, with test-time
augmentation as the central mechanism for estimating segmentation stability and
uncertainty.

The study must determine which pipeline, or pipeline combination, is most robust
for heterogeneous real-world brain MRI inputs, and must state when an output is
safe for visualization, when it is useful only as QC or uncertainty evidence,
and when it should be rejected as a failure mode.

## Scientific Claim To Support

The final claim should be:

> For this set of heterogeneous MRI inputs and validation datasets, pipeline X or
> consensus strategy Y gives the best tradeoff between anatomical plausibility,
> test-retest stability, spatial agreement, runtime feasibility, and calibrated
> uncertainty; outputs outside the stated quality gates should not be interpreted
> anatomically.

The study should not claim manual ground truth, biological disease status, or
true anatomical correctness unless those are supported by manual labels,
phantom/histology evidence, or an external validated reference.

## Scope

### Data Scope

Required application data:

- Kate 2018 primary T1-like scan;
- Kate 2022 low-resolution/thick-slice T1-like scan;
- Kate 2024 3DI scan;
- Kate 2024 FFE alternatives used as candidate rescue/reference views.

Required research data:

- SIMON test-retest or repeated-session data available locally;
- SBPR/SRPBS travelling-head or multisite data available locally;
- any additional dataset only if it has clear acquisition metadata and can be
  processed by the same scoring layer.

If raw data are unavailable and only derivatives exist, the dataset can be used
only for a documented secondary analysis, not as proof of raw-input robustness.

### Method Scope

Required method families:

- SynthSeg;
- FastSurfer and FastSurfer Long;
- FreeSurfer 7/8 longitudinal or clinical streams where feasible;
- ReconAny or recon-all-clinical;
- TIGERBx;
- BrainChop;
- SIAM/siamize;
- OpenMAP-T1 or another atlas/parcellation comparator if model access is
  available;
- harmonized preprocessing branch.

Each method must be classified as one of:

- `included_completed`;
- `included_failed_runtime`;
- `included_failed_quality`;
- `candidate_recorded_not_run`;
- `excluded_with_reason`.

## Definition Of Done

The goal is complete only when all criteria below are satisfied by current
repository artifacts, logs, summaries, and reproducible commands.

### 1. Frozen Method Roster

There must be a frozen method roster with source URLs, versions, model weights
or containers, licenses, expected outputs, local command wrappers, and inclusion
or exclusion status.

Evidence required:

- method/source manifests under `experiments/kate_n1_2026`;
- status rows in `data/kate_n1_2026/method_status_matrix.csv`;
- explicit exclusion reasons for methods that cannot be run.

### 2. Reproducible Execution Layer

Every included method must have a reproducible command wrapper or documented
manual blocker. The wrapper must record inputs, outputs, runtime location,
version, and failure mode.

Evidence required:

- shell/Python wrappers under `experiments/kate_n1_2026`;
- small tracked CSV summaries;
- no reliance on unrecorded manual GUI output for final claims.

### 3. Kate N=1 Application Completed

All runnable methods must be applied to the defined Kate scan set, or marked as
failed/blocked with evidence. Outputs must be classified as anatomical candidate,
QC-only, uncertainty-only, or rejected.

Evidence required:

- per-method reports in `docs/kate_n1_2026`;
- summary rows in `method_status_matrix.csv`;
- visual QC overlays for any output considered for visualization.

### 4. Test-Retest Research Benchmark Completed

At least SIMON and SBPR/SRPBS must be evaluated, or one must have a documented
data-access blocker and a justified substitute. The benchmark must include
repeatability metrics, not only single-scan outputs.

Evidence required:

- input manifests for test-retest/travelling-head datasets;
- per-method output summaries;
- scan-to-scan CV, ICC or paired disagreement metrics where applicable;
- failure-rate table by method and dataset.

### 5. TTA Applied To Real Outputs

The TTA label-ensemble evaluator must be populated with real inverse-resampled
label maps for at least SynthSeg and two comparator branches. Comparator
branches should include one anatomical method and one fast/QC-oriented method
where feasible.

Evidence required:

- TTA manifests using `tta_label_ensemble_inputs.schema.csv`;
- per-label CV, hard-vote, vote-fraction, and entropy summaries;
- runtime NIfTI uncertainty maps stored outside git but referenced by manifest.

### 6. Spatial Accuracy And Consensus Layer

Volume agreement alone is insufficient. Promoted methods must be scored in a
documented common space with spatial metrics.

Evidence required:

- registered or subject-template pseudo-GT workflow;
- Dice, Jaccard, HD95, ASSD or surface Dice;
- leave-one-method-out consensus where a method is evaluated against consensus;
- explicit note that consensus is pseudo-ground-truth, not manual truth.

### 7. Uncertainty Validation

The study must test whether TTA uncertainty is informative. Low entropy or low
CV must be compared against test-retest stability and source-vs-consensus
agreement.

Evidence required:

- correlation or stratified analysis linking TTA metrics to test-retest/spatial
  error;
- examples of low-uncertainty failures and high-uncertainty failures;
- final decision rule for using uncertainty maps.

### 8. Visual QC Gates

Any output promoted to `your-brain-mri-visualization` must pass visual QC. Any
known failure mode must have representative overlays or documented inspection.

Evidence required:

- overlay manifest;
- selected PNG/QC references outside git or small tracked thumbnails if safe;
- report stating promote/reject/QC-only per method and scan.

### 9. Final Recommendation

The project must end with a final ranked recommendation, not just a collection
of experiments.

Evidence required:

- final report with method ranking by use case;
- recommended robust pipeline for Kate's data;
- recommended general benchmark pipeline for test-retest datasets;
- documented failure modes and compute requirements;
- clear list of what should and should not be imported into
  `your-brain-mri-visualization`.

## Non-Goals

- Do not claim manual ground truth without manual labels.
- Do not treat TTA consensus as anatomical truth.
- Do not promote a method because it is new, popular, or branded as SOTA.
- Do not average incompatible label ontologies without an explicit mapping.
- Do not use brain-age outputs as segmentation validation.

## Current Completion Status

As of 2026-06-27, the project has a strong Kate n=1 application base, a
pseudo-GT/visual-QC layer, and a reusable TTA evaluator. It is not complete as a
full SOTA test-retest benchmark because real multi-method TTA outputs and
external dataset validation are not yet populated.
