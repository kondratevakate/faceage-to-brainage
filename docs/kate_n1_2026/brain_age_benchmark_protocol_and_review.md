# Brain-age benchmark protocol and scoping review

Date: 2026-06-26

## Objective

Build a reproducible brain-age benchmark across:

- Kate n=1 local MRI inputs;
- SIMON longitudinal/local derivative inputs with chronological ages;
- SRPBS Traveling Subjects with participant ages and multi-site repeated scans.

This is an application and QC benchmark. A model output does not validate segmentation or morphometry, and a single Kate prediction is not a biological-age claim. Kate estimates can be reported only after the model passes labeled sanity checks and robustness checks on SIMON/SRPBS.

## Review Question

Which publicly runnable brain-age models with explicit age heads are suitable for local benchmarking on heterogeneous clinical/research MRI inputs, and what preprocessing/data branches are needed to evaluate them reproducibly on Kate, SIMON, and SRPBS?

## Search Strategy

Search date: 2026-06-26.

Sources already checked are in `data/kate_n1_2026/brain_age_benchmark_sources.csv`. The first pass prioritized primary sources: official GitHub repositories, Hugging Face model/space pages, tool documentation, and openly accessible papers. Search terms included combinations of `brain age prediction MRI`, `MIDIBrainAge`, `BrainAgeNeXt`, `SynthBA`, `brainage-prediction-mri`, `ANTsPyNet brain_age`, `SFCN UKBiobank brain age`, and `Brain Age Standardized Evaluation`.

Inclusion criteria:

- explicit brain-age head/regressor/classifier-to-age output;
- public code and either public weights or auto-downloadable model assets;
- feasible local execution in isolated WSL environments;
- clear enough preprocessing to define a reproducible branch.

Exclusion/defer criteria:

- feature/foundation models without public age-head weights;
- models requiring unavailable private data or opaque preprocessing;
- segmentation/morphometry-dependent pipelines unless used as secondary comparators;
- age ranges incompatible with Kate/SIMON/SRPBS without an explicit limitation label.

## Evidence Framework

The benchmark follows the BASE logic: evaluate not only accuracy but also robustness, reproducibility, and consistency across sites, test-retest, and longitudinal settings. BASE specifically argues that comparing brain-age models requires standardized data, protocols, performance metrics, and statistical models for accuracy and robustness.

For this project:

- SIMON is the primary longitudinal age-calibration branch.
- SRPBS is the primary site/test-retest robustness branch.
- Kate n=1 is the application branch and must not be used as the validation set.

Claim levels:

- `technical_smoke_not_age_claim`: model runs and emits adult-year output, but evidence is too narrow.
- `small_sanity_not_validation`: labeled subset looks plausible but is too small/narrow.
- `robustness_qc`: useful for protocol/site sensitivity, not age accuracy.
- `invalid_for_age_claim`: units/domain/preprocessing behavior failed basic adult plausibility.
- `candidate_for_kate_reporting`: only after labeled SIMON/SRPBS gates and at least one independent comparator pass.

## Model Taxonomy

### Current P1 branch: MIDIBrainAge

Source: https://github.com/MIDIconsortium/BrainAge

MIDIBrainAge is currently the strongest local branch because the T1 ensemble already runs in `/home/kate/.venvs/midi_brainage_py311`. The official T1 path uses HD-BET skull stripping plus affine MNI registration and model-specific resampling/cropping. It should be treated as a preprocessed-model branch even when the input file is a raw NIfTI, because the tool performs substantial internal preprocessing.

Current evidence:

- Kate 2018 smoke: predicted age 29.8 years, no age claim.
- SIMON-3 smoke: MAE 4.45 years, n=3, not validation.
- SIMON stratified-12 FastSurfer `orig.mgz`: 12/12 completed, MAE 8.99 years, bias -7.66 years, Pearson r -0.43, and slope -0.35. This derivative branch fails the labeled sanity gate for Kate-age reporting.
- SRPBS siteATTd1: 9/9 successful, age range 24-32, MAE 1.74 years, bias +1.53 years, r 0.85, not validation because n=9 and one site.

### P1/P2 comparator: BrainAgeNeXt

Source: https://github.com/FrancescoLR/BrainAgeNeXt

BrainAgeNeXt is an important independent T1 comparator. Its documented branch requires skull stripping, N4 bias correction, and affine registration to FSL MNI152 before inference. It should not be run on arbitrary raw images without reproducing that preprocessing.

Next work: download HF assets, create isolated venv, implement preprocessing manifest, and run Kate + SIMON/SRPBS gates.

### P2 raw robustness comparator: SynthBA

Sources: https://github.com/LemuelPuglisi/SynthBA and https://arxiv.org/html/2406.00365v2

SynthBA is the main raw/heterogeneous MRI comparator because its purpose is robustness across contrast and resolution using domain randomization and internal preprocessing. It is the best candidate for a "no external preprocessing" branch.

Next work: install in a separate venv, run a tiny Kate/SIMON/SRPBS raw smoke, then scale if units and failures are sane.

### P2/P3 raw T1 comparator: Westman brainage-prediction-mri

Source: https://github.com/westman-neuroimaging-group/brainage-prediction-mri

This tool accepts unprocessed T1 NIfTI and predicts age after minimal preprocessing, but the repository warns that validation across protocols, scanners, and populations is limited. It needs FSL/nipype discipline and visual registration QC.

### P2/P3 independent implementation: ANTsPyNet brain_age

Source: https://antsx.github.io/ANTsPyNet/docs/build/html/utilities.html

ANTsPyNet exposes a DeepBrainNet-style `brain_age` utility. Documentation states that the training preprocessing included N4 bias correction, brain extraction, and affine registration to MNI, and that internal preprocessing can be used for raw T1. This is useful as an independent implementation if TensorFlow/ANTs dependencies are acceptable.

### P3 limited-age comparator: SFCN UKBiobank

Source: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

SFCN has public UKB pretrained weights and documented input shape, but its age-bin/domain assumptions make it risky for younger adult scans. It should not be first-line for Kate/SIMON/SRPBS unless the preprocessing and age decoding limitations are explicit.

## Data Branches

### Kate n=1

Current manifest:

- `experiments/kate_n1_2026/midi_brainage_kate_t1_like_inputs.csv`

Rows: 5 T1-like local NIfTI inputs from 2018, 2022, and 2024. No chronological-age column is currently encoded in the manifest; Kate predictions remain application outputs, not validation.

### SIMON

Current manifests:

- `experiments/kate_n1_2026/midi_brainage_simon_stratified12_inputs.csv`
- `experiments/kate_n1_2026/midi_brainage_simon_session_first_inputs.csv`
- `experiments/kate_n1_2026/midi_brainage_simon_all_orig_inputs.csv`

The visible local SIMON branch uses FastSurfer `orig.mgz` derivatives, not raw BIDS/NIfTI. It is still useful for labeled age sanity checks, but raw SIMON remains a separate data-availability problem.

### SRPBS Traveling Subjects

Current manifests:

- `experiments/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_siteATTd1_inputs.csv`
- `experiments/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_inputs.csv`

The full visible FastSurfer branch has 143 `orig.mgz` inputs: 9 subjects across 16 sites with one missing point. The raw source archive is `D:\data\SRPBS_TS.tar.gz`; only metadata has been extracted so far. Raw T1 extraction should target only `SRPBS_TS/sourcedata/sub-*/ses-site*/anat/*_T1w.nii.gz`.

## Metrics

Labeled age metrics:

- n, success/failure count;
- MAE, median absolute error, RMSE;
- mean signed error/bias and SD;
- Pearson r and age-prediction slope/intercept;
- bootstrap confidence intervals for SIMON full/session-first if n is sufficient.

Robustness metrics:

- within-subject across-site SD and mean absolute deviation for SRPBS;
- site fixed-effect summaries;
- ICC or linear mixed-effects model when the full SRPBS matrix is available;
- paired raw-vs-derivative differences for SRPBS and any available raw SIMON branch.

Sex/moderator metrics:

- predicted-sex accuracy or calibration only where known labels are available;
- predicted-sex probability and sex-class flips under preprocessing,
  resampling, rotation, skull-stripping, and site perturbations;
- brain-age residual summaries by sex, age bin, TIV, scanner/site, and
  preprocessing branch;
- age-by-sex and feature-by-sex interaction terms for population datasets.

These metrics are QC, fairness, shortcut-learning, and biological-moderator
evidence. They are not a "male/female brain health" score. The broader
multimodal feature palette is defined in
`docs/kate_n1_2026/multimodal_health_feature_palette.md` and
`docs/kate_n1_2026/multimodal_feature_possibility_map.md`.

Kate reporting:

- per-scan prediction and preprocessing provenance;
- no model averaging until at least two independent models pass labeled gates;
- do not promote 2024 3DI if preprocessing failures or cross-protocol instability dominate.

## Current Results

Machine-readable current results are in `data/kate_n1_2026/brain_age_benchmark_results_index.csv`.

The only currently plausible adult-year branch is MIDIBrainAge. BrainIAC is retained as a failure-mode/protocol-sensitivity branch because adult labeled scans produced implausible age units. BrainFM remains feature-only because no public age-head checkpoint was found.

## Reproducible Commands

Build manifests:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
python3 experiments/kate_n1_2026/build_midi_brainage_kate_inputs.py \
  --output-csv experiments/kate_n1_2026/midi_brainage_kate_t1_like_inputs.csv
python3 experiments/kate_n1_2026/build_midi_brainage_simon_inputs.py \
  --mode stratified --stratified-count 12 \
  --output-csv experiments/kate_n1_2026/midi_brainage_simon_stratified12_inputs.csv
python3 experiments/kate_n1_2026/build_midi_brainage_simon_inputs.py \
  --mode session-first \
  --output-csv experiments/kate_n1_2026/midi_brainage_simon_session_first_inputs.csv
python3 experiments/kate_n1_2026/build_midi_brainage_srpbs_inputs.py \
  --source fastsurfer-orig --site siteATTd1 \
  --output-csv experiments/kate_n1_2026/midi_brainage_srpbs_travelling_fastsurfer_orig_siteATTd1_inputs.csv
```

Run MIDIBrainAge batch:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_midi_brainage_simon_stratified12.sh
```

Equivalent explicit commands:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
/home/kate/.venvs/midi_brainage_py311/bin/python \
  experiments/kate_n1_2026/run_midi_brainage_batch.py \
  --manifest experiments/kate_n1_2026/midi_brainage_simon_stratified12_inputs.csv \
  --output-csv data/kate_n1_2026/midi_brainage_simon_stratified12_predictions.csv \
  --project-prefix midi_simon_strat12 \
  --return-metrics --resume
/home/kate/.venvs/midi_brainage_py311/bin/python \
  experiments/kate_n1_2026/summarize_midi_brainage_results.py \
  --predictions-csv data/kate_n1_2026/midi_brainage_simon_stratified12_predictions.csv \
  --output-csv data/kate_n1_2026/midi_brainage_simon_stratified12_summary.csv \
  --summary-id midi_simon_stratified12 \
  --group-cols dataset branch \
  --claim-level small_labeled_sanity_not_validation \
  --interpretation "SIMON stratified-12 labeled sanity gate on existing FastSurfer orig.mgz derivatives; not a Kate biological-age validation claim."
```

Do not commit raw MRI, extracted NIfTI, model weights, checkpoints, logs, or temporary MIDIBrainAge project directories.

## Next Experiments

1. Do not promote the MIDIBrainAge FastSurfer `orig.mgz` SIMON branch for Kate-age reporting; the stratified-12 sanity gate failed.
2. Run MIDIBrainAge on raw/documented T1 inputs where available and keep the FastSurfer-derivative branch only as failure-mode evidence.
3. Add BrainAgeNeXt as the next independent preprocessed T1 comparator.
4. Add SynthBA as the first raw/heterogeneous MRI comparator.
5. Run full SRPBS FastSurfer-orig 143 matrix only as a long CPU robustness batch or on faster hardware; evaluate within-subject site spread.
6. Extract a small SRPBS raw T1 subset from `SRPBS_TS.tar.gz` and run raw-vs-FastSurfer paired gate.

## Scientific Guardrails

Brain-age features or predictions are not segmentation accuracy metrics. A good SRPBS/SIMON brain-age result does not validate FreeSurfer, FastSurfer, SynthSeg, TIGERBx, BrainFM embeddings, or morphometry. It only supports the specific age-head branch under the specific preprocessing and data distribution tested here.
