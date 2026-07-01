# BrainFM and BrainIAC Pipeline Plan for Kate n=1

Date: 2026-06-17

## Scope

This branch adds reproducible launchers for two recent brain MRI foundation-model
families:

- BrainFM: modality-agnostic, multi-task brain imaging model for synthesis,
  anatomy-related outputs, bias-field estimation, registration-like outputs, and
  feature extraction.
- BrainIAC: structural brain MRI foundation encoder trained with contrastive
  self-supervision and exposed as a 768-dimensional ViT feature extractor, with
  downstream examples for brain age, MCI, sequence classification, stroke timing,
  survival, IDH, and tumor segmentation.

The code added here does not track raw MRI, model weights, or generated NIfTI
outputs. It tracks only input manifests, wrappers, source pins, and small
tabular outputs after the methods are run.

## Scientific Position

These methods answer a different question from FreeSurfer, FastSurfer, SynthSeg,
and ReconAny. They are not a replacement ground truth for cortical/subcortical
segmentation. In this n=1 study they should be used as exploratory stress tests:

1. Do embeddings/features remain stable across the 2018, 2022, and 2024 scans?
2. Do feature shifts mostly track acquisition/protocol differences already seen
   in segmentation methods?
3. Do generated outputs or saliency maps highlight failure modes such as thick
   slices, unusual contrast, skull stripping errors, registration failures, or
   left-right/orientation sensitivity?

Any claim that a foundation model is "better" must be supported by visual QC and
region-level comparison against already computed segmentation outputs. Stable
embeddings alone do not prove anatomical accuracy.

## Local Execution Status

Current status:

- BrainFM: official `brainfm_pretrained.pth` is present locally, SHA256
  verified, and the feature-only CPU run completed with `WRITE_VOLUMES=0`.
  Compact feature-distance QC summaries are tracked in `data/kate_n1_2026/`.
- BrainIAC: still blocked by missing `BrainIAC.ckpt` or `backbone.safetensors`.

The wrappers intentionally fail fast when weights are missing. This avoids
starting long runs that produce empty or non-interpretable outputs. Disk space is
also constrained on `D:`; BrainFM refuses to run by default below 80 GB free
unless `ALLOW_LOW_DISK=1` is set. For the completed first BrainFM run,
`ALLOW_LOW_DISK=1` was acceptable only because dense volume writing was disabled.

## Added Pipelines

Environment setup is intentionally external to the repository because BrainFM
and BrainIAC have different dependency pins. Use separate environments:

```bash
# BrainIAC
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r /mnt/d/projects/02_academia/_external/BrainIAC/requirements.txt
pip install safetensors

# BrainFM
conda create -n brainfm python=3.11
conda activate brainfm
pip install -r /mnt/d/projects/02_academia/_external/BrainFM/requirements.txt
```

BrainIAC:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
bash experiments/kate_n1_2026/run_brainiac_features_local.sh
```

Default output:

```text
/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/foundation_models/brainiac/features/brainiac_embeddings.csv
```

BrainFM:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
export PYTHON=/mnt/d/projects/02_academia/_external/.venvs/brainfm_py311/bin/python
export WRITE_VOLUMES=0
export DEVICE=cpu
export ALLOW_LOW_DISK=1
bash experiments/kate_n1_2026/run_brainfm_local.sh
```

Default output:

```text
/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/foundation_models/brainfm/brainfm_feature_summaries.csv
```

Feature QC summary:

```bash
python3 experiments/kate_n1_2026/summarize_brainfm_features.py \
  --features-csv /mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years/reprocessed_2026/foundation_models/brainfm/brainfm_feature_summaries.csv \
  --scan-summary-csv data/kate_n1_2026/brainfm_feature_scan_summary.csv \
  --pairwise-csv data/kate_n1_2026/brainfm_feature_pairwise_distances.csv \
  --contrast-summary-csv data/kate_n1_2026/brainfm_feature_contrast_summary.csv
```

## Planned Measurements After Successful Runs

- BrainFM within-method feature stability: cosine distance and Euclidean distance between
  2018, 2022, and 2024 embeddings.
- BrainFM protocol sensitivity: compare primary T1-like inputs against secondary FLAIR/T2
  inputs where available.
- BrainIAC feature extraction after checkpoint acquisition.
- Cross-method consistency: relate foundation-model feature shifts to existing
  SynthSeg, FastSurfer Long, and FS7/FS8 longitudinal CV/failure notes.
- Failure notes: record preprocessing failures, skull-strip artifacts,
  registration errors, missing checkpoints, GPU/CPU limitations, and low-disk
  aborts.

## Source Pins

See `experiments/kate_n1_2026/foundation_model_sources.json` for pinned external
repositories, commits, model cards, and license notes.
