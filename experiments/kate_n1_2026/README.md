# Kate n=1 Brain MRI Segmentation Experiments, 2026

This directory contains reproducible experiment code for the Kate n=1
segmentation/stability study.

Tracked here:

- shell scripts that reproduce the FreeSurfer/FastSurfer/FS8/ReconAny runs;
- Python comparison code for FastSurfer Long symmetry metrics;
- wrappers/manifests for exploratory BrainFM and BrainIAC foundation-model runs;
- wrappers/manifests for BrainChop CLI application runs;
- wrappers/manifests for TIGERBx/OpenMAP-T1 Asian MRI morphometry candidates;
- source manifest for emerging SOTA segmentation candidates such as SIAM;
- small derived CSV summaries in `data/kate_n1_2026`;
- markdown research notes in `docs/kate_n1_2026`.

Not tracked here:

- raw MRI files (`.nii`, `.nii.gz`, DICOM);
- FreeSurfer/FastSurfer subject folders;
- `.mgz/.mgh` volumes;
- long runtime logs;
- Docker layer caches.

The scientific interpretation rule for this experiment is conservative:
stability/reproducibility metrics do not prove anatomical accuracy. Any method
claim must separate processing floor, scanner/protocol effects, and visual QC.

## Local Data Root

The original local data root used for these runs was:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years
```

Scripts use WSL paths and assume the same folder is mounted at:

```text
/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years
```

## Experiment Scripts

| Script | Purpose |
|---|---|
| `run_fs_long_2018_2022.sh` | FreeSurfer 7.4.1 cross/base/long chain for 2018 + 2022. |
| `run_fastsurfer_long_symmetry.sh` | FastSurfer Long v2 rotation-pair experiment. |
| `compare_fastsurfer_long_symmetry.py` | Extracts and compares FastSurfer Long symmetry metrics. |
| `run_fs82_local.sh` | Local WSL fallback run for FreeSurfer 8.2.0 with high swap. |
| `check_fs82_status.sh` | FS8 local run status check. |
| `run_reconany_local.sh` | ReconAny launcher; checks whether `run_recon-any` exists in the selected image/build. |
| `foundation_model_inputs.csv` | Relative-path manifest for BrainFM/BrainIAC candidate inputs. |
| `run_brainiac_features_local.sh` | BrainIAC preprocessing + 768-d embedding extraction launcher. |
| `extract_brainiac_embeddings.py` | BrainIAC embedding extractor supporting `.ckpt` and `.safetensors` weights. |
| `build_brainiac_brainage_inputs.py` | Builds Kate/SIMON manifests for exploratory BrainIAC brain-age application. |
| `build_simon_segmentation_test_retest_manifest.py` | Builds SIMON FreeSurfer8 derivative session/pair manifests for repeatability scaffolding. |
| `brainiac_brainage_infer.py` | Runs local BrainIAC Brain Age Space inference without volume outputs. |
| `summarize_brainiac_brainage.py` | Summarizes exploratory brain-age outputs and SIMON session medians. |
| `run_brainfm_local.sh` | BrainFM inference launcher with disk-space guard. |
| `brainfm_infer_kate.py` | BrainFM wrapper with corrected config paths and feature summaries. |
| `summarize_brainfm_features.py` | Summarizes BrainFM feature-only outputs into compact QC distance tables. |
| `foundation_model_sources.json` | Pinned BrainFM/BrainIAC source, model-card, and license manifest. |
| `sota_segmenter_sources.json` | Pinned SIAM/siamize and emerging SOTA segmentation candidate sources. |
| `brainchop_inputs.csv` | Manifest for T1-like BrainChop candidate inputs. |
| `brainchop_sources.json` | Pinned BrainChop CLI/browser source and local runtime manifest. |
| `run_brainchop_local.sh` | BrainChop 0.2.5 venv setup and batch launcher. |
| `run_brainchop_batch.py` | Timeout-managed BrainChop CLI batch runner. |
| `summarize_brainchop_smoke.py` | Summarizes BrainChop runtime results and compact label stats. |
| `build_tta_uncertainty_evidence.py` | Builds the compact TTA/uncertainty evidence ledger and robust-pipeline v0 report. |
| `tta_label_ensemble_inputs.schema.csv` | Manifest schema for inverse-resampled TTA label maps in a common space. |
| `evaluate_tta_label_ensemble.py` | Scores TTA label ensembles with per-label CV, hard vote, vote fraction, and entropy. |
| `synthseg_2018_rotation_tta_label_ensemble_inputs.csv` | Real 9-angle SynthSeg 2018 rotation-TTA label ensemble manifest. |
| `fastsurfer_long_2018_rotation_pair_label_ensemble_inputs.csv` | Real FastSurfer Long v2 2018 +/-3 degree DKT+ASEG comparator ensemble manifest. |
| `asian_mri_tools_inputs.csv` | Relative-path manifest for TIGERBx/OpenMAP-T1 candidate inputs. |
| `run_tigerbx_local.sh` | TIGERBx brain extraction, ASEG/deep-gray/HLC launcher. |
| `summarize_tigerbx_bmad.py` | Summarizes TIGERBx label volumes, QC logs, and pairwise consistency. |
| `run_openmap_t1_local.sh` | OpenMAP-T1 280-region parcellation launcher; waits for manual model folder. |
| `extract_label_volumes.py` | Generic NIfTI label-map volume extractor. |
| `pseudo_gt_volume_inputs.csv` | Manifest for volume-level pseudo-GT sources. |
| `evaluate_pseudo_gt_volume.py` | Builds volume-level pseudo-GT references and scores sources against them. |
| `pseudo_gt_spatial_inputs.csv` | Manifest for header-affine spatial pseudo-GT pilot sources. |
| `evaluate_pseudo_gt_spatial_header_affine.py` | Resamples 2024 label maps to one grid, builds hard-vote pseudo-GT, and scores Dice/Jaccard. |
| `pseudo_gt_registered_inputs.csv` | Manifest linking 2024 images and labels for registered spatial pseudo-GT. |
| `evaluate_pseudo_gt_spatial_registered.py` | Affine-registers 2024 images, resamples labels, builds pseudo-GT, and scores Dice/Jaccard/HD95/ASSD. |
| `generate_visual_qc_overlays.py` | Generates native TIGERBx and registered pseudo-GT visual QC overlays. |
| `asian_mri_tools_sources.json` | Pinned source/review manifest for Asian MRI morphometry candidates. |

## Current Result State

Completed:

- FS7.4.1 longitudinal 2018 + 2022.
- SynthSeg rotation/TTA summary from prior pipeline.
- ordinary FastSurfer rotation floor summary.
- FastSurfer Long rotation-pair experiment.
- TIGERBx `bmadq` first-pass application run on 2018, 2022, 2024 3DI, and two
  2024 FFE alternatives.
- BrainFM feature-only application run and compact feature-distance QC summaries.
- BrainIAC Brain Age Space exploratory local run on Kate and SIMON derivatives.
  The result is an OOD/preprocessing-sensitivity finding, not a valid adult
  brain-age estimate.
- BrainChop 0.2.5 CLI installed and model listing verified; `tissue_fast`
  completed on 2024 3DI, FFE 401, and FFE 601 as a quick tissue-QC branch.
  `mindgrab` timed out at 5 minutes per scan, and anatomical subcortical/atlas
  outputs are not promoted.
- 2024 visual QC overlays for TIGERBx native outputs and registered
  SynthSeg/TIGERBx pseudo-GT sources.
- TTA/uncertainty evidence ledger v0 combining SynthSeg TTA, FastSurfer
  rotation, registered pseudo-GT, and BrainChop tissue-QC evidence.
- TTA label-ensemble execution schema v0 with a self-tested evaluator for
  inverse-resampled labels, hard-vote consensus, vote fraction, entropy, and
  per-label CV.
- First real populated TTA label ensemble: SynthSeg 2018 9-angle rotation sweep,
  with hard vote, vote fraction, entropy maps, and tracked CSV summaries.
- First comparator populated through the same evaluator: FastSurfer Long v2
  2018 +/-3 degree DKT+ASEG pair.
- SIMON derivative-level segmentation test-retest manifest: 69/73 sessions have
  FreeSurfer8 DKT+ASEG labels and 66/72 consecutive pairs are usable as a
  secondary repeatability scaffold.
- SIAM / Segment It All Model recorded as an emerging benchmark candidate with
  Python SIAM and `siamize` source routes; not yet run.

Running or pending:

- FS8.2 local WSL fallback run; 2018/2022 completed and 2024 3DI probe stopped
  after extreme topology-correction runtime.
- ReconAny/recon-all-clinical branch.
- harmonized preprocessing branch.
- BrainIAC foundation-model branch; wrapper is ready, but local model weights
  are not yet present.
- BrainChop subcortical-mini runtime test or GPU/WebGPU execution for heavier
  anatomical models.
- SIAM/siamize guarded smoke run on one 2024 FFE candidate after RAM/GPU
  feasibility check and label ontology mapping plan.
- Apply the TTA label-ensemble evaluator to test-retest data and validate
  whether TTA uncertainty predicts repeatability error.
- Locate or generate raw/harmonized SIMON method outputs for SynthSeg and one
  comparator; the current SIMON scaffold is derivative-level only.
- OpenMAP-T1 Asian morphometry branch; wrapper is ready, but the local model
  folder is not yet present.
