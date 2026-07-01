# BrainFM feature-only application branch

Date: 2026-06-21

## Scope

BrainFM was run locally on the Kate n=1 foundation-model manifest as an application/QC branch, not as a validation claim for segmentation or morphometry. The run used 11 heterogeneous MRI candidates from 2018, 2022, and 2024.

## Checkpoint and environment

- BrainFM repo: `/mnt/d/projects/02_academia/_external/BrainFM`
- Official model card: `https://huggingface.co/peirong26/BrainFM`
- Checkpoint: `/mnt/d/projects/02_academia/_external/BrainFM/ckp/brainfm_pretrained.pth`
- Checkpoint SHA256: `227263e184004f044c6f62d4b436f4ffb87ecf815eb16923e7a7b1de1c53fec8`
- Runtime: isolated WSL venv at `/mnt/d/projects/02_academia/_external/.venvs/brainfm_py311`
- Device: CPU
- Volumes: disabled with `WRITE_VOLUMES=0`

`ALLOW_LOW_DISK=1` was used only because this first pass did not write dense volume outputs.

## Output check

Output root:

`D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\foundation_models\brainfm`

Checked outputs:

- `brainfm_inputs_resolved.csv`: 11 resolved input rows
- `brainfm_feature_summaries.csv`: 11 feature-summary rows, 8072 columns
- `brainfm_volume_outputs.csv`: header only; no NIfTI/MGZ/NPY/NPZ outputs were written
- `brainfm_metadata.json`: `write_volumes=false`, `device=cpu`, `n_images=11`

## Interpretation

BrainFM feature summaries are foundation-model embeddings/features for QC, robustness, and protocol-sensitivity review. They do not demonstrate segmentation accuracy, morphometric accuracy, biological change, or superiority over other methods. Any anatomical or longitudinal claim still requires visual QC, cross-method consistency checks, and explicit validation against the relevant target outcome.

## Follow-up QC Summary

Compact scan-level, pairwise-distance, and contrast-level summaries were generated with `experiments/kate_n1_2026/summarize_brainfm_features.py` and are interpreted in `docs/kate_n1_2026/brainfm_feature_qc_summary.md`.

## Integration notes

The first local attempts exposed setup issues, not method-performance results: a shell quoting bug in the disk-space guard, missing Python dependencies in system WSL Python, and an upstream BrainFM import-time relative path assumption for `files/gca.mgz`. The wrapper and run script were adjusted, and inference completed through the isolated Python 3.11 CPU environment.
