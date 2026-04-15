# Brain-Age Reproduction Note

This repository now starts brain-age reproduction with `SFCN` first, aiming for a usable project baseline rather than a full paper-metric replication.

## Current first milestone

- Model: `SFCN` (`Peng et al. 2021`)
- Datasets:
  - `IXI`: `561` age-clean T1 scans
  - `SIMON`: `99` age-clean T1 scans across `73` sessions
- Runtime target: `Colab GPU`
- Next model after SFCN: `SynthBA`

## What is already implemented in the repo

- Config-driven SFCN runtime setup:
  - tracked example config: `config/brain_age_runtime.example.json`
  - ignored local config: `config/local/brain_age_runtime.json`
- Explicit SFCN preprocessing helpers in `src/brain_age.py`:
  - `run_synthstrip(...)`
  - `prepare_sfcn_input(...)`
  - `predict_sfcn(...)`
  - configurable age-bin decoding
- Config-driven batch runner:
  - `scripts/batch_sfcn.py`

## Required environment before first run

You still need the following in the runtime where inference will actually run:

1. Python with `torch`, `pandas`, `nibabel`, and repo requirements
2. Official SFCN repo cloned into `vendor/SFCN`
3. Official pretrained SFCN weight:
   - `vendor/SFCN/brain_age/run_20190719_00_epoch_best_mae.p`
4. FreeSurfer / SynthStrip on PATH:
   - `mri_synthstrip`
5. CUDA-visible GPU for the preferred runtime

## Recommended Colab bring-up

1. Open a GPU-backed Colab runtime and attach VS Code to it.
2. Clone this repo.
3. Install repo requirements.
4. Clone the official SFCN repo into `vendor/SFCN`.
5. Download the official pretrained weight into `vendor/SFCN/brain_age/`.
6. Install FreeSurfer or otherwise make `mri_synthstrip` available.
7. Copy `config/brain_age_runtime.example.json` to `config/local/brain_age_runtime.json`.
8. Fill in local paths and device.

## First run sequence

Smoke test:

```bash
python scripts/batch_sfcn.py --dataset ixi --limit 3
python scripts/batch_sfcn.py --dataset simon --limit 3
```

Pilot:

```bash
python scripts/batch_sfcn.py --dataset ixi --limit 25
python scripts/batch_sfcn.py --dataset simon --limit 10
```

Full batch:

```bash
python scripts/batch_sfcn.py --dataset ixi
python scripts/batch_sfcn.py --dataset simon
```

## Expected output schema

The batch runner writes per-dataset CSVs with these core fields:

- `dataset`
- `subject_id`
- `session_id`
- `scan_id`
- `chron_age`
- `model_name`
- `predicted_age`
- `brain_age_gap`
- `input_path`
- `preproc_path`
- `status`
- `error`

This schema is intentionally stable so `SynthBA` can be added later without changing downstream analysis.

## Important caution

The current default age-bin settings in the config are still treated as runtime-configurable rather than permanently fixed in code. Before the first large SFCN batch, confirm the official inference decoding assumption against the official SFCN repo/notebook in the actual runtime.
