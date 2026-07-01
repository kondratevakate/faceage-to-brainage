# BrainIAC brain-age application branch

This is an exploratory local application of the public BrainIAC Brain Age
Prediction Space, not a clinical or biological-age claim.

## Source and setup

- Source Space: `https://huggingface.co/spaces/Divytak/BrainIAC-Brainage-V0`
- Local Space checkout: `/mnt/d/projects/02_academia/_external/BrainIAC-Brainage-V0`
- Checkpoint: `/mnt/d/projects/02_academia/_external/BrainIAC-Brainage-V0/src/BrainIAC/checkpoints/brainage_model_latest.pt`
- Checkpoint SHA256: `0bbc78dc2237f08655c6b8cfb6b56d065e43c17d93d6e826d455a331189f3d8f`
- Venv: `/home/kate/.venvs/brainiac_brainage_py311`
- Inference preprocessing used here: load MRI volume, resize to `128x128x128`,
  scale intensity to `0..1`, and run the model on CPU. No volume outputs were
  written.

The Space app divides raw model output by 12 before displaying years. Its sample
`subpixar009_T1w` has label `42.0` and predicts raw output about `43.15`, so the
tables retain both `raw_model_output` and `predicted_age_years_if_months`. The
raw-as-years column is kept only as a diagnostic because it conflicts with the
Space app and local BrainIAC downstream documentation.

## Inputs

Kate:

- `5` direct T1-like NIfTI inputs with no preprocessing beyond model resize/scale.
- `5` existing TIGERBx `tbet` brain-extracted inputs. These are not a fresh
  BrainIAC full-preprocessing run and are not MNI-registered by this branch.

SIMON:

- `94` existing FastSurfer `*_orig.mgz` derivatives, grouped into `73` sessions.
- Local raw SIMON no-preprocessing branch is blocked: `/mnt/mydisk` is not
  mounted and `D:/data` contains derivatives, not raw BIDS/NIfTI sourcedata.

## Results

All Kate and SIMON inputs completed without runtime failure.

Kate direct raw T1-like branch:

- Raw model output range: `43.282394` to `265.950623`; median `64.222618`.
- App-style years (`raw/12`) range: `3.606866` to `22.162552`; median `5.351885`.

Kate existing TIGERBx `tbet` branch:

- Raw model output range: `44.724785` to `173.522156`; median `91.792534`.
- App-style years (`raw/12`) range: `3.727065` to `14.460180`; median `7.649378`.

SIMON FastSurfer `orig.mgz` branch:

- `94/94` runs completed.
- Chronological age range from `SIMON_pheno.csv`: `29.69` to `46.41` years.
- Raw model output range: `41.551476` to `75.216232`; median `49.669725`.
- App-style years (`raw/12`) range: `3.462623` to `6.268019`; median `4.139144`.

## Interpretation

These outputs are not valid adult brain-age estimates for Kate or SIMON. Under
the Space app interpretation (`raw/12`), adult SIMON scans are predicted as
approximately preschool-age, and Kate scan predictions are strongly
protocol-dependent. Treat this as evidence of domain/preprocessing sensitivity
and unit ambiguity, not as evidence that the subject is biologically younger or
older.

The model description also says it was trained with registration to MNI, N4 bias
correction, histogram equalization, and skull stripping. This branch did not run
that full preprocessing pipeline for all data because it would create additional
derived volumes and the current disk budget on `D:` is low. Existing conformed or
brain-extracted derivatives are therefore only sensitivity probes.

## Outputs

- `data/kate_n1_2026/brainiac_brainage_predictions_kate.csv`
- `data/kate_n1_2026/brainiac_brainage_predictions_simon_fastsurfer_orig.csv`
- `data/kate_n1_2026/brainiac_brainage_branch_summary.csv`
- `data/kate_n1_2026/brainiac_brainage_simon_session_summary.csv`
- `data/kate_n1_2026/brainiac_brainage_kate_scan_summary.csv`
- `data/kate_n1_2026/brainiac_brainage_blockers.csv`

## Next step

For an actual adult brain-age claim, use an adult-calibrated model with
documented units and preprocessing, then validate on SIMON chronological ages
before applying to Kate n=1.
