# Brain-age model search report

Date: 2026-06-23

Scope: find models with an explicit brain-age head or age regressor for Kate n=1 and the SIMON longitudinal dataset. This is an application and QC branch, not a validation claim.

## Main finding

BrainFM is not currently a valid age-prediction source in this local setup. The public Hugging Face repository lists `assets/brainfm_pretrained.pth` and age configs, but I did not find a separate public age-head checkpoint. The local `brainfm_pretrained.pth` state dict also lacks age-head keys. Therefore BrainFM should stay in the foundation-model feature/QC branch until an official fine-tuned age checkpoint is found.

The strongest age-head candidates found are:

1. `MIDIBrainAge_T1_ensemble`: already local with real `.pt` regression weights in `D:\projects\02_academia\faceage-to-brainage\vendor\MIDIBrainAge`. This is the best immediate next run because the weights are present and the upstream script is explicit about T1 handling.
2. `BrainAgeNeXt`: public five-model MedNeXt ensemble on Hugging Face with `BrainAge_1.pth` through `BrainAge_5.pth`. It requires external preprocessing before inference.
3. `SynthBA`: package-level model with bundled weights and a claimed robust raw multi-contrast preprocessing path. Good as a raw/no-harmonization robustness comparator.
4. `Westman brainage-prediction-mri`: raw T1-oriented tool with release weights, but it depends on FSL/nipype registration.
5. `SFCN UKBiobank`: official age-bin model, but the public bins are 42.5 to 81.5 years. This makes it a poor standalone estimator for younger adult scans and a risky fit for SIMON sessions below 42 years.

The full inventory is in `data/kate_n1_2026/brain_age_model_inventory.csv`.

## Preprocessing discipline

Do not compare "raw" and "preprocessed" as if preprocessing is one universal operation. Each model has its own training-domain preprocessing.

`MIDIBrainAge` T1 official path runs HD-BET skull stripping, N4/ANTs affine registration to MNI, RAS reorientation, 1.4 mm spacing, crop/pad to 130 cubed, z-score normalization, and intensity clamp. It has no upstream raw-T1 T1 model path.

`BrainAgeNeXt` expects skull-stripped, N4-corrected, affine-registered FSL MNI152 T1 inputs. Its script then applies MONAI spacing, foreground crop, padding, center crop to 160x192x160, and TorchIO foreground z-normalization.

`SynthBA` should be run on raw inputs if used, because its value proposition is internal robust preprocessing and domain randomization.

`BrainIAC Brainage` has already been run, but its adult outputs were implausible on SIMON. Keep it as an out-of-domain/protocol-sensitivity result only.

## Comparison rule

Use SIMON as the calibration gate before saying anything about Kate's age. A model should not be promoted to a Kate age estimate unless it passes basic SIMON sanity checks:

- units are documented and predictions are in plausible adult years;
- successful cases have a positive relationship with chronological age;
- MAE and bias are not pathological for the 29.7 to 46.4 year SIMON range;
- repeated runs/sessions do not show preprocessing artifacts larger than the biological signal;
- failures and preprocessing branches are recorded per scan.

For Kate n=1, report each model as an exploratory estimate with preprocessing provenance. Do not average models unless all have passed SIMON sanity checks and the averaging rule is predeclared.

## Next run order

1. Smoke-test `MIDIBrainAge_T1_ensemble` in an isolated WSL venv on 1 Kate input plus 3 SIMON inputs. If preprocessing and units look sane, run the full Kate/SIMON comparison.
2. Download `BrainAgeNeXt` weights and code into `_external`, install MedNeXt in a separate WSL venv, and run only on the documented preprocessed branch. Record raw model output separately from the script's chronological-age bias-corrected output.
3. Install/run `SynthBA` as the raw robustness comparator.
4. Add `Westman` and `ANTsPyNet brain_age` only if the first three leave ambiguity or fail.

## Sources checked

- BrainFM HF model: https://huggingface.co/peirong26/BrainFM
- MIDIBrainAge: https://github.com/MIDIconsortium/BrainAge
- BrainAgeNeXt: https://github.com/FrancescoLR/BrainAgeNeXt and https://huggingface.co/FrancescoLR/BrainAgeNeXt
- SynthBA: https://github.com/LemuelPuglisi/SynthBA
- Westman brainage-prediction-mri: https://github.com/westman-neuroimaging-group/brainage-prediction-mri
- SFCN UKBiobank_deep_pretrain: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
- ANTsPyNet brain_age documentation: https://antsx.github.io/ANTsPyNet/docs/build/html/utilities.html#antspynet.utilities.brain_age
- DeepBrainNet: https://github.com/vishnubashyam/DeepBrainNet
- BrainAge_DenseNet HF model: https://huggingface.co/SisInfLab-AIBio/BrainAge_DenseNet
- brainageR: https://github.com/james-cole/brainageR
- MASILab BrainAGE: https://github.com/MASILab/BrainAGE
