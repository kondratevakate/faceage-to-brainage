# Vendor Model Versions

Pinned versions used for the MIDL 2026 paper results.
Update this file whenever you change a model version.

---

## FaceAge

- **Repo**: https://github.com/AIM-Harvard/FaceAge
- **Commit**: *(run `git -C vendor/FaceAge rev-parse HEAD` after cloning)*
- **Weights**: `FaceAge_weights.pt`, `age_regressor.pt`
  - Download from Google Drive link in `vendor/FaceAge/README.md`
  - Place in `vendor/FaceAge/models/` (gitignored)
- **Paper**: Bontempi et al., *The Lancet Digital Health*, 2025
  DOI: 10.1016/S2589-7500(25)00045-4

```bash
git clone https://github.com/AIM-Harvard/FaceAge vendor/FaceAge
```

---

## SynthBA

- **Install**: `pip install synthba`
- **Version used**: `0.2.0`
- **No manual weight download needed** — weights bundled with the package
- **Paper**: Lemaître et al., 2022

```bash
pip install synthba==0.2.0
```

---

## SynthStrip

- **Part of FreeSurfer** (standalone binary also available)
- **Version used**: FreeSurfer 7.4+ or standalone SynthStrip
- **Paper**: Hoopes et al., *NeuroImage*, 2022
  DOI: 10.1016/j.neuroimage.2022.119474

Standalone install (used in Colab notebooks):
```bash
wget https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/synthstrip.1.6.sif
```

---

## SFCN

- **Repo**: https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
- **Commit**: `4e56dc1522ac89caadc5c6fc779799ee58c23adc`
- **Weights**: `run_20190719_00_epoch_best_mae.p`
  - Download from repo releases / Figshare (see upstream README)
  - Place in `vendor/SFCN/brain_age/` (gitignored)
- **Status**: 🔄 age-bin decoding under validation
- **Paper**: Peng et al., *Medical Image Analysis*, 2021
  DOI: 10.1016/j.media.2020.101871

```bash
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN
cd vendor/SFCN && git checkout 4e56dc1522ac89caadc5c6fc779799ee58c23adc
```

---

## MIDIBrainAge

- **Repo**: cloned into `vendor/MIDIBrainAge`
- **Commit**: *(run `git -C vendor/MIDIBrainAge rev-parse HEAD`)*
- **Status**: 🔄 in progress (notebook 09)

---

## BrainIAC

- **Status**: 🔄 in progress (notebook 10)
- **Paper**: Tak et al., 2026
