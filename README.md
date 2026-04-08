# faceage-to-brainage

**Does the face in your MRI scan know how old your brain is?**

A proof-of-concept research repository on paired face-age and brain-age estimation from the same MRI scan.

Current active implementation status:
- face rendering and FaceAge wrappers are in place
- brain-age reproduction starts with `SFCN`
- cleaned T1 manifests are available for `IXI` and `SIMON`
- the first GPU workflow is `notebooks/05_sfcn_colab_bootstrap.ipynb`

---

## Research Question

Standard brain-age models predict age from brain parenchyma: grey and white matter volumes, cortical thickness, and related structural signals. But every non-defaced T1 MRI also contains the full 3D morphology of the face: subcutaneous fat distribution, orbital recession, facial bone structure, and soft-tissue shape.

This project asks: if you extract two age estimates from the same T1 scan, one from the brain and one from the face, do they agree? Do they capture the same underlying aging process, or partially independent biological signals?

Why this is interesting:
- prior work typically paired MRI with photographs taken separately
- this repo extracts both signals from a single file
- the face signal here is morphological and MRI-derived, not a standard photo-age setting

---

## Hypotheses

| ID | Hypothesis | Primary test |
|----|------------|--------------|
| H1 | FaceAge applied to MRI-rendered faces detects a monotonic aging signal | Spearman correlation on IXI |
| H2 | Brain-age gap and face-age gap from the same T1 are positively correlated | Spearman correlation on IXI |
| H3 | Face-age variability across scanners is lower than brain-age variability | CV comparison on SIMON |
| H4 | Multi-contrast renderings improve face-age signal versus T1-only renders | Delta correlation on IXI with T1/T2/PD |

The main baseline question comes from prior work showing mixed evidence on whether facial aging and brain aging track together. This repo tests that link with AI-derived MRI face renders rather than subjective ratings.

---

## Pipeline

```text
T1 MRI (.nii.gz / .mgz)
        |
        |-- skull-strip -> brain parenchyma
        |       `-- SynthSeg / SFCN -> brain_age
        |
        `-- marching cubes on face surface
                `-- PyVista frontal render -> face.png
                        `-- FaceAge -> face_age

Per subject:
  chron_age
  brain_age
  face_age
  brain_age_gap = brain_age - chron_age
  face_age_gap  = face_age  - chron_age
```

---

## Datasets

### SIMON

Single-subject MRI reliability dataset.

- 1 healthy male
- 73 sessions
- 36 scanners
- 13 MRI models
- ages 29-46
- faces present in the non-defaced data

Use in this repo:
- scanner-variance and repeatability analyses
- first Colab `SFCN` baseline

Current cleaned working subset:
- `99` T1 scans
- `73` sessions
- age populated from session-level phenotype metadata

### IXI

Cross-sectional healthy brain MRI from three London sites.

- about 580 subjects
- ages 20-86
- T1, T2, and PD modalities
- sites: Guys, HH, IOP
- non-defaced

Use in this repo:
- main cross-sectional face-age versus brain-age analysis
- chronological-age benchmarking
- multi-contrast render experiments

Current cleaned T1 subset:
- `581` T1 files inspected
- `561` T1 subjects with resolved age
- `20` excluded from automatic age assignment because metadata were missing or conflicting

Download reference: <https://brain-development.org/ixi-dataset/>

---

## Repository Structure

```text
faceage-to-brainage/
|-- src/
|   |-- render.py
|   |-- face_age.py
|   |-- brain_age.py
|   `-- utils.py
|
|-- scripts/
|   |-- batch_render.py
|   |-- batch_brain_age.py
|   |-- batch_sfcn.py
|   `-- batch_face_age.py
|
|-- notebooks/
|   |-- 01_poc_single_scan.ipynb
|   |-- 02_simon_reliability.ipynb
|   |-- 03_ixi_main_experiment.ipynb
|   |-- 04_multicontrast_rgb.ipynb
|   `-- 05_sfcn_colab_bootstrap.ipynb
|
|-- notes/
|   |-- literature_review.md
|   `-- brain_age_reproduction.md
|
|-- related works/
|   |-- README.md
|   |-- face_age.md
|   |-- brain_age.md
|   `-- research_questions.md
|
|-- config/
|   `-- brain_age_runtime.example.json
|
|-- data/
|   `-- README.md
|
|-- vendor/
|-- results/
|-- requirements.txt
`-- environment.yml
```

Related docs:
- `related works/README.md`: current shortlist of testable and promising related works
- `related works/research_questions.md`: dated memo on face-age versus brain-age questions and future work
- `notes/brain_age_reproduction.md`: current `SFCN`-first reproduction note

---

## Setup

### Option A: conda

```bash
conda env create -f environment.yml
conda activate faceage
```

### Option B: pip

```bash
pip install -r requirements.txt
```

### External model repos

```bash
# FaceAge
git clone https://github.com/AIM-Harvard/FaceAge vendor/FaceAge
# Put FaceAge_weights.pt and age_regressor.pt into vendor/FaceAge/models/

# SFCN
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain vendor/SFCN
# Put run_20190719_00_epoch_best_mae.p into vendor/SFCN/brain_age/
```

### Runtime config

Paths for `SFCN` inference are intentionally not hardcoded in repo-tracked code.

- tracked example: `config/brain_age_runtime.example.json`
- local ignored runtime file: `config/local/brain_age_runtime.json`

Copy the example into `config/local/` and fill in your local paths before running `scripts/batch_sfcn.py`.

### FreeSurfer / SynthStrip

- `SynthSeg` still requires a local FreeSurfer installation if you want the volumetric pipeline
- the Colab `SFCN` path does not use `google.colab.drive.mount()`
- `notebooks/05_sfcn_colab_bootstrap.ipynb` installs standalone `SynthStrip` inside the Colab runtime and calls it via `mri_synthstrip`

---

## Quickstart

### Local workflow

```bash
# 1. One-scan proof of concept
jupyter notebook notebooks/01_poc_single_scan.ipynb

# 2. Batch render face images
python scripts/batch_render.py data/simon/ results/simon_renders/ --workers 4

# 3. Run FaceAge on rendered PNGs
python scripts/batch_face_age.py results/simon_renders/ results/simon_face_ages.csv --faceage vendor/FaceAge --bypass-mtcnn

# 4. Run SynthSeg brain-age branch
python scripts/batch_brain_age.py data/simon/ results/simon_brain_ages/

# 5. Run SFCN from local config
python scripts/batch_sfcn.py --dataset simon --limit 3
```

### Colab `SFCN` quickstart

The current Colab path is `SIMON`-first and expects one downloadable archive, not a mounted Google Drive folder.

Expected archive layout:

```text
bundle_root/
  simon/
    manifest.csv
    images/
    metadata/
  weights/
    run_20190719_00_epoch_best_mae.p
```

Recommended flow:

1. Prepare one archive file such as `simon_bundle.zip` or `simon_bundle.tar.gz`.
2. Put `simon/manifest.csv`, `simon/images/`, `simon/metadata/`, and the official `SFCN` weight into that archive.
3. Upload the archive somewhere with a direct downloadable URL.
4. Open `notebooks/05_sfcn_colab_bootstrap.ipynb` in a GPU-backed Colab runtime.
5. Set `SIMON_BUNDLE_URL` in the first config cell.
6. Run the notebook top-to-bottom.

What the notebook does:
- clones this repo
- clones the official `SFCN` repo
- installs standalone `SynthStrip`
- downloads and extracts the `SIMON` bundle under `/content/brain_assets`
- rewrites `manifest.csv` into `manifest_colab.csv`
- writes `config/local/brain_age_runtime.json`
- runs standalone `SynthStrip` validation
- runs `python scripts/batch_sfcn.py --dataset simon --limit 3`

Current scope of the notebook:
- `SIMON` only
- no `drive.mount()` dependency
- no hardcoded local data paths in repo-tracked code

---

## Brain-Age Reproduction Status

The current brain-age plan is:

1. `SFCN` first
2. `SynthBA` second
3. `MIDIconsortium BrainAge` after a stable baseline
4. `BrainIAC` later, if needed

Why `SFCN` first:
- public repo and public pretrained weight
- T1-only baseline matches both `IXI` and `SIMON`
- already supported by the current wrapper and batch runner
- lowest-cost first reproduction target

Current local code support is strongest for `FaceAge`, `SFCN`, and the `SynthSeg`-based volumetric pipeline.

Important caution:
- treat the current `SFCN` wrapper in `src/brain_age.py` as provisional until age-bin decoding is fully validated against the official implementation

---

## Key External Tools

| Tool | Paper / source | Use in this project |
|------|----------------|---------------------|
| FaceAge | Bontempi et al. 2025 | face age from MRI renders |
| SynthSeg | Billot et al. 2023 | segmentation-based volumetric branch |
| SFCN | Peng et al. 2021 | first brain-age baseline |
| BrainIAC | Tak et al. 2026 | later high-capacity reference baseline |

See `notes/literature_review.md` and the `related works/` folder for the broader literature context.

---

## Computing

Experiments can run locally in VS Code. GPU-heavy inference is currently expected to run in Colab.

Current preferred Colab path for brain-age:
- official `SFCN`
- standalone `SynthStrip`
- one downloadable `SIMON` bundle archive
- no `google.colab.drive.mount()` dependency

All `src/` modules are intended to remain Colab-compatible. PyVista rendering uses offscreen mode where needed.

---

## Contact

Ekaterina Kondrateva - kondratevakate@github

Affiliations:
- Maastricht University
- MAASTRO Clinic

Collaboration:
- AIM-Harvard
- Mass General Brigham

