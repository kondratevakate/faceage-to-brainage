# BrainChop 0.2.5 Application Branch

Date: 2026-06-26

## Purpose

Add the latest reproducible BrainChop command line branch to the Kate n=1
application study without promoting unverified outputs into the pseudo-GT or
visualization projects.

## Source and version

- Package: `brainchop==0.2.5`.
- Release observed on PyPI: 2026-06-24.
- CLI repository: `https://github.com/neuroneural/brainchop-cli`.
- Browser application repository: `https://github.com/neuroneural/brainchop`.
- Live browser app: `https://brainchop.org/`.
- Local source manifest: `experiments/kate_n1_2026/brainchop_sources.json`.

The CLI branch is preferred over manual browser use for this project because it
can be pinned, logged, timed out, and rerun from a manifest.

## Local setup result

Local WSL setup completed:

- Python 3.12 venv: `~/.venvs/brainchop`.
- Installed package: `brainchop==0.2.5`.
- `brainchop --list` works.
- Models observed: `robust_tissue`, `big_robust_tissue`, `tissue_fast`,
  `subcortical`, `subcortical-mini`, `DKatlas`, `mindgrab`, `aparc50`.
- `clang` was installed because tinygrad CPU inference requires it; `gcc` alone
  failed due a clang-specific `--target=x86_64-none-unknown-elf` compile flag.

## Smoke-run status

Attempted command class:

```bash
brainchop 401_t1w_ffe.nii.gz \
  -m subcortical \
  --inverse-conform \
  --no-optimize \
  -o kate_2024_t1_ffe_401_subcortical.nii.gz
```

Observed failure modes:

1. Directly calling the venv binary failed because `niimath` was not on `PATH`.
   Activating the venv fixes this.
2. CPU inference initially failed because `clang` was missing.
3. With `brainchop==0.2.4`, the `subcortical` model reached inference but did
   not finish within a 15 minute local CPU smoke window on one 2024 FFE scan.
   The process was stopped and no subcortical output label map was promoted.

## BrainChop 0.2.5 smoke result

The CLI was upgraded to `brainchop==0.2.5` and rerun with bounded per-scan
timeouts on 2024 T1-like candidates.

Tracked summary:

- `data/kate_n1_2026/brainchop_0.2.5_smoke_results.csv`

Runtime output root:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\brainchop\brainchop_0.2.5
```

`tissue_fast` completed on all three 2024 candidates:

| Scan | Status | Runtime | Labels |
|---|---:|---:|---:|
| 2024 3DI | done | 31.912 s | `1;2` |
| 2024 T1 FFE 401 | done | 22.974 s | `1;2` |
| 2024 T1 FFE 601 | done | 17.835 s | `1;2` |

`mindgrab` timed out at the 5 minute per-scan limit on all three candidates.
This makes `mindgrab` unsuitable as the quick local brain-extraction baseline in
the current CPU wrapper.

`subcortical-mini` was tested on 2024 T1 FFE 401 with a 10 minute hard timeout
and did not complete. This blocks local CPU use of BrainChop as an ASEG-like
anatomical segmentation source for now.

## Scientific interpretation

There is now a completed BrainChop tissue-level result, but no completed
BrainChop subcortical or atlas-parcellation result. `tissue_fast` is useful as a
fast tissue-QC candidate and possible uncertainty/contrast-sensitivity signal.
It is not an ASEG, DKT, or FreeSurfer/SynthSeg replacement.

BrainChop should enter anatomical pseudo-GT tables only after:

- a label map is produced in original image space or a documented common space;
- the model ontology is mapped to the SynthSeg/TIGERBx/FreeSurfer structures;
- visual overlay QC passes;
- the label map is evaluated against the registered 2024 FFE pseudo-GT with
  Dice, Jaccard, HD95, ASSD, and volume error.

## Next execution options

Use the timeout-managed wrapper:

```bash
cd /mnt/d/projects/02_academia/brain-mri-segmentation
BRAINCHOP_MODELS=tissue_fast BRAINCHOP_SCAN_IDS=kate_2024_t1_ffe_401 \
  BRAINCHOP_TIMEOUT_SECONDS=300 \
  bash experiments/kate_n1_2026/run_brainchop_local.sh
```

Next candidates:

- GPU/WebGPU or Docker execution for `subcortical-mini`;
- `subcortical` only after `subcortical-mini` proves feasible;
- `DKatlas` / `aparc50` only after runtime feasibility and ontology mapping are
  explicit.

Do not add BrainChop outputs to `your-brain-mri-visualization` until visual QC
and a registered-space comparison pass. The completed `tissue_fast` outputs can
be used as a QC/uncertainty branch, not as primary anatomical labels.

## FastSurfer location and boundary

FastSurfer outputs already exist locally:

- `reprocessed_2026/fastsurfer/2018_ge_fspgr`
- `reprocessed_2026/fastsurfer/2018_ge_fspgr_full`
- `reprocessed_2026/fastsurfer/2024_phi_3di`
- `reprocessed_2026/symmetry/fastsurfer`
- `reprocessed_2026/symmetry/fastsurfer_long_v2`

FastSurfer is not part of the current 2024 registered pseudo-GT reference
because the available valid FastSurfer evidence is the 2018 rotation/Long
branch, while the 2024 3DI FastSurfer run is already documented as a collapse
case rather than a trusted 2024 FFE-compatible label source.
