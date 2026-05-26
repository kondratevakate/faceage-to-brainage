# IXI data audit — what we have vs what we should have

Date: 2026-04-24 · Branch: `overnight-gap-confound-hypothesis`

## Summary

We are losing **82% of IXI brain-age subjects and nearly 100% of the IOP
site** before reaching the gap-correlation experiment. The paper's
headline numbers are computed on an underpowered and **site-biased**
sub-cohort. Before any further modelling, this pipeline needs a
re-run and a per-stage failure log.

## Subject counts by pipeline stage

| Stage | Source file | N subjects | % of IXI-581 |
|---|---|---|---|
| IXI full cohort (published) | brain-development.org | **~581** | 100% |
| Face multi-view raw | `ixi_faceage_multiview_raw.csv` | **406** | 70% |
| Face splits (train+val+test) | `ixi_faceage_splits.csv` | **476** | 82% |
| Brain-age SynthBA | `ixi_brainage_synthba.csv` | **105** | **18%** ← bottleneck |
| Paired (brain ∩ face) | `ixi_gap_correlation.csv` | **93** | **16%** ← final |

- **Brain-age pipeline coverage is the dominant loss**: 105 of 581 (18%).
  SynthBA is cross-sequence robust and has a packaged runtime — there is
  no scientific reason to stop at 105 subjects. Most likely the batch
  run was interrupted or scoped to a subset.
- **80 subjects listed in face splits but absent from raw multiview**:
  these are labelled train/val/test but were never actually rendered,
  i.e.\ they were planned but the render pipeline stopped short. These
  are part of the "half missing" the user flagged.

## Site distribution (site bias)

Face multi-view raw — by IXI site:

| Site | N in raw multiview | IXI published N | Coverage |
|---|---|---|---|
| Guys | 275 | ~320 | 86% |
| HH | 129 | ~185 | 70% |
| **IOP** | **2** | **~70** | **3%** ← IOP is effectively gone |

Face splits (`test/val/train`) show only **Guys** and **HH**:
```
site   test  train  val
Guys     48    215   42
HH       24    120   27
IOP       0      0    0
```

**Implication**: every IXI analysis we ran is Guys + HH only.
The "3 London sites" framing in the paper is misleading. This needs
either (a) re-running the face render pipeline on IOP scans with
different parameters, or (b) honest reporting of 2 sites in the paper.

## Diagnostic checklist (what to investigate first)

1. **Why only 105 brain-age subjects?** Likely candidates:
   - batch script `scripts/batch_brain_age.py` hit a memory/time limit
   - list of input NIfTIs was pre-filtered to a subset
   - SynthStrip failed on specific subjects and we didn't log per-stage errors

2. **Why IOP is nearly absent from face rendering?**
   - IOP T1 sequence has different intensity distribution → marching-cubes
     threshold `t=30` may cut below the skin on IOP scans
   - IOP field-of-view may be different (head coil, partial volume)
   - need a per-site histogram of intensity at the skin iso-surface

3. **Why 80 subjects in splits but not in raw renders?**
   - the splits file was generated from a planned cohort; the render
     pipeline never actually ran on them
   - probably just a stalled batch; re-run should recover them

## Action items, ranked

### A1. Diagnose and fix brain-age coverage (highest priority)

Full IXI should run on a single consumer GPU in about 4 h.

```bash
python scripts/batch_brain_age.py \
    papers/data/ixi/T1/ \
    papers/tables/ixi_brainage_synthba_full.csv \
    --skull-strip synthstrip \
    --log-per-subject \
    --resume 2>&1 | tee ixi_brainage_full.log
```

Add per-stage failure logging so we can see exactly which subjects fail
at SynthStrip vs at SynthBA vs at output writing.

### A2. Diagnose IOP site rendering

Before re-running 70 IOP subjects blind, inspect ONE working IOP scan:

```python
import nibabel as nib
import numpy as np
img = nib.load("papers/data/ixi/T1/IXI029-IOP-xxx-T1.nii.gz")
arr = img.get_fdata()
# look at the intensity histogram at skin depth
print(np.percentile(arr, [50, 75, 90, 95, 99]))
```

If IOP intensity peaks are systematically different from Guys/HH, the
marching-cubes threshold `t=30` is the wrong cut. Fix by per-scan
adaptive threshold (Otsu, or a simple N(5th, 95th) percentile cut).

### A3. Re-render the 80 planned-but-unrendered face subjects

These already have train/val/test assignments. Fill them in so the
splits are usable.

### A4. Per-subject failure log

Every batch script should write a log line per subject:
`subject_id, stage, status, wall_time, error_message`. Without this we
cannot even see which stage is failing; we can only see that N is too
small.

## Expected post-fix numbers

| Metric | Current | Expected after A1–A3 |
|---|---|---|
| Brain-age N (IXI) | 105 | **~550** |
| Face-age N (IXI) | 406 | **~550** |
| Paired N (IXI) | 93 | **~500** |
| Sites represented | 2 (Guys, HH) | **3** (Guys, HH, IOP) |

At n = 500 the gap correlation is powered for effects as small as
r ≈ 0.13 at α = 0.05, 80% power. At n = 93 we can only detect effects
r ≥ 0.28 reliably, which is why the raw r = 0.31 sits exactly at the
edge of detectability and why our partial r = −0.015 looks so clean.

## Cross-reference to SOTA design brief

See [`sota_design.md`](sota_design.md) for the published-protocol
recommendations the expanded pipeline should follow:

- **Preprocessing**: SynthStrip before SynthBA (already planned above),
  NeuroHarmonize across sites for brain-age features
- **Baselines**: minimum 4 brain (SFCN, SynthBA, BrainIAC, MIDIBrainAge)
  and 2–3 face (FaceAge, MiVOLOv2, FAHR-FaceAge if released)
- **Reporting**: raw MAE + Smith-2019 bias-corrected MAE, both with
  10k-bootstrap 95% CIs; ICC(2,1) with CI for reproducibility;
  Bland-Altman limits of agreement
- **External OOD cohort**: add OpenBHB or CamCAN before final submission;
  IXI+SIMON alone will not satisfy reviewers on generalisation claims
- **Minimum N for r = 0.2 correlation claim**: ≈194 subjects (80% power,
  α=0.05). We currently have 93.

## Not in scope for this audit (but flagged for later)

- Face-age pipeline may need MiVOLO-v2 as a chronological-age comparator
  in addition to FaceAge (which targets biological age).
- The morphometric branch's training data needs disclosure: was it
  trained on IXI? If yes, all IXI morphometric MAE numbers are biased
  optimistic (subject leakage in k-fold is a known Paplhám 2024 failure
  mode).
- SIMON face morphometrics have 82 scans but face multiview has 90;
  check why the two face methods differ in sample size on the same cohort.
