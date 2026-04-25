# Experiment Journal — Index

Single source of truth: what was actually run, by whom, with what
result, and what is still unfinished. **Cards exist only for
experiments that produced data** (table or analysis output).
Everything else is a row in the "started but did not finish" section.

> 📖 For the chronological narrative — what hypotheses we started with,
> what surprised us, what each phase saw — read
> **[`TIMELINE.md`](TIMELINE.md)** first.

For the **interactive** entry point, open
[`../../notebooks/00_EXPERIMENT_JOURNAL.ipynb`](../../notebooks/00_EXPERIMENT_JOURNAL.ipynb).
That notebook is the only one we should keep editing — the older
`01_…` … `11_…` notebooks are kept for reference only (most never ran,
see audit at the bottom of this file).

Add a new experiment by adding a new card here; do **not** create new
notebooks.

## Authors

| Tag | Person | Owns |
|---|---|---|
| `KK` | Ekaterina Kondrateva | brain-age pipelines, MRI preprocessing, SIMON, MIDL paper, infra |
| `RK` | Ramil Khafizov (`smileyenot983`) | FaceAge integration, MRI render → photo pipeline |
| `GB` | Gleb Bobrovskikh (`BobrG`) | face morphometrics (BioFace3D-20, GPA + EDMA, Ridge) |
| `CL` | Claude (autoresearch) | analyses, audits, hypothesis tests, documentation |

## Completed experiments (have cards + data)

| ID | Date | Author | Topic | Headline | Card |
|---|---|---|---|---|---|
| E01 | 2026-04-15 | RK | FaceAge on IXI multi-view renders | MAE 11.34 yr, bias +10 yr, r=0.83 | [E01](E01_faceage_ixi_multiview.md) |
| E02 | 2026-04-15 | KK | Linear calibration of FaceAge on IXI | MAE 10.24 yr, bias −2.1 yr (test N=80) | [E02](E02_faceage_ixi_calibration.md) |
| E03 | 2026-04-15 | KK | Photorealistic SD-augmented renders | MAE 19.91 yr — **negative result** | [E03](E03_photorealistic_renders.md) |
| E04 | 2026-04-16 | GB | Face morphometrics (GPA+EDMA, Ridge) IXI | MAE 7.81 yr, r=0.78 | [E04](E04_face_morphometrics_ixi.md) |
| E05 | 2026-04-14 | KK | SynthBA on IXI (subset) | MAE 6.33 yr — **only 105/581 subjects** | [E05](E05_synthba_ixi.md) |
| E06 | 2026-04-14 | KK | SynthBA on SIMON | SD 1.21 yr ✅, bias −16.2 yr ❌ | [E06](E06_synthba_simon.md) |
| E07 | 2026-04-16 | KK | Raw gap correlation IXI | r=0.31 — **overturned by E11** | [E07](E07_gap_correlation_raw.md) |
| E11 | 2026-04-24 | CL | IXI gap age-bias confound check | partial r \| age = −0.015 | [E11](E11_gap_confound_check.md) |
| E12 | 2026-04-24 | CL | SIMON longitudinal slope test | only SynthBA tracks aging (15%) | [E12](E12_simon_slope.md) |
| E13 | 2026-04-25 | CL | Data audit | IOP gone, 105/581 brain-age coverage | [E13](E13_data_audit.md) |

## Started but did not finish (no cards, planned to retry)

| Date | Author | What | Status | Why retry |
|---|---|---|---|---|
| 2026-04-08 | KK | SFCN on IXI/SIMON | notebook 05 partial (1/15 cells), no CSV | needed as 2nd brain baseline |
| 2026-04-14 | KK | SFCN preprocessing in `06_brainage_colab.ipynb` | 7/14 cells, **3 errors** | same |
| 2026-04-15 | KK | MIDIBrainAge on SIMON (`09_midi_…`) | ran 5/9 cells, **no CSV exported** | needed as 3rd brain baseline |
| 2026-04-15 | KK | BrainIAC on SIMON (`10_brainiac_…`) | ran 8/13 cells, **no CSV exported** | needed as 4th brain baseline |
| — | KK | Re-run SynthBA on full IXI | not started | 18% coverage is the bottleneck |
| — | KK | Adaptive marching-cubes for IOP | not started | recover IOP site |
| — | RK | Subject-disjoint k-fold CV per Paplhám 2024 | not started | make MAE numbers defensible |

## Active hypotheses (next ratchet)

| # | Hypothesis | Owner | Blocking on |
|---|---|---|---|
| AH-1 | Full-IXI SynthBA recovers `r > 0.4` between brain-PAD and brain-PAD' (rerun reproducibility) | KK | A1 — Colab GPU rerun + per-stage logging |
| AH-2 | IOP can be recovered with adaptive marching-cubes threshold | KK | inspect one IOP scan first |
| AH-3 | After bias correction, multi-modal brain (T1 + T2 + PD) face renders beat T1-only | KK + RK | E01 redone with multi-contrast |
| AH-4 | NeuroHarmonize before SynthBA reduces site bias to within ICC(2,1) ≥ 0.95 | KK | install + integrate |
| AH-5 | Confound check (E11) replicates on a SIMON-disjoint cohort | KK | dataset access — see [`test_retest_datasets.md`](../related_works/test_retest_datasets.md) |
| AH-6 | Geometry-only (E04) tracks longitudinal aging in a 5-yr+ cohort, not in SIMON's 17-yr-1-subject window | GB | larger longitudinal cohort |

## Notebook reality check (2026-04-25)

| Notebook | Code cells | Executed | Errors | Status |
|---|---|---|---|---|
| 01_poc_single_scan.ipynb | 7 | 0 | 0 | NEVER-RAN (stub) |
| 02_simon_reliability.ipynb | 7 | 0 | 0 | NEVER-RAN (stub) |
| 03_ixi_main_experiment.ipynb | 9 | 0 | 0 | NEVER-RAN (stub) |
| 04_multicontrast_rgb.ipynb | 8 | 0 | 0 | NEVER-RAN (stub) |
| 05_sfcn_colab_bootstrap.ipynb | 15 | 1 | 0 | partial |
| 06_brainage_colab.ipynb | 14 | 7 | 3 | ran with errors |
| 07_synthba_colab.ipynb | 6 | 4 | 0 | **RAN-OK → E05** |
| 08_synthba_simon_colab.ipynb | 7 | 6 | 0 | **RAN-OK → E06** |
| 09_midi_simon_colab.ipynb | 9 | 5 | 0 | RAN-OK but no CSV exported |
| 10_brainiac_simon_colab.ipynb | 13 | 8 | 0 | RAN-OK but no CSV exported |
| 11_results_analysis.ipynb | 6 | 0 | 0 | NEVER-RAN (stub) |

Real-experiment yield from `notebooks/`: **2 / 11** that produced data
tables (07, 08). The face branches (E01, E04) were run from `src/`
scripts directly, not notebooks. Stubs (01–04, 11) were created during
initial scaffolding and should be **deleted** once the master journal
notebook (`00_EXPERIMENT_JOURNAL.ipynb`) is in place.

**New policy**: no more single-purpose notebooks. Add a card here, run
code from `00_EXPERIMENT_JOURNAL.ipynb`, append numbers to
`papers/midl2026/RESULTS.tsv`.

## Cumulative results log

Machine-readable: [`../midl2026/RESULTS.tsv`](../midl2026/RESULTS.tsv).
Appended by every analysis script under `papers/midl2026/` via the
shared `log()` helper.

## Reference docs

- [`../related_works/sota_design.md`](../related_works/sota_design.md) — SOTA design brief, 2024–2026
- [`../related_works/data_audit.md`](../related_works/data_audit.md) — IXI/SIMON coverage audit (E13)
- [`../related_works/literature_review.md`](../related_works/literature_review.md) — full literature review
- [`../related_works/avatar_3d_datasets.md`](../related_works/avatar_3d_datasets.md) — 2D+3D face datasets
- [`../related_works/clinical_facial_datasets.md`](../related_works/clinical_facial_datasets.md) — clinical face datasets
- [`../related_works/test_retest_datasets.md`](../related_works/test_retest_datasets.md) — small/obscure test–retest T1 cohorts (in progress)
