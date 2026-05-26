# Project Timeline — what we did, what we saw, what we are doing next

A chronological narrative of the `faceage-to-brainage` project. The
experiment cards (`E01`–`E13`) are flat per-result; this file is the
*story* — hypotheses, surprises, dead ends, course corrections. Read
top to bottom for the reasoning trail; jump to a card for the numbers.

---

## Phase 0 · Setup — late March 2026

**2026-03-26** · Kate (KK) opens the repo. Claude scaffolds an initial
structure: empty notebooks `01..04`, vendor folders for SFCN and FaceAge,
batch scripts, MIDL paper template. **None of those scaffolded notebooks
will ever be executed**; they remain stubs (audit at the bottom of
`README.md`).

The project's working hypothesis at this point is the framing question:
> Does the face in your MRI scan know how old your brain is? — i.e.
> if we extract face age and brain age from the *same* T1 volume, do
> the two aging signals correlate?

Datasets considered at the start: **IXI** (581 healthy adults, 3 London
sites, T1/T2/PD, non-defaced) and **SIMON** (single subject, 36
scanners, longitudinal). Both already non-defaced and openly licensed.

---

## Phase 1 · First brain-age attempts on Colab — 8–14 April

**2026-04-08 (KK, `129cc76`)** — first Colab notebook for SFCN brain
age. The next 6 days (April 14, ~25 commits in one day) are spent
fighting infrastructure: MONAI `AddChannel` removal, hd-bet CLI flag
churn (v1 → v2 dropped `-mode`, renamed `-device` → `--device`), Drive
manifest schema mismatch, MIDI `pre_process.py` regex patches, numpy
ABI issues forcing kernel restart.

This is the period the user later describes as "что-то странное с
препроцессингом". **Key takeaway**: SFCN and MIDIBrainAge are very
sensitive to vendor versions. By April 16, neither produces a saved
output table on IXI or SIMON — both are listed as "started but did
not finish" in the experiments index.

Output tables produced in this phase: **none that survive**. Notebooks
05 (`05_sfcn_colab_bootstrap.ipynb`) and 06 (`06_brainage_colab.ipynb`)
end up partial or with errors and are now archived.

---

## Phase 2 · SynthBA on IXI and SIMON — 14–15 April

**2026-04-14 (KK)** — pivot to **SynthBA** (Puglisi et al.,
arXiv:2406.00365). It is contrast-agnostic, has packaged inference, and
sidesteps the SFCN/MIDI vendor mess. Within a day:

- **E05** — SynthBA on IXI. **MAE 6.33 yr, r=0.84, bias −4.15** on
  N=105. *Open question already at this point*: why only 105 / 581?
  No per-subject failure log was kept, so the answer doesn't surface
  for another 10 days (it surfaces in [E13](E13_data_audit.md)).
- **E06** — SynthBA on SIMON. **SD across scans = 1.21 yr, bias =
  −16.16 yr, MAE = 16.2 yr.** This is the paper's load-bearing finding:
  the model is highly reproducible across hardware *and* highly wrong
  for this subject. *Hypothesis at this stage*: scanner reproducibility
  ≈ validity. (Will be qualified later by [E12](E12_simon_slope.md).)

---

## Phase 3 · Face branches integrated — 15–16 April

**2026-04-15/16 (RK, `1521d6f`)** — Ramil ports the FaceAge pipeline
into the repo (`src/faceage/faceage_mri/FaceAge/run_multiview*.py`) and
runs it on IXI MRI renders.

- **E01** — FaceAge on IXI multi-view: **MAE 11.34, bias +10.04,
  r=0.83, N=406.**

Ramil's renders go through marching cubes at threshold `t = 30`.
Crucially: this threshold turns out to drop almost all of the IOP site
(2 of ~70 IOP scans survive — only revealed by the audit in [E13](E13_data_audit.md)).

**2026-04-15 (KK)** — calibration ablation:

- **E02** — linear calibration on 20, applied to 80: MAE 10.24 yr,
  bias −2.12 yr.
- **E03** — Stable Diffusion photorealistic augmentation. *Hypothesis*:
  closing the photo-vs-render domain gap with synthetic skin/texture
  should reduce FaceAge MAE. *Result*: **MAE 19.91 yr** — almost 2× worse,
  bias = MAE (every prediction is 20 yr too old). The generative model
  injects photorealistic age cues uniformly — a strong **negative
  result** that ends up cited in the paper as a caution against
  appearance prettifying.

**2026-04-16 (GB, `c3773f0`)** — Gleb commits the morphometric branch:
BioFace3D-20 landmarks → GPA → EDMA → Ridge regression.

- **E04** — face morphometrics on IXI: **MAE 7.81, r=0.78** on N=72.
  Better than FaceAge on the same data, despite using only geometry
  (no skin, no texture) — interpreted then as evidence that MRI face
  surfaces carry meaningful aging shape information.

---

## Phase 4 · MIDL submission — 16 April

**2026-04-16 (KK)** — the IXI gap correlation is computed and the MIDL
short paper is assembled.

- **E07** — gap correlation on N=93 paired subjects: **r = +0.308,
  ρ = +0.318, both p < 0.01**.

This becomes the headline of the abstract: *"the two aging gaps
computed from the same scan show a statistically significant positive
correlation, suggesting partial but incomplete overlap between facial
and cerebral aging signals"*. Submission goes out with this framing.

Open questions left explicit at submission:

1. SFCN, MIDIBrainAge, BrainIAC baselines incomplete (notebooks 09, 10
   ran but produced no CSVs).
2. Multi-contrast (T1+T2+PD) renders not tried (notebook 04 never ran).
3. Scanner reproducibility of *face* age on SIMON not measured (only
   brain side).
4. No clinical/disease labels.

---

## Phase 5 · Overnight autoresearch — 24–25 April

**2026-04-24 (Claude, branch `overnight-gap-confound-hypothesis`)** —
a Karpathy-style single-hypothesis ratchet:

### H₁ (pre-registered) — the gap correlation is age-bias confound

*Decision rule*: ACCEPT if partial Pearson r controlling for
chronological age ≤ 0.15. REJECT if ≥ 0.20.

**Result**:
- Both face_gap and brain_gap have strong negative slopes against
  chronological age (face: r = −0.64; brain: r = −0.50). Each predictor
  over-estimates young subjects and under-estimates old ones — the
  classical brain-age bias (Smith et al. 2019).
- **Partial r | chron_age = −0.015** (p = 0.89).
- Bias-corrected r (Smith-style residuals): **−0.015**.

→ **ACCEPT H₁**. The headline correlation in the paper is shared
age-bias, not shared biology. See [E11](E11_gap_confound_check.md).

### A surprise from the secondary ratchet

While re-examining SIMON ([E12](E12_simon_slope.md)), the slope test
produced a result that flatly contradicts what the paper claimed
visually:

| method | slope (yr/yr) | p |
|---|---|---|
| **SynthBA** | **+0.148** | 6.9 × 10⁻⁵ ✓ |
| Face morphometrics | +0.038 | 0.84 ✗ |
| FaceAge multi-view | −0.140 | 0.44 ✗ |

The paper had said "face morphometrics show a detectable positive
trend with chronological age" — that was a visual artifact of
scanner/session scatter at fixed time-points. The real test is
p = 0.84. **Both face methods are statistically flat in this subject.**
SynthBA, despite predicting ≈27 yr regardless of true age, recovers
~15 % of the actual aging rate.

Paper rewritten on the same day to reflect the real numbers.

---

## Phase 6 · Data audit and SOTA review — 24–25 April

Two structural problems became unignorable while computing the above:

### "Where are the rest of the patients?"

[E13](E13_data_audit.md) — full audit:

| Stage | N | % of IXI-581 |
|---|---|---|
| IXI full cohort | ~581 | 100 % |
| Face raw multi-view | 406 | 70 % |
| **Brain-age SynthBA** | **105** | **18 %** |
| Paired (final) | 93 | 16 % |

Site bias (face raw multi-view): Guys 86 %, HH 70 %, **IOP 3 %** (2 of
~70). The "3 London sites" framing is not honest — we have Guys + HH
only. 80 IXI subjects are listed in face splits but were never rendered.

### SOTA review

[`sota_design.md`](../related_works/sota_design.md) confirms what the
audit suggests: with N = 93 we are under the n ≈ 194 minimum for a
publishable r = 0.2 correlation claim. The field standard since
Paplhám CVPR 2024 is also subject-disjoint k-fold CV; our face-side
splits are not formally checked for that.

---

## Phase 7 · Datasets considered as we tightened the design

Across phases 5–6 we accumulated four reference docs in
`papers/related_works/`. Each one captures a class of datasets we
would *like* to add:

| Class | Doc | Top picks |
|---|---|---|
| OOD test–retest healthy T1 | [`test_retest_datasets.md`](../related_works/test_retest_datasets.md) | Cao 2015 (N=57), Huang 2016 (N=61), Maclaren ds000239 (3×40), B-Q MINDED, GSP retest |
| Public 2D + 3D face | [`avatar_3d_datasets.md`](../related_works/avatar_3d_datasets.md) | Headspace/LYHM (swim caps = no hair), FaceScape (eyes-closed expression), MICA-unified |
| Clinical face | [`clinical_facial_datasets.md`](../related_works/clinical_facial_datasets.md) | FaceBase 3DFN, AAOF Legacy, MEEI Facial Palsy, FDTooth, MMDental |
| SOTA brain-age training | [`sota_design.md`](../related_works/sota_design.md) | OpenBHB, CamCAN, SRPBS Travelling Subjects, OASIS-3 |

We have **not yet ingested any of them**. They are the queue for the
next phase.

---

## Phase 8 · What we are doing next

The active hypothesis queue from
[`README.md`](README.md)`#active-hypotheses`:

| # | Hypothesis | Owner | Blocking on |
|---|---|---|---|
| AH-1 | Re-running SynthBA on full IXI (581) restores cohort and lets us re-test E11 with adequate power | KK | Colab GPU rerun + per-stage logging (audit action **A1**) |
| AH-2 | Adaptive marching-cubes threshold recovers IOP site | KK | inspect one IOP scan first |
| AH-3 | Multi-modal face renders (T1+T2+PD) reduce FaceAge bias | KK + RK | E01 redone with T2/PD |
| AH-4 | NeuroHarmonize before SynthBA gives ICC(2,1) ≥ 0.95 across IXI sites | KK | install + integrate |
| AH-5 | E11 confound replicates on a SIMON-disjoint cohort | KK | dataset access (Cao + Huang BNU) |
| AH-6 | Geometry-only branch (E04) tracks aging on a longitudinal multi-subject cohort | GB | larger cohort than SIMON |

Two tracks for the longer arc:

1. **Make the existing claim publishable** — fix data audit (A1–A4),
   add the four expected brain baselines, report bias-corrected MAE +
   ICC + Bland-Altman, replicate the negative gap-correlation finding
   on at least one external test–retest cohort.

2. **Build a new direction** — photo → MRI-iso-surface mesh as a
   training target. No public dataset pairs photograph + MRI-derived
   face skin surface; that is a defensible scientific contribution.
   See [`avatar_3d_datasets.md`](../related_works/avatar_3d_datasets.md)
   for the data-collection plan and the Phace / Cong physics-based
   options for an internal-anatomy follow-on (surgery use case).

---

## Hypothesis ledger (one-glance summary)

| Stage | Hypothesis | Verdict | Card |
|---|---|---|---|
| Phase 1 | SFCN works out of the box on Colab for IXI | ✗ infrastructure dead-ends | (archived) |
| Phase 2 | SynthBA gives a sensible chronological MAE on IXI | ✓ MAE 6.33 (but only 105 subjects) | [E05](E05_synthba_ixi.md) |
| Phase 2 | SynthBA is reproducible across SIMON's 36 scanners | ✓ SD 1.21 yr | [E06](E06_synthba_simon.md) |
| Phase 3 | FaceAge produces a meaningful age signal on MRI renders | ✓ but with +10 yr bias | [E01](E01_faceage_ixi_multiview.md) |
| Phase 3 | Linear calibration removes most of the bias | ✓ bias +10 → −2 | [E02](E02_faceage_ixi_calibration.md) |
| Phase 3 | Photorealistic SD augmentation closes the photo-vs-render gap | ✗ MAE doubles | [E03](E03_photorealistic_renders.md) |
| Phase 3 | Geometry-only morphometrics beats photo-based FaceAge on MRI | ✓ MAE 7.81 vs 11.34 | [E04](E04_face_morphometrics_ixi.md) |
| Phase 4 | Face-PAD and brain-PAD share an aging signal (raw r) | ✓ r=0.31 (then overturned) | [E07](E07_gap_correlation_raw.md) |
| Phase 5 | The r=0.31 is shared age-bias, not shared biology | **ACCEPT** partial r=−0.015 | [E11](E11_gap_confound_check.md) |
| Phase 5 | Some method tracks longitudinal aging on SIMON | only SynthBA, ~15 % of true rate | [E12](E12_simon_slope.md) |
| Phase 6 | The IXI cohort we computed everything on is complete | ✗ 18 % brain coverage, IOP gone | [E13](E13_data_audit.md) |

---

## Reading order for someone arriving cold

1. This file (TIMELINE.md) — project narrative.
2. [`README.md`](README.md) — index of cards + active hypotheses.
3. [E11](E11_gap_confound_check.md) and [E12](E12_simon_slope.md) — the
   findings that flipped the paper.
4. [E13](E13_data_audit.md) — why the cohort is the bottleneck before
   any new statistics.
5. [`../related_works/sota_design.md`](../related_works/sota_design.md) —
   the bar the project needs to clear.
