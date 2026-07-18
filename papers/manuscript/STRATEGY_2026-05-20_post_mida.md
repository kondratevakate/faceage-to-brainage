# Strategy Update — Post-MIDA Reframe

**Date**: 2026-05-20 · **Verified**: 2026-05-26 · **Branch**: `overnight-gap-confound-hypothesis`
**Trigger**: MIDA v1.0 acquired (`C:\Projects\data\brain_images\MIDAv1-0.zip`, 483 MB) — single-subject whole-head label volume + 115 per-tissue STL meshes at 0.5 mm³. Local zip check confirms `MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii` and 115 files under `MIDA_v1.0/MIDA_v1_surfaces/*.stl`.

## TL;DR

The April 24 confound finding (`partial r | age = −0.015` on IXI N=93) killed the paper's headline claim that face-age and brain-age share a biological signal. **MIDA arriving on disk this week opens a clean way to recover a real publishable result**: use MIDA as a controlled in-silico phantom to test whether the two predictors *can in principle* be dissociated, and report honestly that the IXI-observed correlation is a regression-to-the-mean confound, not biological shared signal.

This is a stronger paper than the original, not a weaker one. We replace a fragile correlational claim with a **mechanistic decomposition** built on a phantom.

---

## What MIDA actually provides (and what it does not)

### Provides

- **One whole-head label volume** at 0.5 mm³ isotropic (480 × 480 × 350 voxels), 116 anatomical labels — FDA + IT'IS Foundation, manual delineation. The FDA tool page describes MIDA as a detailed anatomical head-and-neck model with 116 structures, 500 µm isotropic spatial resolution, CAD objects, and a single-subject limitation.
- **115 per-tissue STL surface meshes** — pre-extracted, no marching cubes needed
- Critical labels for our project:
  - `Epidermis/Dermis` (label 51) — the visible face surface
  - `Subcutaneous Adipose Tissue` (label 62) — facial fat compartment; treat the FaceAge paper as evidence that facial appearance carries biological-age information, but do not yet overclaim that this exact compartment is the dominant FaceAge feature without a direct citation to a facial-fat attribution analysis.
  - 25 facial muscles individually (labels 60–84) — incl. `Orbicularis Oculi`, `Zygomaticus Major/Minor`, `Masseter`, `Buccinator`
  - Eye sub-anatomy: `Eye Lens`, `Cornea`, `Retina/Choroid/Sclera`, `Vitreous`, `Aqueous` — **closed-eye scanner pose can be re-opened by perturbing these meshes**
  - Three skull layers: `Skull Outer Table`, `Skull Diploë`, `Skull Inner Table`
  - Full brain parcellation with deep nuclei, brainstem, cerebellum, 24 cranial nerve segments

### Does NOT provide

- **No synthetic T1w volume.** Only the label map. To produce a T1w-like image we would need a Bloch-equation forward model and per-tissue T1/T2/PD values (IT'IS Tissue Properties Database is free and public, so this is solvable, but ~3 days of work).
- **No realistic skin texture.** MIDA's surface is geometric only — no wrinkles, no pigmentation, no scleral colour. This is *consistent with our actual IXI pipeline* (we also render geometric surfaces) but means MIDA cannot test texture-sensitivity of FaceAge.
- **Fixed identity.** One ~29-year-old male per ITIS documentation. Aging effects must come from synthetic perturbations, not from real longitudinal data.

---

## The new hypothesis (post-MIDA)

### H₀ (the dead claim, retired)

> "Face-age and brain-age from the same MRI share a biological aging signal."

Killed by partial r = −0.015 on IXI N=93.

### H₂ (the new pre-registered hypothesis MIDA enables)

> **The face-age and brain-age predictors, when applied to the same non-defaced T1w volume, draw their signal from anatomically disjoint regions. The previously reported IXI correlation (raw r=0.31) is therefore a regression-to-the-mean confound, not biological shared signal.**

### Operationalisation (pre-registered, before any experiment runs)

We will test H₂ on the MIDA phantom with **two orthogonal perturbation experiments**:

- **PE1 — Brain perturbation, face held fixed.**
  Keep MIDA's skin/muscle/skull/eye labels identical. Replace the brain interior (GM, WM, deep nuclei, cerebellum, brainstem, CSF ventricles — labels 1–22, 99–116) with a brain transplanted from an IXI subject of known age. Run *both* predictors.
  - Predicted: **FaceAge unchanged** within renderer error. **Brain-age moves** toward the IXI subject's age.
  - Decision rule: H₂ is supported if Δ_FaceAge < 1 year AND Δ_brain-age > 5 years on at least 10 transplants.

- **PE2 — Face perturbation, brain held fixed.**
  Keep MIDA's brain identical. Perturb skin surface, subcutaneous adipose volume, and facial muscle thickness using aging-trajectory parameterisations from dermatology / craniofacial literature. Run both predictors.
  - Predicted: **brain-age unchanged**. **FaceAge moves** in the direction of the perturbation.
  - Decision rule: H₂ is supported if Δ_brain-age < 1 year AND |Δ_FaceAge| > 2 years on at least 5 perturbation magnitudes.

### What rejects H₂

- If PE1 moves FaceAge by > 1 year → renderer is leaking brain shape into face-age signal → the SIMON failure has a different mechanism, and the original IXI correlation might survive with a better renderer
- If PE2 moves brain-age by > 1 year → SynthBA is sensitive to skull / face shape (not just brain) → we have a calibration problem, not a biological finding
- If neither moves either way → both predictors are stuck on age-bias prior → entire pipeline is uninformative

---

## Experiments — concrete plan

### EXP-0. MIDA baseline render — `notebooks/12_mida_baseline.ipynb`

**Goal**: a sanity check. Render MIDA's `Epidermis_Dermis.stl` from the same 9 viewpoints as the IXI pipeline. Run FaceAge.

- Input: `Epidermis_Dermis.stl`
- Output: 9 PNG renders + 1 FaceAge prediction
- Expected: FaceAge prediction ≈ 25–35 years (MIDA is documented as ~29-year-old male)
- Risk: STL has no texture; FaceAge might fail or hallucinate. *That itself is a finding* — it means our IXI face-age signal is geometry-only, not texture-based, which constrains interpretation.

**Outcome decides**: do we proceed with MIDA-as-phantom (if baseline render gives sensible age) or scope back to a different approach.

### EXP-1. PE1 brain transplant — `notebooks/13_brain_perturbation.ipynb`

**Method**:
1. Pick 10 IXI subjects with widely varying ages (20–75 y), all with FreeSurfer-processed brain labels available
2. For each: affine-register the IXI brain region to MIDA's brain bounding box, then label-by-label transplant brain interior into MIDA's label volume. Smooth the transition zone with a 2-voxel morphological dilation.
3. Forward-simulate a T1w volume from the chimeric label map using IT'IS tissue T1/T2/PD values + a simple MPRAGE Bloch model
4. Run our full pipeline: marching cubes → 9 renders → FaceAge; SynthBA → brain-age
5. Report Δ_FaceAge and Δ_brain-age for each transplant

**Tables**:
- `papers/tables/mida_pe1_transplants.tsv` — one row per transplant
- Columns: `donor_ixi_id`, `donor_age`, `mida_baseline_brain_age`, `chimera_brain_age`, `delta_brain_age`, `mida_baseline_face_age`, `chimera_face_age`, `delta_face_age`

### EXP-2. PE2 face perturbation — `notebooks/14_face_perturbation.ipynb`

**Method**:
1. Three independent perturbation dimensions on MIDA's STLs (keep brain & skull unchanged):
   - **Skin sag**: displace `Epidermis_Dermis.stl` vertices below the cheekbone downward along the gravity vector by `d ∈ {2, 4, 6, 8} mm`
   - **Adipose loss**: shrink `Subcutaneous Adipose Tissue.stl` (label 62) volume by `k ∈ {10, 20, 30, 40}%` via uniform inward normal displacement
   - **Muscle atrophy**: shrink each of 12 facial expression muscles by `k ∈ {10, 20, 30}%`
2. Voxelise back to a label volume; forward-simulate T1w as in PE1
3. Run both predictors
4. Report Δ for each perturbation magnitude

**Tables**: `papers/tables/mida_pe2_perturbations.tsv`

### EXP-3 (stretch). Eye-state intervention — `notebooks/15_eye_opening.ipynb`

**Goal**: address the closed-eye limitation of MRI face renders. Replace the closed-eye region of MIDA's skin with a synthetic open-eye configuration by moving `Epidermis_Dermis` vertices over the orbit upward and exposing the underlying `Eye Cornea` + `Eye Retina/Choroid/Sclera` STLs. Does this move FaceAge?

If FaceAge changes substantially when eyes are opened (independent of any other perturbation), this is a publishable separate finding: **the closed-eye MRI pose systematically biases face-age prediction**, regardless of brain anatomy.

---

## Paper reframe

### Old structure (mostly dead)

1. Intro: face has age signal, brain has age signal, same scan has both → correlation
2. Method: pipeline
3. Result: r=0.31 on IXI
4. Discussion: shared biological aging

### New structure (alive, MIDA-enabled)

1. **Intro**: "Both face and brain encode age. Can a single non-defaced T1w give us both? We test whether the two predictors share biological signal or merely share a statistical confound."
2. **Method**:
   - Pipeline as before
   - **New: MIDA-based phantom protocol for dissociation testing**
3. **Results**:
   - 3.1 IXI raw correlation = 0.31 (replicate from prior draft)
   - 3.2 Age-bias correction collapses it to −0.015 (the April 24 finding — now a feature, not a bug)
   - 3.3 SIMON longitudinal: brain-age tracks (slope +0.148), face-age does not
   - 3.4 **MIDA phantom**: PE1 brain transplants move brain-age by Δ̄ years, face-age by ε. PE2 face perturbations move face-age by Δ̄ years, brain-age by ε. (If H₂ supported)
4. **Discussion**:
   - The two predictors are anatomically dissociable in principle (PE1 + PE2)
   - The IXI correlation is a regression-to-the-mean confound, not biological
   - The within-subject SIMON failure of FaceAge suggests our 9-view geometric render is *insufficient* for capturing within-subject aging — a render-pipeline limitation, not a biological one
   - **A phantom-based dissociation protocol** is offered as a reusable methodology for any future face/brain-age combination paper

This is honest, methodologically clean, and *publishable*. Negative result + reusable phantom protocol > spurious correlation.

---

## Risks and how to mitigate

| Risk | Probability | Mitigation |
|---|---|---|
| Forward MR simulation (Bloch + IT'IS tissue properties) takes longer than 3 days | Medium | Skip simulation: do PE1 / PE2 at the **mesh level only**. Render the perturbed STL directly through the FaceAge pipeline as we already do for IXI. SynthBA needs a volumetric T1w though — fall back to perturbing IXI subjects directly and using SynthBA's own brain-age output, treating MIDA as a *geometric* phantom only |
| FaceAge fails on textureless MIDA renders | Medium | Acceptable finding. Reframe as "FaceAge requires texture; our IXI pipeline strips texture; therefore IXI face-age is operating on geometry only — and geometry alone is insufficient for within-subject aging detection". This is *the* paper. |
| MIDA license forbids publishing renders derived from the model | Low | Verify ITIS license terms for derived works. If forbidden, publish summary statistics and Δ values only, not renders |
| PE1 transplants create unrealistic chimeras the predictors refuse to score | Medium | Smooth label transitions, run on multiple smoothing scales, report sensitivity. If still fails, switch to in-place atrophy / hypertrophy (no transplant, just GM thickness modulation à la Rusak 2022 SynthAtrophy) |
| Insufficient runway after a venue is selected | Unknown until a target and deadline are chosen | Time-box EXP-0 to 3 days. If the baseline render gives nonsense, keep the narrower confound and SIMON story without the phantom angle |

---

## Decision points (your call, not mine)

1. **Accept the reframe?** The headline becomes a negative result on the original claim PLUS a phantom-based methodology contribution. The original "face-age and brain-age are correlated" story is gone. Are you OK with that, or do you want to first try a larger cohort (OASIS / Cam-CAN) to see if the correlation survives there?

2. **Venue and schedule?** No target is currently selected. Complete the core confound and SIMON analysis first, then choose a venue whose scope and format fit the evidence rather than designing to an assumed deadline.

3. **Forward MR simulation or mesh-only phantom?** The mesh-only path is faster but cannot use SynthBA (volumetric). The simulation path is more rigorous but adds 3–5 days. Recommendation: start mesh-only for FaceAge, decide on brain-age path after EXP-0.

4. **Brain perturbation method — transplant or modulation?**
   - Option A (transplant): chimerise MIDA brain with IXI brains. Realistic but ugly seams.
   - Option B (modulation): keep MIDA's brain labels but apply SynthAtrophy-style cortical thinning fields parameterised by age. Cleaner but requires reimplementing Rusak 2022 or contacting them for code. Recommendation: A first (1 week), B as stretch goal.

---

## Files to create when this plan is approved

- `notebooks/12_mida_baseline.ipynb` — render MIDA skin STL + FaceAge (EXP-0)
- `notebooks/13_brain_perturbation.ipynb` — PE1 transplants
- `notebooks/14_face_perturbation.ipynb` — PE2 perturbations
- `notebooks/15_eye_opening.ipynb` — stretch (EXP-3)
- `scripts/mida_label_to_t1w.py` — Bloch forward simulator (only if forward-simulation route chosen)
- `papers/tables/mida_pe1_transplants.tsv` — PE1 results
- `papers/tables/mida_pe2_perturbations.tsv` — PE2 results
- `papers/manuscript/manuscript.tex` — rewrite Sections 3 + 4 around new structure

## References checked 2026-05-26

- MIDA v1.0 — DOI `10.13099/ViP-MIDA-V1.0`, FDA + IT'IS Foundation 2015: <https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-0>
- MIDA v1.1 — DOI `10.13099/ViP-MIDA-V1.1`, FDA + IT'IS Foundation 2018: <https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-1>
- FDA MIDA regulatory-science tool page: <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>
- IT'IS Tissue Properties Database (for forward simulation): <https://itis.swiss/virtual-population/tissue-properties/database/>
- Bontempi et al. 2025, FaceAge, *The Lancet Digital Health*, DOI `10.1016/j.landig.2025.03.002`: <https://doi.org/10.1016/j.landig.2025.03.002>
- Rusak et al. 2022, quantifiable brain atrophy synthesis, *Medical Image Analysis*, DOI `10.1016/j.media.2022.102576`: <https://doi.org/10.1016/j.media.2022.102576>
- Larson and Oguz 2022, SynthAtrophy cortical-surface perturbation, *Frontiers in Neuroimaging*, DOI `10.3389/fnimg.2022.861687`: <https://doi.org/10.3389/fnimg.2022.861687>
- Smith et al. 2019, brain-age delta / age-bias correction, *Human Brain Mapping*, DOI `10.1002/hbm.24741`: <https://pubmed.ncbi.nlm.nih.gov/31201988/>
- [HYPOTHESIS.md](HYPOTHESIS.md) (2026-04-24) — the original confound test
- [FINDINGS.md](FINDINGS.md) (2026-04-24) — the SIMON longitudinal failure result
