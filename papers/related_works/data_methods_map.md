# Data and Methods Map

Living inventory of (a) datasets we have or want to acquire and (b) third-party models we benchmark against or build on.
Updated: 2026-05-26.

---

## 1. Local data inventory

Root: `C:\Projects\data\brain_images`

| Path | Content | N (T1w) | Modalities | State | Notes |
|---|---|---|---|---|---|
| `t1_only/ixi/images/` | Curated IXI T1w, one per subject | 561 | T1 | Ready | Filtered by `metadata/ixi_subject_resolution.csv` — drops scans w/ missing/ambiguous age (see `excluded_t1_missing_or_ambiguous_age.csv`) |
| `t1_only/ixi/metadata/` | `IXI.xls`, resolution + exclusion CSVs | — | — | Ready | Demographics + age provenance |
| `t1_only/simon/images/` | Curated SIMON T1w, one per session | 99 | T1 | Ready | Single subject sub-032633 across 99 sessions × multiple scanners |
| `t1_only/simon/metadata/t1_sidecars/` | BIDS JSON sidecars per T1w | 99 | — | Ready | Per-session acquisition params |
| `guest-20260316_094258/ixi/<ID>/<ID>_<Site>/{T1,T2,PD,MRA}/NIfTI/` | Raw multi-modal IXI dump | 584 subjects | T1/T2/PD/MRA + QC GIFs | Ready | The "source of truth" for IXI; `t1_only/` derives from it. Each subject has snapshot QC GIFs |
| `SIMON_data/SIMON_BIDS/sub-032633/ses-XXX/{anat,dwi,...}` | Full BIDS SIMON | 73 sessions | T1/T2/FLAIR/T2\*/PD/DWI/fMRI/ASL/SWI | Ready | `SIMON_pheno (4).csv` has age, site, scanner per session |
| `SIMON_data.tar.gz`, `SRPBS_TS_traveling_subjects.tar.gz` | Archives | — | — | Pending unpack | SRPBS Traveling Subjects = another multi-site multi-scan-of-same-subject set |
| `474881d9…5235680496a444e1bb1ddbaa9ccc8af8.zip` (634 MB, 2026-04-30) | Unknown | — | — | **Corrupted/truncated** | `unzip` reports no end-of-central-directory; `zipfile.BadZipFile`. Probably interrupted download — need to re-fetch from the source |
| **`MIDAv1-0.zip`** (483 MB, 2026-05-08) | **MIDA v1.0 whole-head phantom** | 1 subject, 116 labels | label-volume (0.5 mm³) + 115 STLs | **Ready (verified 2026-05-26)** | FDA + IT'IS Foundation. Local zip contains `MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii` and 115 `MIDA_v1.0/MIDA_v1_surfaces/*.stl` meshes. **Now central to the new strategy** — see [`STRATEGY_2026-05-20_post_mida.md`](../midl2026/STRATEGY_2026-05-20_post_mida.md). Includes Epidermis/Dermis, Subcutaneous Adipose Tissue, 25 facial muscles, eye sub-anatomy (Lens / Cornea / Sclera / Aqueous / Vitreous), 3 skull layers, full brain parcellation. Official pages: [MIDA v1.0](https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-0), [MIDA v1.1](https://itis.swiss/virtual-population/regional-human-models/mida-model/mida-v1-1), [FDA tool page](https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck) |

**Personal photo + MRI test set (5 subjects, n_personal):** not yet on disk. Broken zip from April 30 is plausibly this — re-fetch and confirm. See §4.

---

## 2. External test cohort: IXI Heads (itis.swiss Virtual Population)

Source: <https://itis.swiss/virtual-population/regional-human-models/ixi-heads>

| Property | Value |
|---|---|
| N subjects | **4** (2F: 29, 66 yr; 2M: 34, 67 yr) |
| Content | 60-tissue segmentation incl. eyes, deep brain, scalp layers, arteries/veins, salivary glands; DTI available |
| Format | **Sim4Life models** (proprietary, not NIfTI) |
| License | Sim4Life-only — not a free dataset |
| Access | Email `virtualpopulation@itis.swiss`; registration/payment unclear |
| Photos / textures | **No** — segmented anatomy only, no surface texture or photographs |
| Relevance for us | Useful as a **structural ground-truth surface** for the MRI-derived face render — eyes are explicitly modeled (we have closed-eyes problem). Limited as a quantitative test cohort (n=4) |

**Action**: write to `virtualpopulation@itis.swiss` to ask whether a research license that exposes the segmentation labels as NIfTI (without requiring Sim4Life runtime) is available, and whether they can confirm the underlying IXI subject IDs (so we can cross-link with the same subjects in our `t1_only/ixi/`).

---

## 3. Third-party methods — training data and applicability

### 3.1 SIAM — *Segment It All Model*

- **Repo**: <https://github.com/romainVala/SIAM>
- **Paper**: [arXiv:2605.02737v1](https://arxiv.org/abs/2605.02737) (4 May 2026) — *SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training*
- **Authors / affiliations**:
  - Romain Valabregue *(corresponding,* `romain.valabregue@upmc.fr`*),* Ines Khemir, Eric Badinet, Reuben Dorent — Sorbonne / ICM Paris Brain Institute
  - François Rousseau — IMT Atlantique, LaTIM INSERM U1101, Brest
  - Guillaume Auzias — Aix-Marseille, Institut de Neurosciences de la Timone
  - Auzias and Dorent share senior authorship
- **Task**: 16-class whole-head segmentation at 0.75 mm. Brain: WM, GM, cerebellar GM, CSF, ventricles, thalamus, putamen, pallidum, caudate, accumbens, amygdala, hippocampus (12 labels). Extra-cerebral: skin/epidermis, head fat, head muscle, salivary glands, air, mucosis, eye ball, skull bone, skull diploë, dura mater, vessel (11 labels) — derived from MIDA's 116 raw labels
- **Training data — exact composition of the 6 templates**:
  > **Note on MIDA confusion**: the public MIDA release (V1.1, DOI 10.13099/ViP-MIDA-V1.1, FDA + ITIS Foundation 2018) is **a single subject** — one head manually segmented into 116 labels at 0.5 mm³. SIAM's "6 templates" = MIDA's one subject + **5 new subjects acquired by the SIAM authors at Paris Brain Institute** and labelled following the MIDA tissue convention. The 5 new ones are *not* in the public MIDA release.
  1. **MIDA template** (N=1) — Iacono et al. 2015 (V1.1 released 2018). **Publicly available**, 0.5 mm³, 116 manual labels. *The only public one of the six.*
  2. **Skull templates** (N=3) — Paris Brain Institute, Bancel et al. 2025. CT (GE Discovery 750 HD, 0.49×0.48×0.62 mm³) + multi-contrast MRI (Siemens Prisma 3T: MP2RAGE 1 mm³, UTE 0.6 mm³, FLAIR 1 mm³), co-registered to UTE @ 0.6 mm³. **Not public** — APHP190407 / IDRCB 2019-A01791-56 / NCT04074031. Remaining 9 subjects from the same cohort form the **private skull test set**
  3. **Vasculature templates** (N=2) — Paris Brain Institute, Siemens CIMAX 3T. MPRAGE + Dixon (water/fat) + T2-SPACE + phase-contrast MRI, all 0.7 mm³ → resampled to 0.5 mm³. **Not public** — IDRCB 2021-A02404-37
- **Synthetic generator**: SynthSeg-family domain randomization extended to **shape domain** — TorchIO affine + elastic at 0.25 mm³ + novel cortical-thickness modulation. Image-generation model follows Billot et al. 2023
- **Evaluation cohorts (8 datasets, N=301)** — every dataset is treated as OOD:
  | # | Name | N | Modality / resolution | Reference |
  |---|---|---|---|---|
  | 1 | **MICCAI 2012** | 20 | T1w 1 mm | Landman & Warfield 2012 — manual, Neuromorphometrics/BrainCOLOR |
  | 2 | **Mindboggle** | 101 | T1w 1 mm, multi-scanner | Klein & Tourville 2012 — FreeSurfer 5.0 silver-standard |
  | 3 | **DBB** | 37 | T1w pediatric (1–18 yr), congenital/acquired abnormalities | Amorosino et al. 2022 — ITK-SNAP active contours + manual correction |
  | 4 | **Ultracortex** | 12 | T1w 9.4T, 0.6 mm MPRAGE/MP2RAGE | Mahler et al. 2025 — high-quality manual GM/WM |
  | 5 | **HCP test-retest** | 41 × 2 = 82 | T1w + T2w 0.7 mm @ 3T | Van Essen et al. 2013 — FreeSurfer v7.4.1 -hires REF |
  | 6 | **dHCP neonates** | 20 | T1w + T2w 0.5 mm | Edwards et al. 2022 — drawEM labels |
  | 7 | **SynthAtrophy** | 20 × 7 = 140 (uses ADNI subj) | GAN-simulated atrophied T1w | Rusak et al. 2022 |
  | 8 | **Skull private test** | 9 | T1w (UNI) + FLAIR + UTE + CT @ 0.6 mm | Paris Brain Institute (held-out from Skull templates) |
- **Inference modalities**: T1, T2, FLAIR, CT — contrast-agnostic by design
- **Three model checkpoints**:
  - Model 1 — 39 regions, trained from 3 subjects (legacy)
  - Model 2 — 12 brain + extra-cerebral, trained from 6 templates
  - Model 3 — extends Model 2 with anomaly robustness (default)
- **Usefulness for faceage-to-brainage**: extracts skull + scalp + brain tissue boundaries → directly feeds face-render mesh and gives us per-tissue control over what gets included in the visible face surface. *Critically: SIAM's training is fully synthetic, so applying it to our IXI/SIMON/personal-cohort data introduces no training-data overlap.*
- **Public datasets we should also download** (to replicate or extend SIAM's eval): **MICCAI 2012**, **Mindboggle**, **DBB**, **Ultracortex**, **dHCP**. HCP requires CRCNS data-use agreement. ADNI (for SynthAtrophy) requires application

### 3.2 NeuroFM

- **Project**: <https://rocknroll87q.github.io/NeuroFM/> · <https://github.com/rockNroll87q/NeuroFM> · medRxiv DOI `10.64898/2026.03.27.26349489`
- **Paper**: *NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation* (Dibble et al., medRxiv March 2026)
- **Authors**: Austin Dibble, Connor Dalby, Michele Sevegnani, Alessio Fracasso, Donald M Lyall, Monika Harvey, Michele Svanera *(senior,* `Michele.Svanera@glasgow.ac.uk`*).* University of Glasgow, School of Psychology & Neuroscience (Muckli Lab); rocknroll87q ≡ Michele Svanera
- **Task**: brain-MRI foundation model; learns morphometric + demographic representations; transfers to clinical, cognitive, developmental, socio-behavioural, and QC tasks via linear probes
- **Training data**: **100,000 healthy synthetic volumes** from LDM100k, generated from a UK Biobank-derived population model according to the [NeuroFM Hugging Face model card](https://huggingface.co/NeuroAI-UofG/NeuroFM). No diagnostic labels were used in pretraining, but demographic bias may inherit from the UK Biobank source distribution.
- **Evaluation / acknowledgement overlap**: the medRxiv v2 full text and Hugging Face card acknowledge multiple real cohorts, including **IXI**, **SIMON**, **HCP**, **OASIS-3**, ADNI, NIFD, FCON1000, UK Biobank, and others. Therefore, do **not** claim IXI/SIMON as unseen external tests for NeuroFM without clarifying training-vs-evaluation usage.
- **Inputs**: **T1-weighted MRI only** (T2/FLAIR would require finetuning)
- **Three sizes — confirmed parameter counts (`docs/models.md`)**:
  - **NeuroFM-S — 484k params** (<10 MB)
  - **NeuroFM-M — 6.5M params**
  - **NeuroFM-L — 10.8M params** (~150 MB)
- **Usefulness for faceage-to-brainage**: NeuroFM-L is the right size for our compute budget and is a plausible drop-in replacement for SFCN's encoder. The model card reports T1w-only input, skull-stripped NIfTI, 1 mm isotropic resampling, and built-in outputs including brain age. **Critical caveat**: IXI and SIMON appear in the acknowledged/evaluated real-cohort set, so NeuroFM should be treated as a benchmark/backbone, not as a clean external-validation result, unless Svanera confirms the exact split and exposure.

### 3.3 Avatar / face-reconstruction methods (photo → 3D head)

These three are **not MRI-aware**, but they are the closest "reverse direction" of our pipeline — they reconstruct head geometry/identity from photos. We need them on the map for two reasons:
1. **Identification risk model** — if any of them could be inverted with our MRI-derived face surface, they bound the worst-case re-identification attack against non-defaced T1s
2. **Photo-side ground truth** — if we collect a personal photo+MRI test set, these provide the photo→mesh path we compare against the MRI→mesh path

| Method | arxiv | Input | Output | Training data | MRI-aware? |
|---|---|---|---|---|---|
| **DenseMarks** | [arXiv:2511.02830](https://arxiv.org/abs/2511.02830) | Single head photo | Dense 3D canonical-cube embedding per pixel (correspondences) | In-the-wild talking-head videos + point-tracker pseudo-GT + face landmarks + segmentation | No |
| **MATCH** | [arXiv:2603.15811](https://arxiv.org/abs/2603.15811) | Calibrated multi-view head images | Gaussian-splat avatar in dense semantic correspondence / UV-like template space (reported 0.5 s/frame) | Verify in full paper before citing exact training-set composition | No |
| **MeshLAM** | [arXiv:2604.22865](https://arxiv.org/abs/2604.22865), [project page](https://meshlam.github.io/) (CVPR 2026) | **Single** photo | Animatable textured head mesh in one forward pass | Project page reports a FLAME-based prior; verify training-set composition before use | No |

---

## 4. Planned cohort matrix

Sketch of how each dataset is used in the experimental design.

| Cohort | N | Role | Distribution | Defaced? | Photo? | Notes |
|---|---|---|---|---|---|---|
| IXI (t1_only) | 561 | Train + in-domain test | Train cohort | No | No | Primary training distribution |
| SIMON | 99 sessions | Intra-subject scanner-variance probe | OOD (multi-scanner, single subject) | No | No | "Same brain, many scanners" |
| SRPBS Traveling Subjects | TBD | Multi-site harmonization probe | OOD (multi-site) | Per source | No | Unpack archive first |
| IXI Heads (itis.swiss) | 4 | Surface ground-truth qualitative case study | OOD (high-detail segmented) | N/A (no face render generated by us) | No | Pending license clarification |
| Personal test set | 5 | OOD with paired photo | OOD (self-collected) | No | **Yes** | Critical for face-render vs photo agreement; **re-fetch broken zip** |

**TODO — user input needed (see §5):** rank the cohorts above by priority for the MIDL resubmission's "external validation" section.

---

## 5. Open decisions

> **Strategy reframe 2026-05-20.** With MIDA acquired, the project pivots from "external test cohorts validate the IXI correlation" to **"a single in-silico phantom dissociates the two predictors"**. Decisions 1–4 below are now *secondary* to the four strategic decisions in `papers/midl2026/STRATEGY_2026-05-20_post_mida.md`. Public eval cohorts in §6 fall in priority — we no longer need them to validate dissociation, only to position the IXI confound finding.

1. **IXI Heads usage** — qualitative case study (n=4) or skip? Depends on whether itis.swiss exposes labels outside Sim4Life. *Lower priority now: MIDA already covers what IXI Heads was meant to.*
2. **NeuroFM as backbone** — adopt for brain-age or keep SFCN? *Lower priority: SynthBA already gives a working brain-age. NeuroFM is a "next paper" optimisation.*
3. **Personal cohort photo protocol** — frontal only (matches MRI render orientation) or also profile? *Still required for the OOD validation, but only after EXP-0 confirms MIDA renders are meaningful.*
4. **Avatar method baseline** — pick one of DenseMarks / MATCH / MeshLAM. *Defer: avatar comparison is a follow-up, not a MIDL story.*

---

## 6. Datasets to download (gap list)

Public datasets used by SIAM (and likely by NeuroFM) that we **do not** have locally:

| Dataset | N | Modality | Source | Why | Effort |
|---|---|---|---|---|---|
| **MIDA template** | 1 (whole-head, 116 labels @ 0.5 mm) | Label volume + CAD/STL tissue surfaces | <https://itis.swiss/virtual-population/regional-human-models/mida-model> | The single public SIAM template — already downloaded locally as `MIDAv1-0.zip`; useful for phantom/protocol work rather than an external cohort | Done locally; keep ITIS email only for license/derived-render questions |
| **MICCAI 2012 challenge** | 20 T1w + manual seg | T1w 1 mm | Neuromorphometrics, commercial distribution | OOD evaluation cohort that SIAM, FastSurfer and many others report on | Paid for full set; public 35-subject teaser via OASIS/MICCAI archive |
| **Mindboggle-101** | 101 T1w | T1w 1 mm, multi-scanner | <https://mindboggle.info/data.html> | Public, widely used OOD cohort. Useful but **not** manually labeled for deep nuclei (FreeSurfer silver-standard) | Free, ~5 GB |
| **DBB (Pediatric Brain Bench)** | 37 T1w pediatric | T1w | Amorosino et al. 2022, OpenNeuro | Anatomical-deformation OOD cohort | Free via OpenNeuro |
| **Ultracortex** | 12 T1w 9.4T | T1w MPRAGE/MP2RAGE 0.6 mm | Mahler et al. 2025 | The cleanest manual cortical GM reference. Tiny (~10 GB) | Free, registration |
| **dHCP** | 783 neonates (we'd use ~20) | T1w + T2w 0.5 mm | <https://biomedia.github.io/dHCP-release-notes/> | Neonatal OOD. Useful only if we want lifespan generalization | Registration + DUA |
| **HCP 1200** | 1113 (we'd use the 41-subj test-retest subset) | T1w + T2w 0.7 mm 3T | <https://db.humanconnectome.org/> | The canonical adult-MRI test-retest cohort. Already in SIAM's eval — overlap risk with NeuroFM | Free via CRCNS DUA |
| **SRPBS Traveling Subjects** | already on disk in tarball | Multi-site multi-scan-of-same-subject | Tanaka et al. 2021, NIMS | Multi-site harmonization probe. Complementary to SIMON | **Unpack the existing tarball** |
| **OASIS-3** | 1098 longitudinal | T1w + multi-modal | <https://www.oasis-brains.org/> | Aging / dementia cohort; pairs well with NeuroFM's dementia-risk claim | Free, application |
| **Cam-CAN** | 656 (18–88 yr) | T1w + T2w + DWI + fMRI | <https://www.cam-can.org/> | Adult-lifespan cohort with paired demographics — good external test for brain-age | Free, application |

**Decision rule**: download the **public + free** ones first (Mindboggle, DBB, Ultracortex, OpenNeuro datasets); request agreements for HCP, dHCP, OASIS-3, Cam-CAN in parallel; defer MICCAI 2012 unless we need it for a specific benchmark comparison.

---

## 7. Author outreach plan

Goal: ask the first/senior authors of methods we depend on (a) how *they* would write a "segment-everything for face-age in MRI" paper and (b) confirm dataset details that remain unclear after the 2026-05-26 source check.

Drafts live in [`papers/outreach/`](../outreach/) — one file per recipient. Each draft has a `TODO: personalize` block where you write a 5–10 line personal intro / specific anatomical question.

| Recipient | Role | Why ask | Draft file |
|---|---|---|---|
| Romain Valabregue (corresponding, SIAM) | Author of the whole-head synthetic-training paper | Get the synthetic generator code, ask about face-skin label fidelity for our face-render task | `papers/outreach/email_valabregue_siam.md` |
| Reuben Dorent (senior, SIAM) | Co-senior, ICM Paris | Same project, second contact in case Valabregue is slow | (mentioned in Valabregue draft) |
| Michele Svanera (senior, NeuroFM) | rocknroll87q, Glasgow | Confirm exact LDM100k generator citation, IXI/SIMON exposure level, and NeuroFM-L finetuning protocol for age regression | `papers/outreach/email_svanera_neurofm.md` |
| Malte Prinzler (first, MATCH) | MPI / ETH Zürich | Ask about closed-eyes / no-hair head meshes — does MATCH's UV-space registration tolerate our MRI-derived target? | `papers/outreach/email_prinzler_match.md` |
| Yisheng He (first, MeshLAM) | Industry (Salesforce?) | Single-photo one-shot mesh — ask if it can ingest an MRI-derived mesh as target supervision | `papers/outreach/email_he_meshlam.md` |
| ITIS Foundation | virtualpopulation@itis.swiss | IXI Heads license + NIfTI export feasibility | `papers/outreach/email_itis_ixiheads.md` |

---

## 8. Provenance

- Local inventory generated from `ls C:\Projects\data\brain_images` on 2026-05-18
- SIAM details checked against arXiv: <https://arxiv.org/abs/2605.02737>
- NeuroFM input requirements, model sizes, synthetic LDM100k training note, and caveats checked against the Hugging Face model card: <https://huggingface.co/NeuroAI-UofG/NeuroFM>
- NeuroFM real-cohort overlap checked against medRxiv v2 full text: <https://www.medrxiv.org/content/10.64898/2026.03.27.26349489v2.full-text>
- MIDA v1.0/v1.1 and FDA tool description checked against IT'IS/FDA pages linked above.
