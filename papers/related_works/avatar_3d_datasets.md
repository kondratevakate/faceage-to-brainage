# 3D Face Avatar Datasets and Internal Anatomy Modeling

Practical reference for the photo → 3D mesh direction, with explicit focus on our constraint: the MRI-derived target surface has **no hair** (T1 iso-surface stops at the skin) and **closed eyes** (supine position in scanner).

---

## 1. Public 2D + 3D face datasets

| Dataset | N | Age | 2D modality | 3D acquisition | Hair in mesh | Eyes | License | Age labels |
|---|---|---|---|---|---|---|---|---|
| **FaceScape** | 847 / 16,940 meshes | 16–70 | 68-cam rig | Multi-view photogrammetry | Loose hair | **Incl. "eyes closed" expression** | Academic, non-commercial | Yes |
| **Headspace / LYHM** | 1519 subj | 3–80+ | 5-view 3dMD | Structured light | **No — swim caps** | Open, neutral | Academic request | Yes |
| **NoW** | 100 subj | Adult | Various RGB | 3dMD | Yes | Neutral, open | MPI academic | No |
| **Stirling/ESRC** | 99 subj | Adult | 7 viewpoints | Di3D stereo | Yes | Open | Signed agreement | Partial |
| **BU-3DFE / BU-4DFE** | 100 / 101 | 18–70 | Frontal + texture | 3dMD | Yes | Open, emotions | Paid, non-commercial | Yes |
| **FaceWarehouse** | 150 × 20 expr | 7–80 | Kinect RGB | Kinect RGB-D | Yes | Some closed | Academic, author request | Yes |
| **FRGC v2.0** | 466 / 4950 | Adult | 2D stills | Minolta laser | Yes | Open | Notre Dame request | No |
| **Bosphorus** | 105 / 4666 | 18–45 | Single RGB | Structured light | Yes | Open | Academic | No |
| **4DFAB** | 180 over 5 yr | 5–75 | Multi-view RGB | 3dMD 4D (60 fps) | Yes | Open, blinks | Imperial ibug | Longitudinal |
| **FaMoS / TEMPEH** | 95 / 600k | Adult | Calibrated multi-view | FLAME-registered | Registered mesh excludes scalp | **28 sequences incl. blinks + closed eyes** | MPI academic | No |
| **DAD-3DHeads** | 44,898 images | In-the-wild | Single photo | Pseudo-GT FLAME | No (FLAME) | FLAME eyeballs | CC-BY-NC | No |
| **NeRSemble** | 220+ | Adult | 16-cam, 73 fps | Multi-view / FLAME | Yes | Open + blinks | TUM academic | No |
| **MICA training set** | ~2300 | Mixed | One image per subj | FLAME-topology from 8 source datasets | Mixed (LYHM = no hair) | Neutral | Per source | Per source |
| **Florence (MICC)** | 53 | Adult | HD + indoor + outdoor | 3dMD | Yes | Open | Academic | No |
| **Multiface (Meta)** | 13 | Adult | 40-view RGB | Multi-view mesh | Yes | Open | Meta research-only | No |
| **FaceSynthetics (MSFT)** | 100,000 synthetic | Synthetic | Rendered 2D | Procedural synthetic | Randomized | Open, random | Research-only | Synthetic |

---

## 2. Match to our constraint (no hair + closed eyes)

### Direct matches
- **Swim-cap / no-hair**: only **Headspace/LYHM** is designed around this constraint. All other academic datasets leave hair in the scan.
- **Closed eyes**: **FaceScape** (explicit "eyes closed" expression per subject), **FaMoS** (blink sequences), **BU-4DFE**, **NeRSemble**, **4DFAB** (spontaneous blinks).
- **Both at once, public**: **no exact match exists.**

### Medical/radiology alternatives
- **Radboud UMC orthognathic archive** — 3D stereophotos + CBCT paired; **not public**, collaborator-only.
- **Skull100** — 100 CT skull + face surface pairs; no photograph, no age labels.
- No large public dataset pairs a front-facing photograph with an MRI-derived skin surface of the same subject. **Genuine gap.**

### Practical options

**Option A — post-process Headspace** (strongest route)
- Scans already have no hair (swim cap).
- Close eyes analytically: fit FLAME to each scan (TEMPEH for multi-view, MICA for single-view) and re-mesh with the eyelid expression coefficient set to fully-closed.
- Result: paired 2D ↔ FLAME mesh where both hair and eye-closure match MRI targets.

**Option B — post-process FaceScape**
- Use the per-subject "eyes closed" capture.
- Remove hair via **Pixels2Points** (arXiv 2504.19718, 2025) — 2D + 3D feature fusion for per-vertex skin/non-skin classification, then crop scalp and reclose via FLAME head-model fit.

**Option C — synthetic augmentation**
- Render FLAME identities with eyes-closed blendshape active and no hair assets.
- Microsoft FaceSynthetics supports randomized hair and expression — modify to disable both.
- Unlimited paired 2D ↔ FLAME with MRI-like appearance; synthetic-to-real domain gap.

### Precedent: photo → MRI-derived face surface
- **No published paper trains a direct photo → MRI-iso-surface model.**
- Adjacent: FaceAge (photo → biological age scalar, not 3D), Skull-to-Face (CT skull → face, inverse direction), Knoops 2019 / ter Horst 2021 (CT-based soft-tissue prediction).
- This project direction appears **genuinely novel** as a photo → MRI-iso-surface mapping.

---

## 3. Internal anatomy modeling (muscles, fascia, fat pads) for surgery

### Multi-layer face models

| Model | Year | Layers | Code |
|---|---|---|---|
| Basel Face Model | 1999, 2009 | Skin surface only | Yes |
| FLAME | 2017 | Skin + articulated jaw/neck/eyeballs (rigid) | Yes |
| **Phace** (Ichim et al.) | SIGGRAPH 2017 | **Passive flesh + active muscles + rigid bone**, physics-based | Partial |
| Sifakis | ACM TOG 2005 | Muscle-activation from MoCap markers | Academic |
| Cong et al. | SCA 2015 | Automatic anatomical template generation | No |
| Animatomy (Disney) | SIGGRAPH Asia 2022 | Line-of-action strands | Closed |
| Chandran et al. (Disney) | ACM TOG 2024 | Generalized physics-based from scans | Proprietary |
| Implicit Physical Face Model | SIGGRAPH Asia 2023 | Neural implicit with muscle-like latents | No |

**FLAME alone has no mimetic muscles.** For any internal-anatomy application, a physics-based layer must be added on top.

### Surgical planning tools (input modality)

| Tool | Primary input | Accepts photo-only? |
|---|---|---|
| Materialise Mimics / ProPlan CMF | CT/CBCT + optional 3D stereophoto | No |
| Dolphin Imaging | CT/CBCT + cephalometrics | No |
| Mimics Enlight CMF / SurgiCase | CBCT | No |
| Knoops 2019 (Sci. Reports) | 3D stereophotos + CT | No |
| Lin 2022/2024 | 3D photogrammetry + CBCT | No |
| Ma 2023 (MICCAI) | CBCT soft-tissue meshes | No |

**No clinical or research system currently accepts a single 2D photograph to infer internal muscle geometry.** Every documented surgical planning pipeline requires volumetric imaging (CT/CBCT/MRI) or at minimum 3D stereophotogrammetry.

### Paired photo + CT/MRI + internal segmentation
- **Masticatory muscle MRI segmentation** (J. Cranio-Maxillofac. Surg. 2025): nnU-Net for masseter/temporalis/pterygoids, MRI only, no paired photograph.
- **TotalSegmentator**: 117+ CT structures including orbital bones and masseter — does **not** include mimetic muscles (orbicularis oculi, zygomaticus, frontalis).
- **No public dataset pairs photograph + CT/MRI + mimetic-muscle segmentation.** Clear gap.

### SOTA photo → muscle / fascia / fat pad
- **Photo → skull**: inverse direction solved (Skull-to-Face 2024); forward direction requires cephalometric X-ray.
- **Photo → mimetic muscle**: **open problem.** No peer-reviewed system does this from a single photograph. AR injector-training platforms (CBAM, Prollenium) overlay generic atlases on landmarks — atlas deformation, not subject-specific inference.
- **Photo → fat pad / SMAS**: not solved. Ann. Anat. 2025 provides reference geometry atlases; no dataset for subject-specific mapping.
- **Botox/filler planning**: automated pre/post classification exists (~89% accuracy, IRMHS 2024), but this is classification, not geometry inference.

---

## 4. Bottom line

**Data**:
1. Request **Headspace/LYHM** (swim caps, 1519 subjects, age-labeled) as the primary corpus.
2. Supplement with **FaceScape eyes-closed** subset (explicit closed-eye capture, age labels).
3. Build on top of **MICA training set** unification so labels are in FLAME topology from day one.
4. Add **Microsoft FaceSynthetics** (hair disabled, eyes forced closed) as synthetic augmentation.

**Build**: fine-tune a MICA/DECA-class photo → FLAME regressor where targets have been analytically modified to "no hair + eyes closed". Cheaper and less error-prone than letting the network learn the domain shift from mixed data. Mask the eye region explicitly in the training loss (FLAME eyeball still shows through some blendshape settings).

**Gaps for the project to own**:
1. No public paired photograph + MRI-derived face skin surface. Collecting this in a clinical collaboration is a defensible scientific contribution.
2. Internal-anatomy prediction (muscles, SMAS, fat pads) from a single photograph is an open problem. Any surgical-planning claim requires volumetric ground truth.
3. The MRI face-surface has no sclera — plan to mask the eye region explicitly in loss to avoid FLAME eyeball artefacts.

---

## Sources

- [FaceScape](https://facescape.nju.edu.cn/) · [GitHub](https://github.com/zhuhao-nju/facescape)
- [Headspace / LYHM (York)](https://www-users.york.ac.uk/~np7/research/Headspace/)
- [NoW Benchmark (MPI)](https://now.is.tue.mpg.de/)
- [FLAME (MPI)](https://flame.is.tue.mpg.de/) · [FLAME-Universe index](https://github.com/TimoBolkart/FLAME-Universe)
- [BU-3DFE / BU-4DFE](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
- [FRGC (NIST)](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc)
- [4DFAB (Imperial ibug)](https://ibug.doc.ic.ac.uk/resources/4dfab/)
- [TEMPEH / FaMoS](https://tempeh.is.tue.mpg.de/)
- [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads)
- [NeRSemble](https://tobias-kirschstein.github.io/nersemble/)
- [MICA (Zielonka, ECCV 2022)](https://github.com/Zielon/MICA)
- [Multiface (Meta)](https://arxiv.org/html/2207.11243v2)
- [Microsoft FaceSynthetics](https://github.com/microsoft/FaceSynthetics)
- [Pixels2Points skin segmentation (2025)](https://arxiv.org/abs/2504.19718)
- [Skull-to-Face (2024)](https://arxiv.org/html/2403.16207v2)
- [Phace: physics-based face (SIGGRAPH 2017)](https://lgg.epfl.ch/publications/2017/Phace/)
- [Learning a Generalized Physical Face Model (ACM TOG 2024)](https://dl.acm.org/doi/10.1145/3658189)
- [Animatomy (SIGGRAPH Asia 2022)](https://dl.acm.org/doi/10.1145/3550469.3555398)
- [Masticatory muscle MRI segmentation (2025)](https://www.sciencedirect.com/science/article/pii/S0901502725013189)
- [ProPlan CMF (Materialise)](https://www.materialise.com/en/healthcare/proplan-cmf)
- [Knoops et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC10037541/)
