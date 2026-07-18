# 3D Avatar Quality, One-Shot Reconstruction, and Perceptual Realism

Updated: 2026-06-21.

## 1. Purpose

This note explains why recent human/head-avatar research is moving toward
Gaussian splatting, hybrid mesh-Gaussian representations, learned texture
models, and feed-forward one-shot reconstruction. The project question is not
"can an avatar look good?" but whether avatar methods help or mislead the
MRI-derived face-age branch.

For `faceage-to-brainage`, the important separation is:

| Claim | Meaning for this project |
|---|---|
| Perceptual realism | The rendered head looks plausible to a human observer or an image metric. |
| Geometric fidelity | The recovered surface is metrically close to the subject's real face/head geometry. |
| Identity fidelity | The avatar preserves the person-specific identity signal. |
| Age-signal preservation | The representation preserves cues that a face-age model or human age rater uses. |

The avatar literature increasingly optimizes the first three claims. Our paper
needs the fourth, and must not infer it from visual realism alone.

## 2. Source Inventory

Local canonical context:

| Source | Authority | Freshness | Role |
|---|---|---|---|
| `README.md` | authoritative | confirmed | Project purpose, pipeline, core tension: photo-age cues are least grounded in MRI. |
| `papers/related_works/avatar_3d_datasets.md` | working | confirmed 2026-05 | Dataset gap: no public paired photo + MRI-derived skin surface. |
| `papers/related_works/data_methods_map.md` | working | confirmed 2026-05 | Current method map: DenseMarks, MATCH, MeshLAM as photo-to-head adjacent baselines. |
| `papers/related_works/sota_design.md` | working | confirmed 2026-05 | Evaluation discipline for face-age/brain-age claims. |

External sources checked for this note:

| Source | Authority | Freshness | Use |
|---|---|---|---|
| Kerbl et al. 2023, 3D Gaussian Splatting | authoritative | confirmed | Core reason 3DGS became attractive: high visual quality with real-time rendering. |
| NeRSemble benchmark, v1/v2 2025-2026 | authoritative | confirmed | Shows field-level move toward held-out views, monocular avatar reconstruction, and single-view 3D face reconstruction. |
| LAM, SIGGRAPH 2025 / arXiv 2502.17796 | authoritative/working | confirmed | One-shot animatable Gaussian head from one image using FLAME canonical queries. |
| MATCH, CVPR 2026 / arXiv 2603.15811 | authoritative | confirmed | Multi-view Gaussian registration in dense semantic correspondence; 0.5 s/frame. |
| MeshLAM, CVPR 2026 / arXiv 2604.22865 | working | confirmed | One-shot textured mesh avatar; explicit texture map as counterpoint to pure Gaussians. |
| OMEGA-Avatar, arXiv 2602.11693 | working | tentative | One-shot full-head Gaussian avatar with diffusion-generated multi-view RGB/normal guidance. |
| MVCHead, CVPR 2026 / arXiv 2605.25220 | working | tentative | Learns 3D Gaussian heads from 2D image collections without 3D/multi-view supervision. |
| HRAvatar, CVPR 2025 / arXiv 2503.08224 | working | confirmed | Relightable/material-aware Gaussian head avatars. |
| RelightAnyone, arXiv 2601.03357 | working | tentative | Generalized relightable Gaussian head model fitted from single or multi-view images. |
| Cutler et al. 2024/2025, photorealistic avatar QoE | authoritative/working | confirmed | Objective metrics alone correlate weakly with most subjective avatar-quality dimensions. |

## 3. World Model

The avatar community is optimizing for a product-shaped target: a person should
be captured quickly, rendered in real time, animated from standard controls, and
look believable under new viewpoint, expression, and lighting. This creates a
different objective from medical shape measurement.

The key stakeholders have different definitions of "quality":

| Stakeholder | Primary quality target | Typical evidence |
|---|---|---|
| Avatar/telepresence user | Looks like a believable person in motion. | Human preference, realism, trust, comfort, creepiness, emotion accuracy. |
| Graphics researcher | Novel-view and reenactment quality. | PSNR, SSIM, LPIPS, FID/FVD, temporal metrics, qualitative videos. |
| Face-reconstruction researcher | Shape, identity, and expression control. | CSIM, AKD, AED/APD, Chamfer, normals, F-score, held-out expressions. |
| This project | MRI-derived face signal is real, stable, and age-relevant. | Same-scan age correlation, scanner reliability, paired photo/mesh validation, shape-vs-texture ablation. |

Open loops for this project:

1. We do not yet have a paired photo + MRI + 3D face ground truth cohort.
2. MRI carries head shape, fat compartments, bone/orbital geometry, and scalp
   surface, but not skin texture, pigmentation, sclera, hair, or normal photo
   illumination.
3. One-shot avatar methods can plausibly fill missing texture and back-of-head
   content, but that filling is a learned prior, not measured evidence.
4. Therefore avatar methods are useful as priors and baselines, not as direct
   validation of the MRI-face age signal.

## 4. Distillation

### 4.1 Why the field moved to Gaussian splatting

3D Gaussian Splatting offered a practical bridge between mesh graphics and NeRFs.
It keeps an explicit 3D representation that can be rasterized efficiently, while
retaining enough volumetric softness to model hair, skin boundaries, view-
dependent appearance, and imperfect surfaces. The original 3DGS paper targeted
real-time high-quality novel-view synthesis by optimizing anisotropic Gaussians
and using a visibility-aware renderer.

For human avatars this matters because faces are not clean CAD objects. Human
observers are highly sensitive to soft boundaries, eye/mouth detail, hair,
skin texture, expression timing, and lighting. A clean mesh with a UV texture
is controllable, but can look synthetic. A pure NeRF can look strong, but is
often slower and harder to edit or animate. A Gaussian layer attached to a
template head gives the field a useful compromise.

### 4.2 Why the current direction is hybrid, not "Gaussians replace meshes"

The newest head-avatar papers usually combine four components:

| Component | Why it is used |
|---|---|
| FLAME/SMPL-X prior | Keeps shape and expression in a controllable anatomical space. |
| Gaussian appearance layer | Captures soft, high-frequency, view-dependent appearance in real time. |
| UV/canonical correspondence | Makes editing, expression transfer, identity interpolation, and tracking possible. |
| Diffusion or large-image prior | Hallucinates unseen views, hair, ears, back of head, and missing texture from one/few images. |

This is why MATCH is important: it predicts Gaussian splat textures in a fixed
UV layout, so Gaussians are semantically corresponding across subjects and
expressions. It is not only making nicer renders; it is making avatars editable
and comparable.

MeshLAM is the useful counterexample. It argues that pure Gaussian heads can
need many primitives to store fine texture, while explicit UV texture maps can
store high-frequency details more efficiently and preserve topology. Its best
result is actually hybrid: a mesh prior improves a Gaussian reconstruction. The
research direction is therefore not "mesh versus Gaussian" but "mesh for
structure and control; Gaussians/neural texture for perceptual detail."

### 4.3 How avatar papers measure quality

It is not only judged by eye. Qualitative videos remain central because human
avatars are perceptual objects, but strong papers now report several metric
families:

| Evaluation layer | Metrics used in recent papers | What it can and cannot prove |
|---|---|---|
| Render fidelity | PSNR, SSIM, LPIPS, FID/FVD, DreamSim | Captures image similarity or distributional realism; does not prove true 3D anatomy. |
| Identity | CSIM / ArcFace cosine similarity | Measures photo-recognition identity preservation; can be fooled by texture and face crops. |
| Expression/pose | AED, APD, AKD, landmark/keypoint distance | Measures driving control; depends on the estimator and 2D landmarks. |
| Geometry | Chamfer, point-to-surface distance, normal consistency, F-score | Best evidence for surface accuracy, but requires 3D ground truth. |
| Temporal behavior | flicker metrics, FVD, video perceptual metrics, benchmark-specific temporal scores | Captures stability in motion; not enough for static anatomical validity. |
| Human perception | MOS, pairwise preference, realism, trust, comfort, creepiness, resemblance, emotion accuracy | Directly addresses user perception; costly and scenario-specific. |

The Cutler et al. avatar QoE study is the clearest warning: standard objective
metrics such as PSNR, SSIM, LPIPS, FID, and FVD correlate weakly with most
subjective dimensions of avatar quality. That means an avatar can improve on
benchmarks without necessarily becoming more trusted, comfortable, or person-
like to humans.

### 4.4 When one-photo avatars become perceptually convincing

One-photo avatars are already close for constrained use cases: frontal talking
head, moderate expressions, stable lighting, small viewpoint changes, and no
demand for biometric correctness outside the visible face. LAM, OMEGA-Avatar,
OMG-Avatar, MeshLAM, and MVCHead all show the same convergence: one/few-shot,
feed-forward, animation-ready reconstruction is now the main research target.

The hard boundary is information-theoretic. One photo does not contain the true
back of the head, ears, hair volume, skin microgeometry, teeth, tongue, or
material response under new lighting. Modern methods can produce plausible
answers using priors, but those priors are guesses. Perceptual plausibility will
arrive before subject-specific fidelity.

Working forecast:

| Time horizon | Likely state | Caveat |
|---|---|---|
| 2026-2027 | One-photo avatars good enough for low-stakes social, effects, simple telepresence, and research demos. | Side/back views, hair, glasses, teeth, and lighting still show failures. |
| 2027-2029 | Few-shot capture or short phone video becomes broadly convincing to non-expert users in normal conditions. | The extra views solve the missing-information problem better than larger priors alone. |
| Beyond that | Single-photo all-view "indistinguishable from real video" may be possible perceptually in common cases. | It will still be hallucinated, so it is not ground-truth geometry or biology. |

## 5. Implications for MRI-Derived Face Age

The strongest position for this project is to say that avatar research explains
how to make faces look real, but also why "looks real" is insufficient for an
MRI-derived biological-age claim.

MRI gives us deterministic measured geometry of the head surface. Avatar models
give us learned perceptual priors for texture, hair, eyes, and missing views.
Combining them can make better pictures, but it can also inject non-MRI age cues.
That matters because face-age models are likely sensitive to skin texture,
periocular appearance, sclera, pigmentation, wrinkles, hairline, and lighting.

Recommended validation matrix:

| Project claim | Current or plausible evidence | Missing evidence | Best next metric |
|---|---|---|---|
| MRI surface can produce recognizable face-like renders. | Deterministic marching-cubes/PyVista renders; face detector success rate; qualitative panel. | Human recognizability against real photos. | Face recognition CSIM if paired photos exist; human forced-choice if not. |
| MRI mesh preserves subject-specific shape. | Same-scan surface extraction and stable camera render. | External surface ground truth. | Chamfer / point-to-surface distance to photogrammetry, CBCT/STL, or high-quality photo-derived mesh. |
| MRI mesh preserves age-relevant morphology. | Chronological-age regression from render/mesh features; correlation with brain-age gap. | Independent age cue validation. | Shape-only age model, anthropometric distances, age-stratified calibration. |
| Texture is the missing modality. | MRI has no skin color/texture/sclera/hair. | Quantified shape-vs-texture contribution. | Compare MRI shape-only render, photo-derived avatar, and texture-augmented avatar on the same subjects. |
| Avatar prior improves appearance without corrupting biology. | Better LPIPS/FID/human realism after texture prior. | Proof that the prior did not add age signal from training distribution. | Age prediction before/after prior, with paired photo baseline and ablation of hallucinated regions. |

The reviewer's likely question is: "Is this a measured MRI signal or an avatar
hallucination?" The answer should be designed into the experiments:

1. Report MRI-only results first.
2. Treat avatar/texture priors as an ablation, not the primary evidence.
3. Mask or separately analyze regions MRI cannot support: eyes/sclera, hair,
   skin texture, beard, teeth, and specular highlights.
4. If paired photos are collected, collect at least frontal plus profile views
   or a short phone video, not only one frontal photo.
5. Where possible, compare to a geometry target: photogrammetry, FLAME/MICA fit,
   or manually checked sparse landmarks.

## 6. Where the Research Is Going

The field is moving from reconstruction to generative reconstruction:

1. Feed-forward one/few-shot avatars instead of per-subject optimization.
2. Full-head completion: hair, ears, neck, back of head, accessories.
3. Dense semantic correspondence so avatars are editable and comparable.
4. Hybrid mesh-Gaussian representations, not purely one or the other.
5. Relightable/material-aware avatars with albedo, normals, roughness, and
   environment lighting.
6. Weakly supervised multi-view consistency from 2D image collections, diffusion
   priors, or self-render critics.
7. Human-centered evaluation because image metrics alone miss trust, comfort,
   creepiness, and resemblance.

For this project, the actionable interpretation is conservative: use avatar
methods to understand the upper bound of perceptual realism and to design
photo-side baselines, but keep the scientific claim anchored to MRI-measured
geometry, age prediction, and reliability.

## 7. Candidate Next Actions

1. Add a short subsection to the manuscript discussion: "Perceptual realism is not
   anatomical fidelity."
2. Add a methods table that maps each claim to a metric and ground truth source.
3. Ask MATCH/MeshLAM authors which geometry-registration metrics they would
   trust for no-hair, closed-eye MRI head meshes.
4. For the planned personal cohort, collect a short phone video or at least
   frontal + left/right profile photos, because one frontal photo cannot validate
   side/back geometry.

## 8. Sources

- Kerbl et al. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. https://arxiv.org/abs/2308.04079
- NeRSemble benchmark. https://kaldir.vc.cit.tum.de/nersemble_benchmark/
- He et al. 2025. LAM: Large Avatar Model for One-shot Animatable Gaussian Head. https://arxiv.org/html/2502.17796v1
- Prinzler et al. 2026. MATCH: Feed-forward Gaussian Registration for Head Avatar Creation and Editing. https://arxiv.org/abs/2603.15811
- MeshLAM 2026. Feed-Forward One-Shot Animatable Textured Mesh Avatar Reconstruction. https://arxiv.org/html/2604.22865v1
- OMEGA-Avatar 2026. One-shot Modeling of 360-degree Gaussian Avatars. https://arxiv.org/html/2602.11693v1
- MVCHead 2026. Multi-view Consistent 3D Gaussian Head Avatars without Multi-view Generation. https://humansensinglab.github.io/MVCHead/
- HRAvatar 2025. High-Quality and Relightable Gaussian Head Avatar. https://arxiv.org/html/2503.08224v2
- RelightAnyone 2026. A Generalized Relightable 3D Gaussian Head Model. https://arxiv.org/html/2601.03357v1
- Cutler et al. 2024/2025. A multidimensional measurement of photorealistic avatar quality of experience. https://arxiv.org/html/2411.09066v3
