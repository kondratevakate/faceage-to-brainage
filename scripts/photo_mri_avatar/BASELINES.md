# Single-Photo Avatar Baselines

Purpose: track which one-photo avatar/reconstruction methods are suitable for
the photo-vs-MRI pilot and what each method can actually prove.

## Current Local Status

| Method | Representation | Local status | Use in this pilot |
|---|---|---|---|
| MediaPipe FaceMesh | sparse-ish face landmarks/mesh | run complete | Pipeline sanity check and landmark baseline. |
| 3DDFA_V2 | dense BFM face mesh | run complete | First dense face-surface baseline. |
| DECA | FLAME mesh + detail | blocked | Needs FLAME/model assets before local inference. |
| MICA | metric FLAME face mesh | blocked | Needs FLAME/MICA assets; strong geometry candidate after setup. |
| EMOCA | expressive FLAME face mesh | blocked | Needs FLAME/EMOCA assets; useful for expression robustness. |
| LAM | animatable Gaussian head, optional mesh export | cloned, not runnable locally yet | Strong perceptual/Gaussian avatar candidate; needs CUDA stack and weights. |
| MeshLAM | animatable textured mesh head | paper/project only in current pass | Best conceptual match for mesh + texture evaluation once code/weights are released. |

## Source Basis

- 3DDFA_V2 official repo: <https://github.com/cleardusk/3DDFA_V2>
- DECA official repo: <https://github.com/yfeng95/DECA>
- MICA official repo: <https://github.com/Zielon/MICA>
- EMOCA official repo: <https://github.com/radekd91/emoca>
- LAM official repo/project: <https://github.com/aigc3d/LAM>,
  <https://aigc3d.github.io/projects/LAM/>
- MeshLAM paper/project: <https://arxiv.org/abs/2604.22865>,
  <https://meshlam.github.io>

## What Each Baseline Measures

MediaPipe and 3DDFA_V2 test whether a visible face surface can be recovered and
aligned to the MRI front region. They do not reconstruct a full head avatar.

DECA/MICA/EMOCA would test FLAME-family identity and face-shape recovery. These
are better geometry baselines than MediaPipe, but still depend on FLAME priors
and usually underrepresent hair, ears, and the back of the head.

LAM tests modern perceptual avatar quality: a one-shot animatable Gaussian head
with realistic rendering. Its mesh export can be compared geometrically, but the
core claim is rendering/identity/perception, not MRI-level surface accuracy.

MeshLAM is the best target baseline for this project if runnable code appears:
it explicitly predicts a complete textured mesh from one image, so both geometry
and texture can be evaluated without converting from Gaussian splats.

## Next Experimental Decision

Do not rank methods by unconstrained ICP distance alone. The next metric upgrade
should add sparse anatomical constraints before point-to-surface scoring:

1. visible photo landmarks from the method output;
2. manually or semi-automatically selected MRI skin-surface landmarks for nose
   tip, chin, forehead/glabella, cheek contour, and ears when visible in MRI;
3. alignment seeded by those landmarks, followed by trimmed point-to-surface
   distances on semantically comparable regions.

After that, run one stronger baseline:

1. MICA/DECA/EMOCA if FLAME/model assets are available;
2. LAM on a CUDA machine if perceptual Gaussian avatar quality is the priority;
3. MeshLAM when code/weights become available.
