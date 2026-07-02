# Single-Photo Avatar Baselines

Purpose: track which one-photo avatar/reconstruction methods are suitable for
the photo-vs-MRI pilot and what each method can actually prove.

## Current Local Status

Last checked: 2026-07-02 with
`scripts/photo_mri_avatar/sota_avatar_preflight.py`.
Cloud transfer instructions are in `scripts/photo_mri_avatar/CLOUD_RUNBOOK.md`.

Hardware/runtime finding: the current Windows environment has no visible
`nvidia-smi`. A separate CPU-only avatar environment has been prepared outside
the repo at `D:\projects\02_academia\_external\.venvs\avatar_cpu_py311` with
PyTorch CPU and DECA-era dependencies. Treat GPU Gaussian-avatar methods as not
locally runnable until a CUDA runtime is available.

| Method | Representation | Local status | Use in this pilot |
|---|---|---|---|
| MediaPipe FaceMesh | sparse-ish face landmarks/mesh | run complete | Pipeline sanity check and landmark baseline. |
| 3DDFA_V2 | dense BFM face mesh | run complete | First dense face-surface baseline. |
| DECA | FLAME mesh + detail | CPU geometry runner prepared, blocked | `deca_model.tar` is local; needs licensed FLAME `generic_model.pkl`. |
| MICA | metric FLAME face mesh | CPU geometry runner prepared, blocked | MICA checkpoint and InsightFace assets are local; needs licensed FLAME2020 `generic_model.pkl`. |
| EMOCA | expressive FLAME face mesh | cloned, blocked | Official path is old conda/PyTorch3D/CUDA-heavy; lower priority for Lenovo CPU. |
| LAM | animatable Gaussian head, optional mesh export | cloned, blocked | Strong perceptual/Gaussian avatar candidate; needs CUDA stack and LAM-20K weights. |
| GAGAvatar | animatable Gaussian head | cloned, blocked | One-shot Gaussian comparator; needs CUDA stack and model assets. |
| MeshLAM | animatable textured mesh head | not locally runnable | Best conceptual match for mesh + texture evaluation; project page currently links back to LAM rather than a separate runnable MeshLAM repo. |

## Source Basis

- 3DDFA_V2 official repo: <https://github.com/cleardusk/3DDFA_V2>
- DECA official repo: <https://github.com/yfeng95/DECA>
- MICA official repo: <https://github.com/Zielon/MICA>
- EMOCA official repo: <https://github.com/radekd91/emoca>
- LAM official repo/project: <https://github.com/aigc3d/LAM>,
  <https://aigc3d.github.io/projects/LAM/>
- MeshLAM paper/project: <https://arxiv.org/abs/2604.22865>,
  <https://meshlam.github.io>
- GAGAvatar official repo: <https://github.com/xg-chu/GAGAvatar>

## What Each Baseline Measures

MediaPipe and 3DDFA_V2 test whether a visible face surface can be recovered and
aligned to the MRI front region. They do not reconstruct a full head avatar.

DECA/MICA/EMOCA would test FLAME-family identity and face-shape recovery. These
are better geometry baselines than MediaPipe, but still depend on FLAME priors
and usually underrepresent hair, ears, and the back of the head.

LAM and GAGAvatar test modern perceptual avatar quality: one-shot animatable
Gaussian heads with realistic rendering. Their mesh/export artifacts can be
compared geometrically if exposed, but their core claim is
rendering/identity/perception, not MRI-level surface accuracy.

MeshLAM is the best target baseline for this project if runnable code appears:
it explicitly predicts a complete textured mesh from one image, so both geometry
and texture can be evaluated without converting from Gaussian splats.

## Preflight Result

Current machine status:

| Method | Runnable now | Main blocker |
|---|---:|---|
| 3DDFA_V2 | yes | None; already run. |
| DECA | no | Missing licensed FLAME `generic_model.pkl`; CPU geometry-only runner is ready. |
| MICA | no | MICA checkpoint, InsightFace assets, and CPU runner are ready; missing licensed FLAME2020 `generic_model.pkl`. |
| EMOCA | no | Missing EMOCA checkpoints and compatible old Python/PyTorch3D stack. |
| LAM | no | Missing torch/CUDA/NVIDIA runtime and LAM-20K weights. |
| GAGAvatar | no | Missing torch/CUDA/NVIDIA runtime and model checkpoints. |
| MeshLAM | no | No separate local runnable checkout/weights. |

Do not upload private face photos to public demos. For LAM/MeshLAM/GAGAvatar,
prefer local GPU or a private controlled GPU job.

## Next Experimental Decision

Do not rank methods by unconstrained ICP distance alone. The next metric upgrade
should add sparse anatomical constraints before point-to-surface scoring:

1. visible photo landmarks from the method output;
2. manually or semi-automatically selected MRI skin-surface landmarks for nose
   tip, chin, forehead/glabella, cheek contour, and ears when visible in MRI;
3. alignment seeded by those landmarks, followed by trimmed point-to-surface
   distances on semantically comparable regions.

After that, run one stronger baseline:

1. **DECA CPU geometry-only** after placing FLAME `generic_model.pkl`; this is
   now the fastest local geometry baseline for MRI comparison.
2. **MICA CPU geometry-only** after the same FLAME file is installed into the
   MICA checkout; it is the stronger metric-FLAME baseline.
3. **LAM/GAGAvatar** on a CUDA machine if perceptual Gaussian avatar quality is
   the priority; report geometry separately from rendering quality.
4. **MeshLAM** when a runnable code/weights release is available; this is the
   best target for a mesh+texture avatar baseline.
