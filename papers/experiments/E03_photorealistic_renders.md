# E03 — Photorealistic SD-augmented renders (negative result)

- **Author**: KK
- **Date**: 2026-04-15
- **Status**: data-saved, negative result (kept as a cautionary baseline)

## Question
Does generative photorealistic augmentation of MRI renders (passing the
plain render through Stable Diffusion image-to-image to add skin
texture, eye colour, hair) close the photo-vs-render domain gap and
reduce FaceAge MAE?

## Method
1. Render frontal MRI face as in [E01](E01_faceage_ixi_multiview.md).
2. Apply SD img2img with a face-preservation prompt to add realistic
   skin / pigmentation / hair.
3. Run FaceAge on the photorealistic output.
4. Compare against E01.

## Result
| metric | E01 plain renders | E03 SD-photorealistic |
|---|---|---|
| MAE | 11.34 yr | **19.91 yr** |
| RMSE | 13.83 yr | 21.12 yr |
| bias | +10.04 yr | **+19.91 yr** |

The bias *equals* the MAE — every prediction is systematically older.

## Interpretation
SD adds appearance features (skin texture, hair, scleral colour) that
look photorealistic to a human and to FaceAge — but FaceAge interprets
them as features of an *older* face. This is consistent with the
literature finding that **skin texture, facial contrast, and scleral
colour together account for 25–33 % of perceived facial age**
(González-Alvarez 2023, Hsieh 2023, Russell 2014). The generative model
does not condition on the underlying biological age, so it injects
"older-looking" appearance roughly uniformly.

This is a strong methodological warning: any future "photo →
3D-avatar → photo" pipeline must be evaluated against its naïve render
baseline — synthetic prettifying can *hurt*.

## What's left undone
1. Did not try alternative diffusion prompts conditioned on subject's
   chronological age (an obvious next ablation).
2. Did not test FAHR-FaceAge — the foundation-model successor with
   better OOD behavior may behave differently on synthetic renders.

## Pointers
- Numbers reported in [`midl-shortpaper.tex`](../midl2026/midl-shortpaper.tex)
- No CSV — only summary numbers retained
