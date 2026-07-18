# Email — Romain Valabregue (SIAM)

**To**: romain.valabregue@upmc.fr
**CC**: Reuben Dorent (ICM Paris, co-senior author)
**Subject**: SIAM applied to non-defaced T1w — question about face-skin label fidelity

---

## Body

Dear Dr Valabregue,

I read your recent preprint *SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training* (arxiv 2605.02737) with great interest. The contrast-agnostic 16-class output and the explicit skin/epidermis label are exactly what I need for a pipeline I am building.

<!-- TODO: personalize — see "Your turn" below. Write 5–10 sentences in your own voice about:
     • who you are and how this work fits your research project
     • what specifically you are doing with faces in MRI (one sentence)
     • why SIAM's whole-head output, not FreeSurfer/FastSurfer, is the right choice for this
-->

I would like to ask three concrete questions:

1. **Synthetic generator code** — is the script that samples synthetic intensities from your six high-resolution label templates released alongside the model weights, or only the inference checkpoint? I would like to retrain on a 7th template (a personal multi-contrast acquisition) and see whether adding one subject closes a known systematic bias.

2. **Face-skin label fidelity** — for our use case the visible face surface from a marching-cubes iso-surface on the T1w is the critical region. Section 3.1 mentions the skin/epidermis label from MIDA but I could not find an evaluation of skin-boundary accuracy. Did you measure it internally, even informally? Any visual examples you could share would help me decide whether to use SIAM's `skin` label directly as a face mask or whether I should fall back to a thresholded intensity contour.

3. **General "segment-everything" paper advice** — I am structuring a short paper that requires segmenting brain *and* face surface from the same non-defaced T1w volume. If you were writing this paper today, would you put the face mask in your pipeline as a SIAM-derived label, a thresholded iso-surface, or both as ablations? I value your perspective more than mine here because you have already navigated the synthetic-vs-real tradeoff.

In return, I would happily share the multi-contrast acquisition I would use as a 7th template, and the brain-age regression error curves I get with SIAM-derived versus FreeSurfer-derived masks. Both could be useful for the next SIAM release.

Best regards,
Ekaterina Kondratyeva

---

## Your turn

Fill the `TODO: personalize` block above. Suggested skeleton (5–10 lines):

> "I am [your role] working on [your one-line project]. Our current manuscript pairs face-age estimation from MRI-derived renders with brain-age estimation from the same T1w volume — the goal is to test whether the face in an MRI scan knows how old the brain is. SIAM's whole-head output is attractive because [your one specific reason — e.g., the explicit skin label avoids manually thresholding a marching-cubes mesh; or the synthetic training means no patient-data overlap with our IXI/SIMON cohorts]."

Tone: respectful, specific, technical. Avoid "I love your paper" — replace with "I read it carefully and used X."

## Reference (for you, not for the email)

- SIAM paper: <https://arxiv.org/abs/2605.02737>
- SIAM repo: <https://github.com/romainVala/SIAM>
- Templates breakdown: 1× MIDA (public) + 3× skull (private) + 2× vasculature (private) — see `papers/related_works/data_methods_map.md` §3.1
