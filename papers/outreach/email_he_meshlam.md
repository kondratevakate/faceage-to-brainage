# Email — Yisheng He (MeshLAM, first author)

**To**: TBD — retrieve from the arXiv author contact panel or the project page before sending
**Subject**: MeshLAM as a single-photo baseline for MRI-derived face mesh comparison

---

## Body

Dear Yisheng,

I read your MeshLAM paper (arxiv 2604.22865) — the one-shot single-photo path to an animatable mesh + texture is exactly what I need as a baseline.

<!-- TODO: personalize — see "Your turn" below. Write 5–10 sentences about:
     • project (paired face-age + brain-age from the same non-defaced T1w MRI)
     • the comparison you want to run: MeshLAM(photo) vs your-own pipeline(MRI-derived render), checked against a personal 5-subject paired photo+MRI cohort
     • single-photo path is critical because most clinical workflows have one frontal photo only
-->

Two specific questions:

1. **Single-photo robustness for unusual targets** — MeshLAM reconstructs an animatable mesh from one photo. If I gave the network a photo of someone whose MRI-derived mesh has *no hair* (T1 iso-surface stops at scalp) and *closed eyes* (supine in scanner), would you expect the reconstructed mesh to be a fair comparison target? Or would the hair/open-eye prior in your training distribution dominate so heavily that the comparison is unfair from the start?

2. **Mesh export format** — for the downstream comparison I need to compare your photo→mesh output against my MRI→mesh output (marching cubes on T1w iso-surface). Is there a documented way to export the MeshLAM mesh in a topology-compatible format (FLAME? a fixed UV layout?) so I can run point-to-point distance?

If MeshLAM works well as a baseline, I will cite it; if it does not, I would still like to understand why so I can write that as a known limitation of the photo→mesh direction for our setting.

Reciprocal: I have 5 paired (frontal photo + non-defaced T1w MRI) subjects that I plan to collect data on this month. That is a niche dataset and I could share anonymized mesh-only versions if you find them useful.

Best regards,
Ekaterina Kondratyeva

---

## Your turn

Fill the `TODO: personalize` block. Suggested skeleton:

> "We are submitting a short paper to [venue] in which we estimate apparent face age from MRI-derived face renders and brain age from the same T1w volume. To validate the face-render branch we want to compare it against a photo-derived mesh — a single-photo method is what we need because the clinical workflow rarely has multi-view photos."

## Reference

- MeshLAM paper: <https://arxiv.org/abs/2604.22865>
- MeshLAM project page: <https://meshlam.github.io/>
- Authors: Yisheng He, Steven Hoi
- Contact still needs manual lookup; do not send until `To` is resolved.
