# Email — ITIS Foundation Virtual Population (IXI Heads access)

**To**: virtualpopulation@itis.swiss
**Subject**: IXI Heads — research access and NIfTI-export feasibility

---

## Body

Dear ITIS Virtual Population Team,

I am working on a research project that pairs face-age estimation from MRI-derived face renders with brain-age estimation from the same non-defaced T1-weighted volume. Your **IXI Heads** collection (four detailed 60-tissue segmented head models) is potentially valuable to us as a high-quality reference for the visible face surface and the closed-eye / eye-globe anatomy that ordinary IXI T1w volumes do not resolve cleanly.

<!-- TODO: personalize — see "Your turn" below. Write 5–10 sentences about:
     • who you are and your project, in plain academic language
     • the role IXI Heads would play (n=4 qualitative case study showing the face-surface ground truth and the eyes)
     • that you do NOT need Sim4Life — only the underlying segmentation labels in NIfTI
-->

Three concrete questions:

1. **Research-license access without Sim4Life** — your website states the models are "licensed for use in Sim4Life". For an academic publication that uses the segmentation labels for qualitative comparison only (no electromagnetic simulation), is a license available that grants the labels as NIfTI volumes, or is Sim4Life required regardless?

2. **Subject-ID cross-link to the IXI MRI release** — IXI Heads are presumably derived from four subjects of the open IXI MRI database. Can you share which subject IDs they correspond to? That would let me cross-link against the matching T1w / T2w / PD scans I already have locally.

3. **Pricing / academic fee** — if a license is needed, is there an academic / non-commercial fee tier you could quote? My grant has a small line item for data access but I need to know the order of magnitude before proceeding.

Happy to sign a data-use agreement that restricts the labels to the specific publication and prohibits redistribution.

Best regards,
Ekaterina Kondratyeva

---

## Your turn

Fill the `TODO: personalize` block. Suggested skeleton:

> "I am [role] at [institution]. We are preparing a short paper that pairs face-age and brain-age estimates from the same MRI volume. The four IXI Heads models would let us validate, qualitatively, that the MRI-derived face mesh from our pipeline captures the correct eye anatomy — which ordinary T1w iso-surfaces miss because the eyes are closed."

Tone: formal, single ask per paragraph, no jargon. ITIS is a foundation, not a research lab.

## Reference

- IXI Heads: <https://itis.swiss/virtual-population/regional-human-models/ixi-heads>
- 4 subjects: 2F (29, 66 yr) + 2M (34, 67 yr), 60 tissues, DTI available
