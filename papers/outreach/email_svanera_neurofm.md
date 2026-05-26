# Email — Michele Svanera (NeuroFM, senior author)

**To**: Michele.Svanera@glasgow.ac.uk
**CC**: Austin Dibble (first author)
**Subject**: NeuroFM-L for brain-age regression — cohort exposure and finetuning question

---

## Body

Dear Dr Svanera,

I have been reading the NeuroFM preprint (medRxiv 10.64898/2026.03.27.26349489) and the Hugging Face model card. The 10.8 M-parameter NeuroFM-L looks like a strong candidate as a brain-age backbone for a short paper I am preparing — it is compact, T1w-only matches our data, and the disease-naive synthetic pretraining is a nice property for an unbiased baseline.

<!-- TODO: personalize — see "Your turn" below. Write 5–10 sentences in your own voice about:
     • who you are and what the project does (paired face-age and brain-age from the same T1w)
     • why NeuroFM-L specifically (parameter count fits our compute, T1-only fits our data, no diagnostic labels means we get an unbiased brain-age signal)
     • that you have IXI (n=561) and SIMON (n=99 sessions) locally and want to evaluate fairly
-->

Three specific questions I would like to clarify before using it as a benchmark:

1. **Synthetic generator / LDM100k citation** — the model card says the 100,000 healthy volumes come from LDM100k and that the generator network was trained on UK Biobank. Is there a preferred citation for the exact LDM100k generator version used for NeuroFM pretraining?

2. **Cohort exposure** — the acknowledgements and v2 text mention IXI and SIMON among the real cohorts used in the paper. Were IXI/SIMON used only for evaluation, or were they involved in any model-selection, calibration, or downstream readout training? I want to avoid presenting them as clean external tests if NeuroFM has already seen them in any tuning loop.

3. **Finetuning advice for chronological age** — your "future dementia risk years before diagnosis" result suggests the embedding already encodes a brain-age-gap signal. If you were finetuning NeuroFM-L for chronological-age regression on a 500-subject cohort like IXI, would you (a) freeze the backbone and add a linear head, (b) finetune the last block, or (c) full finetune? Any rough numbers from your internal experiments would be very valuable.

The reciprocal offer: I would be happy to share the brain-age MAE and the test-retest stability I measure on SIMON (single subject, 99 sessions, multiple scanners) — that is a public dataset and the multi-scanner stability is something foundation-model evaluations rarely report.

Best regards,
Ekaterina Kondratyeva

---

## Your turn

Fill the `TODO: personalize` block above. Suggested skeleton (5–10 lines):

> "I am [role] working on [project: paired face-age + brain-age from a single non-defaced T1w]. We aim to test whether the face in an MRI scan knows how old the brain is. For the brain-age branch we currently use SFCN; NeuroFM-L is attractive because [size fits compute / T1w-only matches data / no disease labels means an unbiased brain-age]. Locally we have IXI (561 T1w with resolved age), SIMON (99 sessions one subject across scanners), and a small personal cohort with paired photos. We would like to add NeuroFM-L as a baseline."

Tone: technical, specific, short. Avoid "fascinating paper" — replace with "the disease-naive training is exactly what we need for an unbiased brain-age".

## Reference

- NeuroFM preprint: <https://www.medrxiv.org/content/10.64898/2026.03.27.26349489v2.full-text>
- NeuroFM model card: <https://huggingface.co/NeuroAI-UofG/NeuroFM>
- NeuroFM repo: <https://github.com/rockNroll87q/NeuroFM>
- NeuroFM params: S=484k, M=6.5M, L=10.8M — see `papers/related_works/data_methods_map.md` §3.2
- Svanera lab: Muckli Lab, School of Psychology & Neuroscience, University of Glasgow
