# AI/ML Experiment Planning: SOTA Practices, Frameworks, and Tools (2023–2026)

*Compiled 2026-04-24 for the faceage-to-brainage project (KK, RK, GB).*
*Scope: scaling from Karpathy-style one-shot ratchet runs to multi-week, pre-registered, reviewer-defensible medical-imaging experiments.*
*Convention: each row distinguishes **[D]** documented standard from **[O]** community opinion / informal wisdom.*

---

## 1. Frameworks for Hypothesis-Driven AI/ML Research

| Framework | What it is | Who uses it | Concrete artifacts | When to use | Tag |
|---|---|---|---|---|---|
| **Karpathy autoresearch** ([repo](https://github.com/karpathy/autoresearch)) | A 630-line Python ratchet: agent edits `train.py`, runs a fixed 5-min training, keeps changes that improve `val_bpb`, reverts otherwise. 12 experiments/h on 1 GPU. Released March 2026. | Solo researchers, Shopify (Lutke +19% val on internal model in 37 runs). | Git history of accepted commits; `analysis.ipynb`; per-run logs. | Local search around a known-good baseline. **Not** for hypothesis-generation or large architectural jumps (greedy local optimum, [Issue #22](https://github.com/karpathy/autoresearch/issues)). | [D] |
| **Sakana AI Scientist v1 / v2** ([v2 paper, 2025](https://arxiv.org/abs/2504.08066), [Sakana](https://sakana.ai/ai-scientist-nature/)) | Agentic tree-search system that formulates hypotheses, codes experiments, writes a full LaTeX manuscript and reviews it via VLM feedback. v2 removes human code templates. | Sakana AI; one v2 paper passed an ICLR 2025 workshop peer review (first fully AI-authored). | Generated `.tex` + `.pdf`, code, figures, AI-reviewer scores. | Exploratory autonomous sweep over a research idea space when budget is large; not for clinical/regulated domains (no provenance guarantees). | [D] |
| **Google AI Co-Scientist** ([arXiv 2502.18864](https://arxiv.org/abs/2502.18864), [research.google](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)) | Multi-agent system on Gemini 2.0 (Generation, Reflection, Ranking, Evolution, Proximity, Meta-review). Self-play scientific debate + tournament ranking. | Stanford (liver fibrosis drug repurposing); Imperial College (AMR — replicated a years-long hypothesis in days). | Ranked, debated hypothesis sets; novelty/feasibility scores; literature-grounded research proposals. | Hypothesis generation phase of a project, before any GPU is touched. | [D] |
| **NeurIPS Paper Checklist 2025** ([guidelines](https://neurips.cc/public/guides/PaperChecklist), [authors' template](https://arxiv.org/html/2505.04037v1)) | Mandatory ~15-item self-assessment on claims, limitations, theory, experiments, reproducibility, assets, broader impact, safeguards. **Papers without it are desk-rejected.** | All NeurIPS / ICML / ICLR submissions. | One-page checklist appended to PDF; mapped to specific paper sections/lines. | Every ML paper, regardless of venue. Treat as a baseline. | [D] |
| **Pre-registration via OSF / AsPredicted for ML** ([OSF Prereg](https://www.cos.io/initiatives/prereg), [arXiv 2311.18807 "Pre-registration for Predictive Modeling"](https://arxiv.org/html/2311.18807)) | OSF templates and the COS preregistration challenge adapted to predictive modelling: declare hypotheses, train/val/test splits, metric, statistical test before touching test set. | Slowly growing in ML4H / NeuroIPS Health workshops; standard in psychology/biomed. | Time-stamped read-only OSF registration; companion `prereg.md` in repo. | Any confirmatory study, especially clinical AI claims. | [D] |
| **Model Cards** (Mitchell et al. 2019, [arXiv 1810.03993](https://arxiv.org/abs/1810.03993)) | Short structured doc shipped with a trained model: intended use, evaluation factors, metrics across subgroups, ethical considerations. | Google, HuggingFace, FDA-tracked submissions. | `MODEL_CARD.md` per checkpoint. | At every model release, even internal. | [D] |
| **Datasheets for Datasets** (Gebru et al. 2018/2021, [arXiv 1803.09010](https://arxiv.org/pdf/1803.09010)) | Structured questionnaire on motivation, composition, collection, preprocessing, uses, distribution, maintenance. | Most major datasets post-2020 (e.g., LAION, BIG-bench). | `DATASHEET.md` with the dataset. | Whenever curating a derivative dataset (e.g., our IXI+SIMON cohort). | [D] |
| **NeurIPS Reproducibility Program / ML Repro Checklist v2.0** (Pineau et al. 2020, [JMLR](https://www.jmlr.org/papers/volume22/20-303/20-303.pdf), [PDF](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)) | Itemised checklist on math description, complexity, data splits, hyperparameters, # runs, env. After introduction, >75% NeurIPS submissions included repro materials. | All major ML conferences. | Filled checklist; Papers-With-Code reproducibility badge. | Standard companion to the NeurIPS Paper Checklist. | [D] |
| **REFORMS** (Kapoor, Narayanan et al., *Sci. Adv.* 2024, [reforms.cs.princeton.edu](https://reforms.cs.princeton.edu/)) | 32-question consensus checklist for ML-based **science** (not just ML for ML). Built specifically to prevent leakage that the authors found in 100s of papers across 17 fields. | Increasingly cited in social/biomedical ML. | Filled REFORMS questionnaire as supplementary. | Whenever ML is used to make a *scientific claim* (e.g. "brain age relates to cognition"). | [D] |
| **Hypothesis-Driven AI manifesto** | No formal Anthropic/Stanford "manifesto" exists with that name (verified April 2026). Closest analogues: Anthropic Transformer Circuits methodology ([transformer-circuits.pub](https://transformer-circuits.pub/)), and the Forde & Paganini "Scientific Method in ML" 2019 ([Semantic Scholar](https://www.semanticscholar.org/paper/Scientific-Method-Forde-Paganini/215d7060ea81a0514779e34ff1db1efbc8ea7dd9)). | Anthropic interpretability team. | Hypothesis → mechanism → falsification cycle docs. | When adapting, do not cite a "manifesto"; cite Forde & Paganini directly. | [O] |
| **Sweep tools, used strategically** ([Hydra+Submitit](https://hydra.cc/docs/plugins/submitit_launcher/), [Optuna](https://optuna.org), [Ray Tune](https://docs.ray.io/en/latest/tune/), W&B Sweeps) | Bayesian / TPE / ASHA / PBT search over config space, optionally launched into SLURM. | Most academic + industry labs. | Pareto plots; best-config artefact; reproducible YAML. | Bottom of the funnel, after a hypothesis is fixed. **Anti-pattern**: running a sweep before stating which metric defines success ([Lipton & Steinhardt 2018](https://arxiv.org/abs/1807.03341), failure mode #2). | [D] tools / [O] discipline |
| **Ablation studies** (Forde & Paganini 2019; Lipton & Steinhardt 2018, [arXiv 1807.03341](https://arxiv.org/abs/1807.03341); Meyes et al. [arXiv 1901.08644](https://arxiv.org/abs/1901.08644)) | Removing components (loss term, augmentation, layer, modality) one-at-a-time to attribute gains. The Lipton/Steinhardt "Troubling Trends" paper makes this **the** signature of credible ML scholarship. | All top ML venues; reviewers ask for it by default. | Ablation table; per-component delta with CIs. | After any positive headline result; **never** ship a paper without it. | [D] |

---

## 2. Reporting Standards Specific to Medical Imaging / Clinical AI

| Standard | 1-line description | Current ver. / year | Canonical citation | Use for OUR project? |
|---|---|---|---|---|
| **TRIPOD+AI** | 27-item update of TRIPOD for clinical prediction models using regression *or* ML. Supersedes TRIPOD 2015. | 2024 (BMJ, 16 Apr) | Collins et al. *BMJ* 2024 — [pmc/PMC11019967](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/), [tripod-statement.org](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf) | **Yes — primary.** Brain age IS a prediction model. |
| **CLAIM** | Best-practice checklist for AI in medical imaging. Each item now Yes/No/NA with manuscript page+line refs. | 2024 Update (RSNA) | Tejani et al. *Radiology: AI* 2024 — [pubs.rsna.org/doi/10.1148/ryai.240300](https://pubs.rsna.org/doi/full/10.1148/ryai.240300), [pmc/PMC11304031](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/) | **Yes — secondary, complements TRIPOD+AI.** Imaging-specific. |
| **MI-CLAIM** | Minimum information for clinical AI modelling: 6-step pipeline (study, data, dev, validation, results, code). | 2020; MI-CLAIM-GEN 2024 ([arXiv 2403.02558](https://arxiv.org/abs/2403.02558)) | Norgeot et al. *Nat. Med.* 2020 — [nature.com/articles/s41591-020-1041-y](https://www.nature.com/articles/s41591-020-1041-y), [github.com/beaunorgeot/MI-CLAIM](https://github.com/beaunorgeot/MI-CLAIM) | Optional; overlaps TRIPOD+AI. Skip unless journal asks. |
| **DECIDE-AI** | Reporting for **early-stage live clinical evaluation** of AI decision-support. | 2022 (*Nat. Med.*) | Vasey et al. — [nature.com/articles/s41591-022-01772-9](https://www.nature.com/articles/s41591-022-01772-9) | **No** — we are pre-clinical. |
| **CONSORT-AI** | RCT report extension: 14 new items on AI intervention specification, error analysis. | 2020 | Liu et al. *Nat. Med.* — [nature.com/articles/s41591-020-1034-x](https://www.nature.com/articles/s41591-020-1034-x) | No — no RCT. |
| **SPIRIT-AI** | Trial **protocol** extension (companion of CONSORT-AI). | 2020; *Lancet* 2025 noted CONSORT/SPIRIT 2025 still lack AI items ([thelancet.com](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(25)01268-1/fulltext)) | Cruz Rivera et al. *Nat. Med.* — [nature.com/articles/s41591-020-1037-7](https://www.nature.com/articles/s41591-020-1037-7) | No. |
| **STARD-AI** | Diagnostic accuracy reporting; 18 new/modified items over STARD 2015. | 2025 (*Nat. Med.*) | Sounderajah et al. — [nature.com/articles/s41591-025-03953-8](https://www.nature.com/articles/s41591-025-03953-8) | Partial — only if we frame as a diagnostic test (e.g. brain-age-gap → MCI). |
| **PROBAST+AI** | Risk-of-bias **assessment** tool for AI prediction models; 16 dev + 18 eval signalling questions, 4 domains. | 2025 (*BMJ*) | Collins, Dhiman, Wolff et al. — [pubmed/40127903](https://pubmed.ncbi.nlm.nih.gov/40127903/) | **Yes** as a self-audit before submission; reviewers will use it. |

Crosswalk for our paper: write to **TRIPOD+AI + CLAIM 2024**, self-score with **PROBAST+AI**, append **NeurIPS Paper Checklist** + **REFORMS** for the methods venue.

---

## 3. Statistical and Design Rigor for ML Experiments

| Topic | Recommendation | Key reference | Tag |
|---|---|---|---|
| **Pre-registration of ML** | Two-stage: (a) hypothesis + analysis plan pre-data-look; (b) confirmatory analysis on locked test set after the exploratory phase. Use OSF templates; keep an in-repo `prereg.md` mirror. | [Forde "Scientific Method in ML" 2019](https://www.semanticscholar.org/paper/Scientific-Method-Forde-Paganini/215d7060ea81a0514779e34ff1db1efbc8ea7dd9); [Pre-registration for Predictive Modeling, arXiv 2023/2024](https://arxiv.org/html/2311.18807); [Munafò *Nat. Hum. Behav.* 2017](https://www.nature.com/articles/s41562-016-0021) | [D] |
| **Power analysis** | For brain-age regression: target Pearson r CI half-width <0.05 → ≈n>1500 healthy controls (Cohen f² with small-effect r=.10 → n>780 for α=.05, β=.20). For test-retest ICC: n≥30 subjects ×2 scans for CI ±0.10 ([Koo & Li 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/)). fMRI/neuroimaging power literature warns common n=20-30 is severely underpowered ([Geuter et al. *PLOS ONE* 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0184923)). | Cohen *Statistical Power Analysis* 1988; PASS / G*Power. | [D] |
| **Multiple comparisons in ML** | Use **Benjamini-Hochberg FDR** when reporting per-region or per-cohort stats; never report >5 hypothesis tests without correction. Bonferroni only for ≤5 pre-specified primaries. | [Benjamini & Hochberg 1995]; in brain-age DL: pace-of-aging × ADAS-13 used BH ([NSF par #10626299](https://par.nsf.gov/servlets/purl/10626299)) | [D] |
| **Bias correction (brain-age specific)** | Apply the **Beheshti** (chronological-age covariate) or **Cole** linear correction *fit on train fold* and applied to test fold to remove regression-to-the-mean bias. **Never** fit correction on test data — that inflates downstream associations. | [Smith et al. 2019, *NeuroImage*](https://www.sciencedirect.com/science/article/pii/S2213158219304103); [Beheshti et al. 2019](https://github.com/Beheshtiiman2/Bias-Correction-in-Brain-Age-Estimation-Frameworks); [de Lange & Cole 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7049655/) | [D] |
| **Permutation tests** | Use for any group-level inference where the test statistic distribution is unknown / non-Gaussian (e.g. comparing brain-age-gap distributions across MRI sites). Default 10,000 permutations. | [Nichols & Holmes 2002, *Hum. Brain Mapp.*](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/NicholsHolmes.pdf); FSL's `randomise`; `mlxtend.evaluate.permutation_test`. | [D] |
| **Bootstrap CIs on metrics** | Resample test set with replacement n≥2000 times; report 2.5/97.5 percentile CIs on MAE / R / Pearson. Keep model fixed (no retrain). | [Raschka 2022 blog](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html); [Ferrer ConfidenceIntervals lib](https://github.com/luferrer/ConfidenceIntervals). | [D] |
| **Subject-disjoint k-fold CV** | **Every subject in exactly one fold.** Random splits at scan level inflate face-age MAE by ~30% via subject leakage. | [Paplhám et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Paplham_A_Call_to_Reflect_on_Evaluation_Practices_for_Age_Estimation_CVPR_2024_paper.pdf), [code](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark) | [D] |
| **Two-stage analysis** | (i) lock train/val + analysis code + statistical plan; (ii) freeze test set, run *one* confirmatory pass; (iii) any further test-set analysis is exploratory and labelled as such in paper. | Munafò 2017; REFORMS Q.20 | [D] |
| **Common ML pitfalls** | Data leakage (preprocessing fitted on whole dataset), test-set re-use, HARKing, ablations cherry-picked. | [Kapoor & Narayanan REFORMS, *Sci. Adv.* 2024](https://www.science.org/doi/10.1126/sciadv.adk3452) | [D] |

---

## 4. Tools and Infrastructure

| Need | Tools | Pick & rationale |
|---|---|---|
| **Experiment tracking** | MLflow (open, OSS); W&B (hosted, polished); Aim (open, fast); Comet; Neptune. | **W&B for MIDL revision.** Best UX, free academic tier, sweep + report features. MLflow if institutional policy forbids cloud — slower UI, but full local control ([ZenML comparison](https://www.zenml.io/blog/mlflow-vs-weights-and-biases), [MLtraq benchmarks](https://mltraq.com/benchmarks/speed/)). |
| **HPO** | Optuna (TPE, lightweight, framework-agnostic), Ray Tune (distributed, ASHA/PBT), W&B Sweeps (UI-coupled), Hyperopt **deprecated** (no longer maintained — Databricks dropped it after 16.4 LTS). | Optuna locally + Ray Tune when SLURM available. Skip Hyperopt. |
| **Config & sweep launcher** | Hydra + Submitit ([Hydra docs](https://hydra.cc/docs/plugins/submitit_launcher/), [Meta blog](https://ai.meta.com/blog/open-sourcing-submitit-a-lightweight-tool-for-slurm-cluster-computation/)) | Hydra YAML + `submitit_slurm` launcher: zero-sbatch sweeps, Optuna sweeper plugin available. |
| **Containers** | Docker (dev/CI); Apptainer/Singularity for HPC (rootless, no daemon, [apptainer.org](https://apptainer.org/)) | Docker → push to registry → `apptainer build .sif docker://` for cluster. |
| **Reproducible Python** | `uv` (fast, lockfile), `conda-lock`, `pip-tools`. | `uv` + `pyproject.toml` + checked-in `uv.lock`. Conda only when binary deps (FreeSurfer, FSL) require it. |
| **Code+data versioning** | DVC (Git-integrated, S3 backend), Pachyderm (k8s-native, lineage), lakeFS (S3 branching), Git-LFS, Dolt. | **DVC** for our scale (sub-TB). lakeFS later if dataset grows >TB. |
| **Notebook discipline** | Karpathy: "no Jupyter for production." Khan: notebooks for EDA only, scripts for training. | Notebooks for figures + EDA only; all training in `src/` scripts called from CLI. nbstripout pre-commit hook (already in our repo). |
| **Folder convention** | OpenAI Baselines / Detectron2 / MMDet: `configs/`, `tools/train.py`, `experiments/{exp_id}/{ckpts,logs,figs}/`, run name = `{date}_{model}_{seed}` | Adopt this verbatim for the IXI×SIMON pipeline. |

---

## 5. Books / Talks / Blog Posts to Read (≤12)

1. **Andrew Ng — *Machine Learning Yearning*** (free PDF) — practical ML project management; error analysis as a discipline.
2. **Lipton & Steinhardt 2018, "Troubling Trends in ML Scholarship"** ([arXiv 1807.03341](https://arxiv.org/abs/1807.03341)) — the four sins: explanation/speculation conflation, unattributed gains, mathiness, language abuse.
3. **Pineau et al. 2021, "Improving Reproducibility in ML"** ([JMLR](https://www.jmlr.org/papers/volume22/20-303/20-303.pdf)) — what the NeurIPS reproducibility programme actually changed.
4. **Munafò et al. 2017, "Manifesto for reproducible science"** ([*Nat. Hum. Behav.*](https://www.nature.com/articles/s41562-016-0021)) — the canonical pre-registration / methods / dissemination reform argument.
5. **Ioannidis 2005, "Why Most Published Research Findings Are False"** ([PLOS Med.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124)) — base-rate, power, and bias arithmetic.
6. **Forde & Paganini 2019, "The Scientific Method in the Science of ML"** — closest thing to the "Hypothesis-Driven AI" doc you imagined.
7. **Carlini, "On Evaluating Adversarial Robustness"** ([nicholas.carlini.com](https://nicholas.carlini.com/papers/2019_howtoeval.pdf)) — gold standard for adversarial-claim sanity checks; principles transfer to any "we improved X" claim.
8. **Karpathy talks** — "A Recipe for Training Neural Networks" (2019); Stanford CS231n lectures; the autoresearch announcement thread.
9. **Anthropic Transformer Circuits** ([transformer-circuits.pub](https://transformer-circuits.pub/)) — exemplar of mechanism-first, hypothesis-tested ML research methodology.
10. **Kapoor & Narayanan, REFORMS 2024** ([Sci. Adv.](https://www.science.org/doi/10.1126/sciadv.adk3452)) — the leakage taxonomy + checklist that will save us from a desk reject.
11. **Eisner, "How to do research at the MIT AI Lab" + "Doing research in NLP"** — durable mentorship classics on framing problems and managing time horizons.
12. **Christopher Ré — Stanford MLSys seminar series** ([mlsys.stanford.edu](https://mlsys.stanford.edu/)) — empirical-ML methodology with systems rigor.

---

## 6. Concrete Recommendations for THIS Project

Context: 3-person team (KK, RK, GB) + Claude as autoresearch ratchet; brain-age × face-age from a single MRI; IXI + SIMON, planning OOD test-retest cohorts; **MIDL 2026 short-paper revision**.

### Top-5 ranked actions

1. **Adopt TRIPOD+AI 2024 as primary reporting standard, CLAIM 2024 as imaging companion, PROBAST+AI as pre-submission self-audit.**
   - Action: copy the [TRIPOD+AI 27-item checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf) and [CLAIM 2024 Word doc](https://pubs.rsna.org/doi/full/10.1148/ryai.240300) into `papers/midl2026/checklists/`.
   - Map every item to a `(section, line)` in the manuscript before resubmission.
   - File path: `c:/Projects/02_academia/faceage-to-brainage/papers/midl2026/checklists/TRIPOD+AI.md` and `.../CLAIM2024.md`.

2. **Use Weights & Biases as the sole experiment tracker for the MIDL revision sprint.**
   - Reason: best-in-class UI, free academic tier, native sweep + report sharing, low setup overhead for a 3-person team. Aim is faster but less polished; MLflow needs a server we don't want to admin. Migrate to MLflow only if a future hospital partner forbids cloud upload of metrics (no PHI is ever logged either way — only IDs and metric values, per global rules).
   - Action: `wandb login`, set `WANDB_PROJECT=faceage-to-brainage`, name runs `{YYYYMMDD}_{model}_{seed}_{cohort}`.
   - File path: integrate in `src/training/train.py`; never log raw scan paths or subject demographics — only subject IDs and aggregate metrics.

3. **Pre-register on OSF with a local mirror in the repo (do both).**
   - Why both: OSF gives a tamper-proof timestamp reviewers trust; local markdown gives an editable diff trail for the team.
   - Structure: one OSF registration per *confirmatory* hypothesis (currently 2: "face-age and brain-age share variance independent of chronological age"; "OOD test-retest ICC of brain-age-gap > 0.7"). Use the OSF "Preregistration Template (with images)" form.
   - File paths:
     - `c:/Projects/02_academia/faceage-to-brainage/prereg/H1_shared_variance.md`
     - `c:/Projects/02_academia/faceage-to-brainage/prereg/H2_test_retest_icc.md`
   - Lock test split BEFORE filling the OSF form. Each `.md` mirrors the OSF fields verbatim and links the OSF DOI once registered.

4. **Ablations expected by MIDL/MICCAI reviewers (ranked by reviewer push-back probability):**
   - (a) **Modality / input channel** — face-only, brain-only, joint. Prove the joint signal exceeds each marginal.
   - (b) **Bias correction** — un-corrected vs Beheshti vs Cole. Reviewers will ask which curve is reported.
   - (c) **Subject-disjoint vs random split** — show the gap (Paplhám et al. CVPR 2024 will be cited *at* you if you don't pre-empt this).
   - (d) **Backbone** — at least one alternative (e.g. SFCN vs ResNet-18 3D vs ViT-S) at fixed compute.
   - (e) **Bootstrap CI** on every reported MAE / r — non-overlapping CIs are the difference between "trend" and "result".
   - (f) **OOD generalisation** — per-site MAE on SIMON travelling-subject scans; ICC of brain-age-gap across repeats. This is the headline novelty — guard it with the most ablations.

5. **First read this week (single item):** **Lipton & Steinhardt, "Troubling Trends in ML Scholarship"** ([arXiv 1807.03341](https://arxiv.org/abs/1807.03341)).
   - Reason: 8 pages, directly maps to the writing failures most likely in our MIDL revision (mixing speculation with explanation when discussing why face-age helps brain-age; mathiness in the loss-function description; HARKed ablations). Read it before drafting the revision response.
   - Companion (≤30 min): the **NeurIPS 2025 Paper Checklist** ([guidelines](https://neurips.cc/public/guides/PaperChecklist)) — fill it for the MIDL paper even though MIDL doesn't require it; reviewers cross-pollinate.

### Process additions (low-cost, high-leverage)

- **Smoke test before every overnight loop** (already in user CLAUDE.md): 1-scan dry run + ETA print + GPU check; abort if smoke fails.
- **Per-experiment `manifest.json`** committed in `experiments/{run_id}/`: git SHA, dataset hash (DVC), seed, hardware, wall-time. Required for any figure that goes into a paper.
- **Weekly 30-min "ratchet review"** (KK, RK, GB): triage W&B dashboard, decide which branches die, which become confirmatory.
- **Audit log** (HIPAA/GDPR/ISO 27001 compliance per global rules): `logs/audit/{YYYY-MM-DD}.jsonl` with `(actor_id, action, entity_id, ts)` only — never values, never names.
- **Container pinning**: build one Apptainer `.sif` per submission: `apptainer build faceage-midl2026.sif Apptainer.def`. Push the recipe to repo; the `.sif` to Zenodo on acceptance for reproducibility-checklist credit.

### Decisions explicitly *not* made

- No Pachyderm / lakeFS yet — overkill at <500 GB.
- No CONSORT-AI / SPIRIT-AI / DECIDE-AI — we are pre-clinical.
- No Sakana AI Scientist runs against this dataset — provenance not auditable enough for a clinical-imaging paper. Karpathy ratchet is the right scale.

---

## Bibliography (compact)

- Beheshti et al. 2019, *NeuroImage* — bias-adjustment in brain age. [doi](https://doi.org/10.1016/j.nicl.2019.102063)
- Carlini 2019 — "On Evaluating Adversarial Robustness". [PDF](https://nicholas.carlini.com/papers/2019_howtoeval.pdf)
- Cohen 1988 — *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. [PDF](https://utstat.toronto.edu/~brunner/oldclass/378f16/readings/CohenPower.pdf)
- Collins, Moons, Dhiman et al. 2024 — TRIPOD+AI. *BMJ*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/)
- Collins, Dhiman, Wolff et al. 2025 — PROBAST+AI. *BMJ*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40127903/)
- Cruz Rivera et al. 2020 — SPIRIT-AI. *Nat. Med.* [link](https://www.nature.com/articles/s41591-020-1037-7)
- de Lange & Cole 2020 — brain-age correction commentary. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7049655/)
- Forde & Paganini 2019 — "The Scientific Method in the Science of ML". [Semantic Scholar](https://www.semanticscholar.org/paper/Scientific-Method-Forde-Paganini/215d7060ea81a0514779e34ff1db1efbc8ea7dd9)
- Gebru et al. 2018/2021 — Datasheets for Datasets. [arXiv](https://arxiv.org/pdf/1803.09010)
- Google Research 2025 — AI Co-Scientist. [arXiv 2502.18864](https://arxiv.org/abs/2502.18864)
- Ioannidis 2005 — Why Most Published Research Findings Are False. [PLOS Med](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124)
- Kapoor & Narayanan 2024 — REFORMS. *Sci. Adv.* [link](https://www.science.org/doi/10.1126/sciadv.adk3452)
- Karpathy 2026 — autoresearch. [GitHub](https://github.com/karpathy/autoresearch)
- Lipton & Steinhardt 2018 — Troubling Trends. [arXiv 1807.03341](https://arxiv.org/abs/1807.03341)
- Liu et al. 2020 — CONSORT-AI. *Nat. Med.* [link](https://www.nature.com/articles/s41591-020-1034-x)
- Mitchell et al. 2019 — Model Cards. [arXiv 1810.03993](https://arxiv.org/abs/1810.03993)
- Munafò et al. 2017 — Manifesto for reproducible science. *Nat. Hum. Behav.* [link](https://www.nature.com/articles/s41562-016-0021)
- Nichols & Holmes 2002 — Permutation tests for fMRI. [PDF](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/NicholsHolmes.pdf)
- Norgeot et al. 2020 — MI-CLAIM. *Nat. Med.* [link](https://www.nature.com/articles/s41591-020-1041-y)
- Paplhám et al. 2024 — A Call to Reflect on Evaluation Practices for Age Estimation. *CVPR*. [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Paplham_A_Call_to_Reflect_on_Evaluation_Practices_for_Age_Estimation_CVPR_2024_paper.pdf)
- Pineau et al. 2021 — Improving Reproducibility in ML. *JMLR* 22. [PDF](https://www.jmlr.org/papers/volume22/20-303/20-303.pdf)
- Sakana AI 2024/2025 — AI Scientist v1/v2. [v2 arXiv](https://arxiv.org/abs/2504.08066)
- Smith et al. 2019 — *eLife* brain aging modes. [link](https://elifesciences.org/articles/52677)
- Sounderajah et al. 2025 — STARD-AI. *Nat. Med.* [link](https://www.nature.com/articles/s41591-025-03953-8)
- Tejani et al. 2024 — CLAIM 2024 Update. *Radiology: AI*. [link](https://pubs.rsna.org/doi/full/10.1148/ryai.240300)
- Vasey et al. 2022 — DECIDE-AI. *Nat. Med.* [link](https://www.nature.com/articles/s41591-022-01772-9)

---
*Last updated: 2026-04-24. Review cycle: every MIDL/MICCAI submission.*
