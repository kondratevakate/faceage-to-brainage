# Preregistration strategy

**Decision:** use two separate secondary-data preregistrations first, followed by
a third paired registration only when both measurement branches are frozen.

## Why separate registrations

FaceAge and BrainAge have different inputs, model-selection histories, validity
gates, age ranges and failure modes. Registering them separately prevents a
successful branch from obscuring a failed one and makes the later paired result
conditional on two independently documented measurements.

The sequence is:

```text
development and prior analyses disclosed
  -> freeze FaceAge model and external cohort
  -> register FaceAge external validation
  -> freeze BrainAge model and external cohort
  -> register BrainAge external validation
  -> run and report both branches
  -> verify same-person overlap and freeze paired covariates
  -> register paired FaceAge-BrainAge analysis
  -> open paired outcomes
```

## Registration type and timing

Use the OSF **Secondary Data Preregistration** template. OSF describes a
preregistration as a timestamped, read-only plan created before data collection
or analysis and permits an embargo when necessary for blinded review or data
negotiation ([OSF guidance](https://help.osf.io/article/330-welcome-to-registrations)).

Submit each branch registration only when these fields are locked:

- named dataset/release and access date;
- evidence that target outcomes or model predictions have not been inspected;
- model repository, commit/checkpoint hash and complete preprocessing;
- primary estimand, baseline, exclusions and success rule;
- expected sample size and a precision calculation;
- model-training/dataset overlap assessment;
- ethics/DUA status and contributor list.

Do not submit the current drafts with bracketed blocking fields. A draft can be
shared privately with a data holder, but registration claims begin only at the
OSF timestamp.

## Prior-knowledge disclosure

The project already contains code, literature review and exploratory or
robustness analyses. They must be named in the registration rather than hidden:

- Kate/Case A and SIMON development/QC work;
- any IXI FaceAge or BrainAge outputs already viewed;
- Maclaren and BNU1 BrainAge repeatability work;
- model documentation, benchmark results and published cohort summaries already
  known to the investigators.

These data can support pipeline development, debugging and transparent
rationale. They cannot become a new confirmatory holdout retroactively.

## Paired-registration trigger

The paired registration is deliberately deferred until:

1. access is granted and a participant-by-modality overlap manifest exists;
2. FaceAge and BrainAge branch registrations are submitted and their pipelines
   are frozen or transparently revised under documented amendments;
3. family/participant grouping, visit matching, covariates, negative controls
   and missingness rules are fixed;
4. no paired residual association, within-pair contrast or outcome-stratified
   result has been viewed.

At that point, register the exact paired primary endpoint before opening the
linked holdout. Dataset feasibility counts and unlabeled image-QC checks may be
performed earlier if the registration states exactly what was visible.

## Optional RetinaAge branch

Fundus photography and retinal OCT should not be folded into the FaceAge
registration: they measure retinal vascular/neural ageing rather than external
facial appearance. If this branch proceeds, use a third measurement
preregistration for RetinaAge, independently validate it, and only then register
the paired RetinaAge-BrainAge or three-way FaceAge-RetinaAge-BrainAge analysis.
The EMIF-AD PreclinAD and Monash twin cohorts provide direct feasibility data;
UK Biobank is the large non-twin benchmark.

## Drafts and standards

- [FaceAge external validation](faceage_external_validation_draft.md)
- [BrainAge external validation](brainage_external_validation_draft.md)
- [FaceAge metric contract](../faceage/metrics_protocol.md)
- [BrainAge metric contract](../brainage/metrics_protocol.md)

Reporting should follow
[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378); risk of bias and
applicability should be assessed with
[PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505). These standards do
not replace the dataset-specific preregistration.
