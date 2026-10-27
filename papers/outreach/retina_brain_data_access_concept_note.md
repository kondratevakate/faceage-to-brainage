# Feasibility concept: RetinaAge and BrainAge in twin cohorts

**Investigator:** Ekaterina Kondratyeva, `[institution/department]`

**Status:** feasibility concept only. Institutional sponsorship, ethics
approval/determination, funding and data access are not yet implied.

## Objective

The project will independently evaluate whether frozen retinal-image and brain
MRI pipelines produce accurate, calibrated and technically reliable age-related
measurements. Only after those validations will a separate registration test
eye-brain associations and within-twin-pair contrasts.

RetinaAge is distinct from FaceAge: retinal fundus photographs and OCT represent
neural and microvascular tissue, not external facial appearance. Neither
RetinaAge nor BrainAge will be interpreted as a causal biological-age rate or a
clinical diagnosis.

## Feasibility information requested

At first contact we request only aggregate counts, an image inventory, data
dictionary and governance constraints. No participant-level files are needed.

| Domain | Minimum approved data for later analysis |
|---|---|
| Linkage | Random family, participant, eye and visit IDs; zygosity |
| Time | Age and retinal/MRI acquisition dates or permitted precision |
| Retina | Source fundus photographs; native OCT volumes when available; device, eye, field and QC metadata |
| Brain | Source T1 MRI; FLAIR/DWI/ASL/SWI when available; scanner/protocol and QC metadata |
| Adjustment | Sex, site, vascular risk, diabetes, smoking, blood pressure, BMI, cognition and recruitment stratum as available |

## Analysis sequence

1. Freeze retinal and MRI model versions, preprocessing, QC and exclusions.
2. Register RetinaAge and BrainAge validations separately before target
   predictions or accuracy outcomes are inspected.
3. Report all attempts, failures, accuracy, calibration and technical
   repeatability with participant/family-clustered uncertainty.
4. Register the paired eye-brain endpoint before linked residuals, retinal
   features or within-pair associations are examined.

All eyes, images and visits from one participant and both members of a family
remain grouped. Target labels are not used for model, crop, layer, preprocessing
or bias-correction selection.

## Governance and outputs

Direct identifiers and linkage keys are not requested. Processing will occur in
the holder-approved environment. Source retinal images and MRI will not be
redistributed; re-identification is prohibited. Publications will contain
aggregate results and disclosure-controlled illustrations only with explicit
permission.

The immediate decision requested is whether the source retinal and MRI files
overlap in the same people and visits, whether the proposed processing is within
consent, and which formal access route should be used.
