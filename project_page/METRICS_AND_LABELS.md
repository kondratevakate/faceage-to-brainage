# Avatar Evaluation Metrics

This note defines the public metric contract for the avatar workstream. It is
not a claim that the current baseline already meets these targets.

## What the Current Millimeter Distances Mean

The earlier `~2.5 mm` value was a source-to-MRI anterior-cap nearest-neighbor
surface median after a landmark-seeded similarity transform.

It is useful as a pipeline sanity check, but it is not a validated anatomical
accuracy claim.

It means:

- a dense one-photo face surface can be placed near the MRI anterior surface;
- the central face oval is compatible at a coarse surface-distance level;
- nose, chin, cheeks, eyelids, jawline, and soft-tissue ptosis are not yet
  validated.

Main caveats:

- MRI is acquired supine; photos are upright or tilted.
- MRI outer-head segmentation is not the same as a clinical skin surface.
- Current MRI landmarks are automatic proxy landmarks, not manual anatomical
  landmarks.
- Similarity alignment can hide scale and shape errors.

## Sample-Size Notation

The current workspace contains several metric tables created at different
pipeline stages. Their `n` values are not interchangeable:

- public Case A visual evidence: `n=4` repeated photos;
- current landmark-constrained crop batch: `n=9` crop-level meshes
  (`1_1`: 4 crops, `2_1`: 5 crops);
- older selected/free-ICP sanity checks: `n=3` selected meshes, retained only as
  historical/internal diagnostics.

When reporting results, state the unit next to `n`: photos, crop meshes,
subject folders, or selected sanity-check meshes.

## Working Accuracy Tiers

| Tier | Surface/landmark error | Meaning |
|---|---:|---|
| Clinical scanner-level | <1 mm | Good 3D face scanner / anthropometry territory. |
| Strong avatar geometry | 1-2 mm | Good target for controlled scan-to-avatar comparison. |
| Usable one-photo face shape | 2-4 mm | Plausible geometry, not enough for fine soft-tissue claims. |
| Visual avatar only | 4-8 mm | May look plausible, weak anatomical geometry. |
| Not acceptable for geometry | >8 mm | Use only as visual or detector/debug output. |

For tissue ptosis and posture-sensitive tissue, use stricter regional metrics:

- eyelid/periorbital region: target <1-2 mm;
- cheek/midface sag: target <2 mm;
- jawline/chin/submental region: target <2-3 mm;
- full head, hair, and ears from one photo: do not treat as metric geometry
  unless validated against a controlled scan.

## Surface-Distance Contract

Do not report avatar-to-MRI distances as project results until the MRI target
passes a segmentation QC gate. Current automatic MRI face segmentation is not
yet a reliable facial skin surface.

For each reconstruction baseline, report:

1. Alignment policy:
   - rigid or similarity transform;
   - landmark set used;
   - whether scale is fixed or estimated.

2. Masking policy:
   - central face;
   - periorbital region;
   - nose/chin/jaw;
   - posture-sensitive soft tissue excluded or reported separately.

3. Sampling policy:
   - balanced point sampling from both surfaces;
   - consistent density across methods;
   - no method-specific favorable sampling.

4. Metrics:
   - median surface distance;
   - p90 and p95 surface distance;
   - directed Hausdorff in both directions;
   - robust Hausdorff, preferably HD95;
   - ASSD / mean bidirectional surface distance;
   - Chamfer distance.

## Supine vs Upright Face

MRI and CT are usually acquired in a horizontal/supine position. Face photos and
3D photography are usually upright. Gravity changes soft tissue differently
across face regions, so MRI-to-photo comparisons should not assume identical
surface geometry.

Protocol implication:

- compare relatively stable landmarks first: nose bridge/tip, glabella/nasion,
  central chin;
- treat cheeks, eyelids, jawline, and submental tissue as posture-sensitive;
- report posture as a covariate: `mri_supine_vs_photo_upright`.

## Single-Subject Consistency

For a case study, consistency should be computed across repeated photos of the
same subject.

Recommended metrics:

1. Detection success:
   - face found;
   - usable crop;
   - no major occlusion.

2. Shape consistency:
   - align avatar meshes from repeated photos by semantic landmarks;
   - report pairwise median/p90/p95 surface distance;
   - report coefficient of variation for stable facial distances.

3. Landmark consistency:
   - nose-chin, cheek-cheek, brow-chin, nose-cheek distances after a fixed scale
     policy;
   - leave-one-photo-out mean residual.

4. MRI consistency:
   - landmark RMSE to manually annotated MRI landmarks;
   - median/p90/p95 distance on matched facial regions;
   - central face reported separately from posture-sensitive soft tissue.

5. Perceptual consistency:
   - human review or blinded pairwise preference;
   - optional face-embedding similarity only with explicit consent and
     identity-sensitive handling.

## Identity Controls

Identity separation is useful as an internal guardrail, but it should not be the
public visual artifact of the project.

If control subjects are used internally:

- use known consented labels only;
- do not infer identity automatically from appearance;
- report same-subject and different-subject distributions only as aggregate
  numbers;
- do not publish control-subject faces, crops, meshes, or overlays by default.

The public page should stay case-only unless additional subject release is
explicitly curated.

## FaceAge/Twin Literature Context

The FaceAge biomarker story is stored separately in
`TWIN_FACEAGE_LITERATURE_CONTEXT.md`.

Operational rule for this project:

- avatar geometry accuracy, identity consistency, and biological-age validity
  are separate claims;
- twin literature supports the premise that perceived facial age can be
  biologically meaningful;
- current FaceAge/FAHR-Face models are not yet twin-validated;
- the strongest future-work hook is a twin-controlled validation of AI facial
  age against perceived age, lifestyle discordance, methylation age, telomeres,
  and outcomes.
