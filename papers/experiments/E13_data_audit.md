# E13 — IXI / SIMON data audit (the half-missing-patients problem)

- **Author**: CL
- **Date**: 2026-04-25
- **Status**: findings only (no new data produced)
- **Reference**: [`papers/related_works/data_audit.md`](../related_works/data_audit.md)

## Trigger
KK flagged that the IXI numbers feel underpowered and that
preprocessing seems "weird". This audit confirms that and quantifies it.

## Findings
| Stage | N subjects | % of IXI-581 |
|---|---|---|
| IXI full cohort | 581 | 100 % |
| Face raw multi-view (E01) | 406 | 70 % |
| Face splits (E02) | 476 | 82 % |
| **Brain-age SynthBA (E05)** | **105** | **18 %** ← bottleneck |
| Paired (E07) | 93 | 16 % |

### Site bias

| Site | in face raw mv | IXI published | Coverage |
|---|---|---|---|
| Guys | 275 | ~320 | 86 % |
| HH | 129 | ~185 | 70 % |
| **IOP** | **2** | **~70** | **3 %** ← gone |

The "3 London sites" framing of the paper is not honest. We have Guys
and HH only. IOP intensity distribution likely differs enough that the
fixed marching-cubes threshold `t = 30` cuts below the skin.

### 80 ghost subjects
80 IXI subjects are listed in `ixi_faceage_splits.csv` with train/val/
test assignments but never appear in `ixi_faceage_multiview_raw.csv` —
the render pipeline stopped before reaching them.

## Consequences
1. Anything reported on N = 93 (the gap correlation, etc.) is
   under-powered: detectable correlation floor is ≈ r = 0.28. We
   cannot claim small-effect-size correlations on this cohort.
2. Site-stratified analyses are not possible.
3. Per-subject failure logging is missing throughout.

## Action items (ranked)
**A1.** Re-run SynthBA on full IXI with per-stage failure logging.
Expected post-fix N: ~550.

**A2.** Diagnose IOP: inspect intensity histogram of one IOP scan,
switch to adaptive (Otsu / N-percentile) threshold for marching cubes.

**A3.** Re-render the 80 planned-but-unrendered face subjects.

**A4.** Add per-subject log line to every batch script:
`subject_id, stage, status, wall_time, error_message`.

## What's left undone (everything above)
None of A1–A4 is done yet. **No new statistics on IXI should be run
until the cohort is restored.** The autoresearch ratchet
([`papers/midl2026/autoresearch_ratchet.py`](../midl2026/autoresearch_ratchet.py))
was prepared but not executed for this reason.

## Pointers
- Full audit doc: [`papers/related_works/data_audit.md`](../related_works/data_audit.md)
- SOTA design context: [`papers/related_works/sota_design.md`](../related_works/sota_design.md)
