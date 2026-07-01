# Visual QC Overlays for 2024 Segmentation Sources

Date: 2026-06-24

## Scope

This report records the first reproducible visual QC overlay pass for the 2024
candidate scans:

- TIGERBx native `tbetmask`, `aseg`, and `dgm` overlays for 2024 3DI, 2024 T1
  FFE 401, and 2024 T1 FFE 601.
- Registered pseudo-GT overlays for SynthSeg and TIGERBx 2024 sources against
  the trusted registered hard-vote consensus.

Runtime PNGs are outside git:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\qc_overlays\kate_n1_2026
```

The reproducible script and small manifest are tracked:

- `experiments/kate_n1_2026/generate_visual_qc_overlays.py`
- `data/kate_n1_2026/visual_qc_overlay_manifest.csv`

## Method

Native TIGERBx overlays use each scan's TIGERBx brain-extracted image as the
background. The source labels are shown as transparent fill with red/orange
contours.

Registered pseudo-GT overlays use the 2024 T1 FFE 401 image resampled into the
registered label grid. Orange contours are the source segmentation. Cyan
contours are the trusted registered hard-vote consensus. This is the correct
space for comparing SynthSeg/TIGERBx labels after registration; it is not a
native-acquisition display.

## Main Findings

1. TIGERBx `tbetmask` on 2024 3DI is not a gross whole-brain mask failure. The
   high tBET QC score is therefore plausible for masking.
2. 2024 3DI remains unsuitable for promotion. Native ASEG/DGM overlays show
   segmentation following a high-detail/noisy IR-like contrast, and registered
   overlays show visible source-consensus disagreement, especially in cortical
   and white-matter boundaries.
3. The visual disagreement matches the numeric registered pseudo-GT results:
   TIGERBx 2024 3DI has median Dice `0.775` and p90 HD95 `25.08 mm`; SynthSeg
   2024 3DI has median Dice `0.831` and p90 HD95 `22.00 mm`.
4. 2024 FFE 401 and FFE 601 do not show an obvious global label-image mismatch
   in native TIGERBx overlays. After registration, their HD95 tails are much
   lower than 3DI. TIGERBx FFE 401 has p90 HD95 `2.45 mm`; TIGERBx FFE 601 has
   p90 HD95 `2.83 mm`.
5. SynthSeg FFE sources remain the strongest current spatial segmentation
   candidates against the registered pseudo-GT: median Dice about `0.903` and
   p90 HD95 about `2.0-2.24 mm`.

## Promotion Decision

Do not promote any 2024 3DI segmentation to `your-brain-mri-visualization`.

Promote only small derived evidence for visualization:

- registered pseudo-GT summary metrics;
- overlay manifest and selected QC thumbnail paths;
- uncertainty/consensus status if the visualization needs a QC layer.

For anatomical label display, prefer the FFE-derived registered consensus and
SynthSeg FFE sources. TIGERBx FFE outputs are useful as comparator/QC evidence,
not as the primary visualization segmentation at this stage.

## Limitations

This is not manual ground truth. The visual QC confirms or challenges algorithm
outputs, but it does not prove anatomical correctness. The registered stage is
affine-only, and the consensus itself is built from algorithmic sources. The
native TIGERBx overlays and registered pseudo-GT overlays answer different
questions and should not be mixed as if they were the same coordinate space.

## Next Step

Create a small visualization export manifest that pulls these QC-safe summaries
into `your-brain-mri-visualization` without copying raw MRI, full label maps, or
model outputs into git.
