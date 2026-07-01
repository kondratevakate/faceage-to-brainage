#!/usr/bin/env python3
"""Build a compact evidence ledger for TTA, robustness, and uncertainty work."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_MEASUREMENT_SUMMARY = Path("data/kate_n1_2026/measurement_summary.csv")
DEFAULT_REGISTERED_SUMMARY = Path("data/kate_n1_2026/pseudo_gt_spatial_registered_source_summary.csv")
DEFAULT_BRAINCHOP_SUMMARY = Path("data/kate_n1_2026/brainchop_0.2.5_smoke_results.csv")
DEFAULT_OUTPUT_CSV = Path("data/kate_n1_2026/tta_uncertainty_method_evidence.csv")
DEFAULT_OUTPUT_MD = Path("docs/kate_n1_2026/tta_uncertainty_robust_pipeline_v0.md")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_measurement(rows: list[dict[str, str]], branch: str, metric: str) -> dict[str, str]:
    matches = [row for row in rows if row["branch"] == branch and row["metric"] == metric]
    if len(matches) != 1:
        raise ValueError(f"Expected one measurement for {branch}/{metric}, got {len(matches)}")
    return matches[0]


def add_measurement_row(
    out: list[dict[str, str]],
    row: dict[str, str],
    *,
    evidence_domain: str,
    method_family: str,
    decision: str,
    uncertainty_use: str,
    limitation: str,
) -> None:
    out.append(
        {
            "evidence_domain": evidence_domain,
            "method_family": method_family,
            "branch": row["branch"],
            "metric_family": row["metric_family"],
            "metric": row["metric"],
            "value": row["value"],
            "unit": row["unit"],
            "comparison": row["comparison"],
            "source_file": row["source_file"],
            "decision": decision,
            "uncertainty_use": uncertainty_use,
            "limitation": limitation,
        }
    )


def add_registered_rows(out: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    key_sources = {
        "synthseg_2024_3di",
        "tigerbx_2024_3di",
        "synthseg_2024_t1ffe_ax",
        "synthseg_2024_t1ffe_sag",
        "tigerbx_2024_t1ffe_401",
        "tigerbx_2024_t1ffe_601",
    }
    for row in rows:
        if row["reference_variant"] != "trusted_registered_hard_vote":
            continue
        if row["source_id"] not in key_sources:
            continue
        source = row["source_id"]
        is_3di = "3di" in source
        decision = "exclude_from_visualization" if is_3di else "candidate_or_comparator"
        if source.startswith("synthseg") and not is_3di:
            decision = "primary_2024_candidate"
        out.append(
            {
                "evidence_domain": "registered_spatial_accuracy",
                "method_family": row["method"],
                "branch": source,
                "metric_family": "registered_source_vs_ffe_consensus",
                "metric": "median_dice_p90_hd95_median_volume_error",
                "value": f"{float(row['median_dice']):.3f};{float(row['p90_hd95_mm']):.2f};{float(row['median_abs_volume_error_pct']):.2f}",
                "unit": "dice;mm;percent",
                "comparison": f"{row['scan_id']} vs trusted registered hard-vote pseudo-GT",
                "source_file": "data/kate_n1_2026/pseudo_gt_spatial_registered_source_summary.csv",
                "decision": decision,
                "uncertainty_use": "source_consensus_disagreement",
                "limitation": "Affine registered pseudo-GT from algorithmic sources; not manual ground truth.",
            }
        )


def add_brainchop_rows(out: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row["model"] == "tissue_fast" and row["status"] == "done":
            decision = "quick_tissue_qc_candidate"
            uncertainty_use = "tissue_level_contrast_sensitivity"
        else:
            decision = "not_promoted_runtime_timeout"
            uncertainty_use = "runtime_feasibility_gate"
        out.append(
            {
                "evidence_domain": "brainchop_runtime_and_tissue_qc",
                "method_family": "BrainChop",
                "branch": f"brainchop_0.2.5_{row['model']}_{row['scan_id']}",
                "metric_family": "runtime_and_label_stats",
                "metric": "status_elapsed_labels",
                "value": f"{row['status']};{row['elapsed_sec']};{row.get('labels', '')}",
                "unit": "status;seconds;labels",
                "comparison": row["scan_id"],
                "source_file": "data/kate_n1_2026/brainchop_0.2.5_smoke_results.csv",
                "decision": decision,
                "uncertainty_use": uncertainty_use,
                "limitation": "tissue_fast has only tissue labels and is not an anatomical ASEG/DKT segmentation.",
            }
        )


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    measurements = read_csv(args.measurement_summary)
    out: list[dict[str, str]] = []

    for metric in [
        "median_tta_cv",
        "plus_minus_3deg_floor",
        "interpolation_only_floor",
        "model_instability_component",
    ]:
        add_measurement_row(
            out,
            find_measurement(measurements, "synthseg_rotation_tta", metric),
            evidence_domain="test_time_augmentation",
            method_family="SynthSeg",
            decision="use_as_primary_tta_volume_floor",
            uncertainty_use="per_structure_rotation_cv_and_tta_mean",
            limitation="Same-model TTA captures orientation instability but cannot remove systematic model bias.",
        )

    add_measurement_row(
        out,
        find_measurement(measurements, "synthseg_cross_scanner", "median_cross_scanner_spread"),
        evidence_domain="scanner_protocol_sensitivity",
        method_family="SynthSeg",
        decision="do_not_interpret_cross_protocol_change_without_uncertainty",
        uncertainty_use="scanner_protocol_spread_context",
        limitation="Mixes scanner, time, protocol, and possible biology.",
    )

    for branch, metrics, family in [
        ("fastsurfer_rotation", ["median_floor"], "FastSurfer"),
        ("fastsurfer_long_symmetry", ["cross_median_cv", "long_median_cv"], "FastSurfer Long"),
    ]:
        for metric in metrics:
            matches = [row for row in measurements if row["branch"] == branch and row["metric"] == metric]
            for row in matches:
                add_measurement_row(
                    out,
                    row,
                    evidence_domain="rotation_repeatability",
                    method_family=family,
                    decision="use_as_method_floor_when_input_qc_passes",
                    uncertainty_use="rotation_pair_cv",
                    limitation="2018 synthetic rotation-pair stability; not a direct multi-year accuracy result.",
                )

    add_registered_rows(out, read_csv(args.registered_summary))
    add_brainchop_rows(out, read_csv(args.brainchop_summary))
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evidence_domain",
        "method_family",
        "branch",
        "metric_family",
        "metric",
        "value",
        "unit",
        "comparison",
        "source_file",
        "decision",
        "uncertainty_use",
        "limitation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_decision: dict[str, int] = {}
    for row in rows:
        by_decision[row["decision"]] = by_decision.get(row["decision"], 0) + 1
    decision_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(by_decision.items()))
    text = f"""# TTA and Uncertainty Robust Pipeline Evidence v0

Date: 2026-06-27

This is a compact evidence ledger for the segmentation robustness study. It
combines already computed Kate n=1 TTA, rotation, registered pseudo-GT, and
BrainChop smoke-test evidence. It is not a final benchmark.

Tracked CSV:

- `data/kate_n1_2026/tta_uncertainty_method_evidence.csv`

## Decision Counts

{decision_lines}

## Current Pipeline Position

Primary 2024 anatomical segmentation candidate:

- FFE-derived registered consensus, with SynthSeg FFE sources as the strongest
  current single-method spatial candidates.

Do not promote:

- 2024 3DI anatomical segmentations from SynthSeg, TIGERBx, FastSurfer, or
  BrainChop anatomical models.
- BrainChop `tissue_fast` as anatomical segmentation. It is tissue-level QC
  only.

Use as uncertainty signals:

- SynthSeg 9-angle rotation TTA CV and TTA mean for per-structure volume
  stability.
- Registered source-vs-consensus disagreement for 2024 FFE/3DI spatial
  uncertainty.
- BrainChop `tissue_fast` as a fast tissue-level contrast/QC branch.
- FastSurfer and FastSurfer Long rotation CV as method-floor evidence when
  input QC passes.

## Scientific Boundary

TTA reduces or characterizes orientation sensitivity. It does not prove
anatomical accuracy and does not solve scanner/protocol harmonization. The
current evidence supports a robustness-aware prediction pipeline, not a claim of
manual ground truth.

## Next Experiments

1. Generalize TTA generation so each method records transforms, inverse-resampled
   labels, per-structure CV, voxel vote fraction, and entropy.
2. Apply the same TTA schema to at least SynthSeg and one fast/light comparator
   on a test-retest dataset before claiming a general pipeline.
3. Add nonlinear/unbiased subject-template registration before final spatial
   accuracy claims.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-summary", type=Path, default=DEFAULT_MEASUREMENT_SUMMARY)
    parser.add_argument("--registered-summary", type=Path, default=DEFAULT_REGISTERED_SUMMARY)
    parser.add_argument("--brainchop-summary", type=Path, default=DEFAULT_BRAINCHOP_SUMMARY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_report(args.output_md, rows)
    print(f"Wrote {len(rows)} evidence rows to {args.output_csv}")
    print(f"Wrote report to {args.output_md}")


if __name__ == "__main__":
    main()
