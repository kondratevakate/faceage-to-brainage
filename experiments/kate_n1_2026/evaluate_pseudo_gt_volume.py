#!/usr/bin/env python3
"""Build volume-level pseudo-GT references and score segmentation sources.

This script intentionally works on derived volume tables, not raw MRI or label
maps. It is the first, registration-free evaluation stage. Spatial Dice and
surface metrics require label maps in a common subject/template space and are
handled by a later pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ASEG_LABELS = {
    2: "left_cerebral_white_matter",
    3: "left_cerebral_cortex",
    4: "left_lateral_ventricle",
    5: "left_inferior_lateral_ventricle",
    7: "left_cerebellum_white_matter",
    8: "left_cerebellum_cortex",
    10: "left_thalamus",
    11: "left_caudate",
    12: "left_putamen",
    13: "left_pallidum",
    14: "third_ventricle",
    15: "fourth_ventricle",
    16: "brain_stem",
    17: "left_hippocampus",
    18: "left_amygdala",
    24: "csf",
    26: "left_accumbens_area",
    28: "left_ventral_dc",
    41: "right_cerebral_white_matter",
    42: "right_cerebral_cortex",
    43: "right_lateral_ventricle",
    44: "right_inferior_lateral_ventricle",
    46: "right_cerebellum_white_matter",
    47: "right_cerebellum_cortex",
    49: "right_thalamus",
    50: "right_caudate",
    51: "right_putamen",
    52: "right_pallidum",
    53: "right_hippocampus",
    54: "right_amygdala",
    58: "right_accumbens_area",
    60: "right_ventral_dc",
}

SYNTHSEG_COLUMNS = {
    "total intracranial": "total_intracranial",
    "left cerebral white matter": "left_cerebral_white_matter",
    "left cerebral cortex": "left_cerebral_cortex",
    "left lateral ventricle": "left_lateral_ventricle",
    "left inferior lateral ventricle": "left_inferior_lateral_ventricle",
    "left cerebellum white matter": "left_cerebellum_white_matter",
    "left cerebellum cortex": "left_cerebellum_cortex",
    "left thalamus": "left_thalamus",
    "left caudate": "left_caudate",
    "left putamen": "left_putamen",
    "left pallidum": "left_pallidum",
    "3rd ventricle": "third_ventricle",
    "4th ventricle": "fourth_ventricle",
    "brain-stem": "brain_stem",
    "left hippocampus": "left_hippocampus",
    "left amygdala": "left_amygdala",
    "csf": "csf",
    "left accumbens area": "left_accumbens_area",
    "left ventral dc": "left_ventral_dc",
    "right cerebral white matter": "right_cerebral_white_matter",
    "right cerebral cortex": "right_cerebral_cortex",
    "right lateral ventricle": "right_lateral_ventricle",
    "right inferior lateral ventricle": "right_inferior_lateral_ventricle",
    "right cerebellum white matter": "right_cerebellum_white_matter",
    "right cerebellum cortex": "right_cerebellum_cortex",
    "right thalamus": "right_thalamus",
    "right caudate": "right_caudate",
    "right putamen": "right_putamen",
    "right pallidum": "right_pallidum",
    "right hippocampus": "right_hippocampus",
    "right amygdala": "right_amygdala",
    "right accumbens area": "right_accumbens_area",
    "right ventral dc": "right_ventral_dc",
}


@dataclass(frozen=True)
class Source:
    source_id: str
    method: str
    session_group: str
    scan_id: str
    format: str
    path: Path
    include_in_reference: bool
    notes: str


def read_manifest(path: Path, data_root: Path) -> list[Source]:
    sources: list[Source] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sources.append(
                Source(
                    source_id=row["source_id"],
                    method=row["method"],
                    session_group=row["session_group"],
                    scan_id=row["scan_id"],
                    format=row["format"],
                    path=data_root / row["relative_path"],
                    include_in_reference=row.get("include_in_reference", "0") == "1",
                    notes=row.get("notes", ""),
                )
            )
    return sources


def read_synthseg_wide(source: Source) -> list[dict[str, str | float | bool]]:
    rows: list[dict[str, str | float | bool]] = []
    with source.path.open("r", newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 1:
        raise ValueError(f"Expected one row in {source.path}, got {len(csv_rows)}")
    row = csv_rows[0]
    for raw_name, canonical in SYNTHSEG_COLUMNS.items():
        value = row.get(raw_name)
        if value in (None, ""):
            continue
        rows.append(volume_row(source, canonical, float(value) / 1000.0, "aseg_volume"))
    return rows


def read_tigerbx_long(source: Source) -> list[dict[str, str | float | bool]]:
    rows: list[dict[str, str | float | bool]] = []
    with source.path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scan_id"] != source.scan_id or row["output_type"] != "aseg":
                continue
            label = int(row["label"])
            canonical = ASEG_LABELS.get(label)
            if canonical is None:
                continue
            rows.append(volume_row(source, canonical, float(row["volume_ml"]), "aseg_volume"))
    if not rows:
        raise ValueError(f"No TIGERBx rows matched {source.scan_id} in {source.path}")
    return rows


def volume_row(
    source: Source,
    structure: str,
    volume_ml: float,
    ontology: str,
) -> dict[str, str | float | bool]:
    return {
        "source_id": source.source_id,
        "method": source.method,
        "session_group": source.session_group,
        "scan_id": source.scan_id,
        "structure": structure,
        "ontology": ontology,
        "volume_ml": volume_ml,
        "include_in_reference": source.include_in_reference,
        "notes": source.notes,
    }


def load_volume_rows(sources: list[Source]) -> list[dict[str, str | float | bool]]:
    rows: list[dict[str, str | float | bool]] = []
    for source in sources:
        if not source.path.exists():
            raise FileNotFoundError(f"Missing source table for {source.source_id}: {source.path}")
        if source.format == "synthseg_wide":
            rows.extend(read_synthseg_wide(source))
        elif source.format == "tigerbx_long":
            rows.extend(read_tigerbx_long(source))
        else:
            raise ValueError(f"Unsupported format: {source.format}")
    return rows


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def mad(values: list[float]) -> float:
    center = median(values)
    return median([abs(v - center) for v in values])


def pct_error(value: float, reference: float) -> float:
    if reference == 0:
        return math.nan
    return (value - reference) / reference * 100.0


def abs_rel_error(value: float, reference: float) -> float:
    if reference == 0:
        return math.nan
    return abs(value - reference) / reference * 100.0


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    index = round((len(clean) - 1) * pct)
    return clean[index]


def build_reference_rows(
    rows: list[dict[str, str | float | bool]],
    min_sources: int,
) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple[str, str], list[dict[str, str | float | bool]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["session_group"]), str(row["structure"]))].append(row)

    refs: list[dict[str, str | float | int]] = []
    for (session_group, structure), group in sorted(grouped.items()):
        refs.extend(reference_for_group(session_group, structure, group, "all_source_median", group, min_sources))
        trusted = [row for row in group if bool(row["include_in_reference"])]
        refs.extend(reference_for_group(session_group, structure, group, "trusted_source_median", trusted, min_sources))

        by_method: dict[str, list[float]] = defaultdict(list)
        for row in trusted:
            by_method[str(row["method"])].append(float(row["volume_ml"]))
        method_medians = [median(values) for values in by_method.values()]
        if len(method_medians) >= min_sources:
            refs.append(
                {
                    "reference_variant": "trusted_method_balanced_median",
                    "session_group": session_group,
                    "structure": structure,
                    "n_sources": len(trusted),
                    "n_methods": len(method_medians),
                    "reference_volume_ml": median(method_medians),
                    "source_median_ml": median(method_medians),
                    "source_mad_ml": mad(method_medians),
                    "reference_strength": strength_label(len(trusted), len(method_medians)),
                }
            )
    return refs


def reference_for_group(
    session_group: str,
    structure: str,
    full_group: list[dict[str, str | float | bool]],
    variant: str,
    selected: list[dict[str, str | float | bool]],
    min_sources: int,
) -> list[dict[str, str | float | int]]:
    if len(selected) < min_sources:
        return []
    volumes = [float(row["volume_ml"]) for row in selected]
    methods = {str(row["method"]) for row in selected}
    return [
        {
            "reference_variant": variant,
            "session_group": session_group,
            "structure": structure,
            "n_sources": len(selected),
            "n_methods": len(methods),
            "reference_volume_ml": median(volumes),
            "source_median_ml": median(volumes),
            "source_mad_ml": mad(volumes),
            "reference_strength": strength_label(len(selected), len(methods)),
        }
    ]


def strength_label(n_sources: int, n_methods: int) -> str:
    if n_sources >= 4 and n_methods >= 2:
        return "moderate"
    if n_sources >= 3 and n_methods >= 2:
        return "limited"
    if n_sources >= 2:
        return "weak"
    return "insufficient"


def build_accuracy_rows(
    volume_rows: list[dict[str, str | float | bool]],
    reference_rows: list[dict[str, str | float | int]],
    min_sources: int,
) -> list[dict[str, str | float | int]]:
    refs_by_key = {
        (str(row["reference_variant"]), str(row["session_group"]), str(row["structure"])): row
        for row in reference_rows
    }
    accuracy: list[dict[str, str | float | int]] = []

    for row in volume_rows:
        for variant in ["all_source_median", "trusted_source_median", "trusted_method_balanced_median"]:
            ref = refs_by_key.get((variant, str(row["session_group"]), str(row["structure"])))
            if ref is None:
                continue
            accuracy.append(accuracy_row(row, ref, variant))

        loo_source = leave_one_reference(volume_rows, row, mode="source", min_sources=min_sources)
        if loo_source is not None:
            accuracy.append(accuracy_row(row, loo_source, "trusted_leave_one_source_out"))

        loo_method = leave_one_reference(volume_rows, row, mode="method", min_sources=min_sources)
        if loo_method is not None:
            accuracy.append(accuracy_row(row, loo_method, "trusted_leave_one_method_out"))
    return accuracy


def leave_one_reference(
    rows: list[dict[str, str | float | bool]],
    target: dict[str, str | float | bool],
    mode: str,
    min_sources: int,
) -> dict[str, str | float | int] | None:
    selected = []
    for row in rows:
        if str(row["session_group"]) != str(target["session_group"]):
            continue
        if str(row["structure"]) != str(target["structure"]):
            continue
        if not bool(row["include_in_reference"]):
            continue
        if mode == "source" and row["source_id"] == target["source_id"]:
            continue
        if mode == "method" and row["method"] == target["method"]:
            continue
        selected.append(row)
    if len(selected) < min_sources:
        return None
    volumes = [float(row["volume_ml"]) for row in selected]
    methods = {str(row["method"]) for row in selected}
    return {
        "session_group": str(target["session_group"]),
        "structure": str(target["structure"]),
        "n_sources": len(selected),
        "n_methods": len(methods),
        "reference_volume_ml": median(volumes),
        "source_mad_ml": mad(volumes),
        "reference_strength": strength_label(len(selected), len(methods)),
    }


def accuracy_row(
    source_row: dict[str, str | float | bool],
    ref_row: dict[str, str | float | int],
    variant: str,
) -> dict[str, str | float | int]:
    value = float(source_row["volume_ml"])
    reference = float(ref_row["reference_volume_ml"])
    signed = pct_error(value, reference)
    absolute = abs_rel_error(value, reference)
    return {
        "reference_variant": variant,
        "session_group": str(source_row["session_group"]),
        "structure": str(source_row["structure"]),
        "source_id": str(source_row["source_id"]),
        "method": str(source_row["method"]),
        "scan_id": str(source_row["scan_id"]),
        "include_in_reference": str(bool(source_row["include_in_reference"])).lower(),
        "source_volume_ml": value,
        "reference_volume_ml": reference,
        "signed_error_pct": signed,
        "abs_rel_error_pct": absolute,
        "operational_accuracy_pct": max(0.0, 100.0 - absolute) if not math.isnan(absolute) else math.nan,
        "n_reference_sources": int(ref_row["n_sources"]),
        "n_reference_methods": int(ref_row["n_methods"]),
        "reference_strength": str(ref_row["reference_strength"]),
    }


def summarize_accuracy(
    accuracy_rows: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str | float | int]]] = defaultdict(list)
    for row in accuracy_rows:
        grouped[
            (
                str(row["reference_variant"]),
                str(row["session_group"]),
                str(row["source_id"]),
                str(row["method"]),
                str(row["scan_id"]),
            )
        ].append(row)

    summaries: list[dict[str, str | float | int]] = []
    for (variant, session_group, source_id, method, scan_id), group in sorted(grouped.items()):
        errors = [float(row["abs_rel_error_pct"]) for row in group if not math.isnan(float(row["abs_rel_error_pct"]))]
        if not errors:
            continue
        summaries.append(
            {
                "reference_variant": variant,
                "session_group": session_group,
                "source_id": source_id,
                "method": method,
                "scan_id": scan_id,
                "n_structures": len(errors),
                "median_abs_rel_error_pct": median(errors),
                "p90_abs_rel_error_pct": percentile(errors, 0.90),
                "max_abs_rel_error_pct": max(errors),
                "median_operational_accuracy_pct": max(0.0, 100.0 - median(errors)),
                "interpretation": interpret_median_error(median(errors)),
            }
        )
    return summaries


def interpret_median_error(error: float) -> str:
    if error < 5:
        return "close_volume_match"
    if error < 10:
        return "caution_moderate_volume_disagreement"
    if error < 20:
        return "large_volume_disagreement"
    return "severe_volume_disagreement"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-sources", type=int, default=2)
    args = parser.parse_args()

    sources = read_manifest(args.input_manifest, args.data_root)
    volume_rows = load_volume_rows(sources)
    reference_rows = build_reference_rows(volume_rows, min_sources=args.min_sources)
    accuracy_rows = build_accuracy_rows(volume_rows, reference_rows, min_sources=args.min_sources)
    summary_rows = summarize_accuracy(accuracy_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pseudo_gt_volume_long.csv", volume_rows)
    write_csv(args.output_dir / "pseudo_gt_volume_references.csv", reference_rows)
    write_csv(args.output_dir / "pseudo_gt_volume_accuracy.csv", accuracy_rows)
    write_csv(args.output_dir / "pseudo_gt_volume_source_summary.csv", summary_rows)
    metadata = {
        "analysis": "volume-level pseudo-GT segmentation accuracy",
        "input_manifest": str(args.input_manifest),
        "data_root": str(args.data_root),
        "min_sources": args.min_sources,
        "n_sources": len(sources),
        "n_volume_rows": len(volume_rows),
        "n_reference_rows": len(reference_rows),
        "n_accuracy_rows": len(accuracy_rows),
        "limitations": [
            "Volume-level only; no spatial Dice/HD95/surface Dice without registration.",
            "Pseudo-GT is an operational consensus reference, not anatomical ground truth.",
            "Leave-one-method/source-out variants reduce but do not remove shared model bias.",
        ],
    }
    (args.output_dir / "pseudo_gt_volume_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote pseudo-GT volume evaluation to: {args.output_dir}")


if __name__ == "__main__":
    main()
