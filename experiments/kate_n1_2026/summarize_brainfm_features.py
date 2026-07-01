#!/usr/bin/env python3
"""Summarize BrainFM feature-only outputs for QC/protocol-sensitivity review."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


FEATURE_RE = re.compile(r"^feat_l(?P<level>\d+)_(?P<stat>mean|std)_\d{4}$")
META_FIELDS = [
    "scan_id",
    "session",
    "modality_hint",
    "analysis_role",
    "image_path",
    "image_sha256",
    "prepared_shape",
    "original_shape_after_orientation",
]


def modality_family(modality_hint: str) -> str:
    if modality_hint.startswith("T1w"):
        return "t1w"
    if modality_hint.startswith("T2_FLAIR"):
        return "flair"
    if modality_hint.startswith("T2_TSE"):
        return "t2_tse"
    if modality_hint.startswith("3D_inversion_recovery"):
        return "ir_like_3d"
    return "other"


def comparison_tag(row_a: dict[str, str], row_b: dict[str, str]) -> str:
    same_session = row_a["session"] == row_b["session"]
    family_a = modality_family(row_a["modality_hint"])
    family_b = modality_family(row_b["modality_hint"])
    same_family = family_a == family_b

    if same_session and same_family:
        return "same_session_same_family"
    if same_session:
        return "same_session_cross_family"
    if same_family:
        return "cross_session_same_family"
    return "cross_session_cross_family"


def read_feature_rows(path: Path) -> tuple[list[dict[str, object]], list[str], dict[int, list[int]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        feature_cols = [col for col in reader.fieldnames if col.startswith("feat_")]
        layer_indices: dict[int, list[int]] = defaultdict(list)
        for idx, col in enumerate(feature_cols):
            match = FEATURE_RE.match(col)
            if match:
                layer_indices[int(match.group("level"))].append(idx)

        rows = []
        for row in reader:
            features = [float(row[col]) for col in feature_cols]
            meta = {field: row.get(field, "") for field in META_FIELDS}
            meta["modality_family"] = modality_family(str(meta["modality_hint"]))
            rows.append({"meta": meta, "features": features})

    if not rows:
        raise ValueError(f"No feature rows found: {path}")
    return rows, feature_cols, dict(layer_indices)


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine_distance(values_a: list[float], values_b: list[float]) -> float:
    norm_a = vector_norm(values_a)
    norm_b = vector_norm(values_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    dot = sum(a * b for a, b in zip(values_a, values_b))
    return 1.0 - dot / (norm_a * norm_b)


def euclidean_distance(values_a: list[float], values_b: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(values_a, values_b)))


def subset(values: list[float], indices: list[int]) -> list[float]:
    return [values[index] for index in indices]


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.9f}"


def write_scan_summary(rows: list[dict[str, object]], feature_count: int, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scan_id",
            "session",
            "modality_hint",
            "modality_family",
            "analysis_role",
            "prepared_shape",
            "original_shape_after_orientation",
            "feature_count",
            "feature_mean",
            "feature_sd",
            "feature_l2_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row["meta"]
            features = row["features"]
            assert isinstance(meta, dict)
            assert isinstance(features, list)
            writer.writerow(
                {
                    "scan_id": meta["scan_id"],
                    "session": meta["session"],
                    "modality_hint": meta["modality_hint"],
                    "modality_family": meta["modality_family"],
                    "analysis_role": meta["analysis_role"],
                    "prepared_shape": meta["prepared_shape"],
                    "original_shape_after_orientation": meta["original_shape_after_orientation"],
                    "feature_count": feature_count,
                    "feature_mean": format_float(statistics.fmean(features)),
                    "feature_sd": format_float(statistics.pstdev(features)),
                    "feature_l2_norm": format_float(vector_norm(features)),
                }
            )


def build_pairwise_rows(
    rows: list[dict[str, object]],
    feature_count: int,
    layer_indices: dict[int, list[int]],
) -> list[dict[str, str]]:
    pair_rows: list[dict[str, str]] = []
    for row_a, row_b in combinations(rows, 2):
        meta_a = row_a["meta"]
        meta_b = row_b["meta"]
        features_a = row_a["features"]
        features_b = row_b["features"]
        assert isinstance(meta_a, dict)
        assert isinstance(meta_b, dict)
        assert isinstance(features_a, list)
        assert isinstance(features_b, list)

        out = {
            "scan_a": str(meta_a["scan_id"]),
            "scan_b": str(meta_b["scan_id"]),
            "session_a": str(meta_a["session"]),
            "session_b": str(meta_b["session"]),
            "modality_a": str(meta_a["modality_hint"]),
            "modality_b": str(meta_b["modality_hint"]),
            "modality_family_a": str(meta_a["modality_family"]),
            "modality_family_b": str(meta_b["modality_family"]),
            "analysis_role_a": str(meta_a["analysis_role"]),
            "analysis_role_b": str(meta_b["analysis_role"]),
            "comparison_tag": comparison_tag(meta_a, meta_b),
            "same_session": str(meta_a["session"] == meta_b["session"]).lower(),
            "same_modality_family": str(meta_a["modality_family"] == meta_b["modality_family"]).lower(),
            "feature_count": str(feature_count),
            "cosine_distance": format_float(cosine_distance(features_a, features_b)),
            "euclidean_distance": format_float(euclidean_distance(features_a, features_b)),
        }
        for level in sorted(layer_indices):
            indices = layer_indices[level]
            out[f"cosine_distance_l{level}"] = format_float(
                cosine_distance(subset(features_a, indices), subset(features_b, indices))
            )
        pair_rows.append(out)

    sorted_cosines = sorted(float(row["cosine_distance"]) for row in pair_rows if row["cosine_distance"])
    denom = max(len(sorted_cosines) - 1, 1)
    for row in pair_rows:
        value = float(row["cosine_distance"])
        rank = sorted_cosines.index(value)
        row["cosine_distance_percentile"] = format_float(rank / denom)

    return pair_rows


def write_pairwise_rows(pair_rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "scan_a",
        "scan_b",
        "session_a",
        "session_b",
        "modality_a",
        "modality_b",
        "modality_family_a",
        "modality_family_b",
        "analysis_role_a",
        "analysis_role_b",
        "comparison_tag",
        "same_session",
        "same_modality_family",
        "feature_count",
        "cosine_distance",
        "euclidean_distance",
        "cosine_distance_percentile",
    ]
    layer_fields = sorted(
        [field for field in pair_rows[0] if field.startswith("cosine_distance_l")],
        key=lambda item: int(item.replace("cosine_distance_l", "")),
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + layer_fields)
        writer.writeheader()
        writer.writerows(pair_rows)


def median_or_blank(values: list[float]) -> str:
    return format_float(statistics.median(values)) if values else ""


def write_contrast_summary(pair_rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def distances(predicate) -> list[float]:
        return [float(row["cosine_distance"]) for row in pair_rows if predicate(row)]

    def add_summary(out_rows: list[dict[str, str]], label: str, predicate, interpretation: str) -> None:
        values = distances(predicate)
        out_rows.append(
            {
                "contrast": label,
                "n_pairs": str(len(values)),
                "min_cosine_distance": format_float(min(values)) if values else "",
                "median_cosine_distance": median_or_blank(values),
                "max_cosine_distance": format_float(max(values)) if values else "",
                "interpretation_guardrail": interpretation,
            }
        )

    out_rows: list[dict[str, str]] = []
    add_summary(
        out_rows,
        "all_pairs",
        lambda row: True,
        "Distribution of BrainFM feature distances only; not anatomical accuracy.",
    )
    add_summary(
        out_rows,
        "same_session_same_family",
        lambda row: row["comparison_tag"] == "same_session_same_family",
        "Closest expected technical/protocol-near contrasts; still not a repeatability proof.",
    )
    add_summary(
        out_rows,
        "same_session_cross_family",
        lambda row: row["comparison_tag"] == "same_session_cross_family",
        "Within-session modality/protocol sensitivity.",
    )
    add_summary(
        out_rows,
        "cross_session_same_family",
        lambda row: row["comparison_tag"] == "cross_session_same_family",
        "Cross-session feature shift among broad modality families.",
    )
    add_summary(
        out_rows,
        "primary_2018_2022_2024_probe",
        lambda row: set([row["scan_a"], row["scan_b"]])
        <= {"kate_2018_t1", "kate_2022_t1", "kate_2024_3di"},
        "Primary/probe trio mixes biological time, scanner, contrast, and slice protocol.",
    )
    add_summary(
        out_rows,
        "2024_ffe_401_vs_601",
        lambda row: {row["scan_a"], row["scan_b"]} == {"kate_2024_t1_ffe_401", "kate_2024_t1_ffe_601"},
        "Same-session alternative T1w FFE contrast check.",
    )
    add_summary(
        out_rows,
        "2024_3di_vs_2024_ffe",
        lambda row: (
            "kate_2024_3di" in {row["scan_a"], row["scan_b"]}
            and bool({"kate_2024_t1_ffe_401", "kate_2024_t1_ffe_601"} & {row["scan_a"], row["scan_b"]})
        ),
        "2024 3DI versus alternative 2024 T1w FFE candidates; a QC contrast, not rescue validation.",
    )
    add_summary(
        out_rows,
        "2024_t2_501_vs_801",
        lambda row: {row["scan_a"], row["scan_b"]} == {"kate_2024_t2_501", "kate_2024_t2_801"},
        "Same-session T2 TSE alternative check.",
    )
    add_summary(
        out_rows,
        "2022_t1_vs_secondary",
        lambda row: (
            "kate_2022_t1" in {row["scan_a"], row["scan_b"]}
            and (row["session_a"] == "2022" and row["session_b"] == "2022")
            and {row["scan_a"], row["scan_b"]} != {"kate_2022_t1"}
        ),
        "2022 thick-slice T1 versus secondary modalities.",
    )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "contrast",
                "n_pairs",
                "min_cosine_distance",
                "median_cosine_distance",
                "max_cosine_distance",
                "interpretation_guardrail",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True, type=Path)
    parser.add_argument("--scan-summary-csv", required=True, type=Path)
    parser.add_argument("--pairwise-csv", required=True, type=Path)
    parser.add_argument("--contrast-summary-csv", required=True, type=Path)
    args = parser.parse_args()

    rows, feature_cols, layer_indices = read_feature_rows(args.features_csv)
    pair_rows = build_pairwise_rows(rows, len(feature_cols), layer_indices)
    write_scan_summary(rows, len(feature_cols), args.scan_summary_csv)
    write_pairwise_rows(pair_rows, args.pairwise_csv)
    write_contrast_summary(pair_rows, args.contrast_summary_csv)

    print(f"Wrote scan summary: {args.scan_summary_csv}")
    print(f"Wrote pairwise distances: {args.pairwise_csv}")
    print(f"Wrote contrast summary: {args.contrast_summary_csv}")


if __name__ == "__main__":
    main()
