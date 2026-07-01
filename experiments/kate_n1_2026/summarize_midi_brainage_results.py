#!/usr/bin/env python3
"""Summarize MIDIBrainAge batch predictions with explicit claim labels."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDNAMES = [
    "summary_id",
    "group_columns",
    "group_values",
    "n_manifest",
    "n_ok",
    "n_failed",
    "chronological_age_available",
    "chronological_age_min_years",
    "chronological_age_max_years",
    "predicted_age_mean_years",
    "predicted_age_sd_years",
    "predicted_age_min_years",
    "predicted_age_max_years",
    "brain_age_delta_mean_years",
    "brain_age_delta_sd_years",
    "mae_years",
    "median_absolute_error_years",
    "rmse_years",
    "pearson_r",
    "slope_predicted_vs_chronological",
    "intercept_predicted_vs_chronological",
    "elapsed_mean_seconds",
    "elapsed_min_seconds",
    "elapsed_max_seconds",
    "claim_level",
    "interpretation",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isfinite(number):
        return number
    return None


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_ss = sum(x * x for x in x_centered)
    y_ss = sum(y * y for y in y_centered)
    if x_ss == 0 or y_ss == 0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / math.sqrt(x_ss * y_ss)


def slope_intercept(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None, None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    if x_var == 0:
        return None, None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / x_var
    intercept = y_mean - slope * x_mean
    return slope, intercept


def summarize_group(
    rows: list[dict[str, str]],
    *,
    summary_id: str,
    group_columns: list[str],
    group_values: tuple[str, ...],
    claim_level: str,
    interpretation: str,
) -> dict[str, str]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    pred = [value for row in ok_rows if (value := parse_float(row.get("predicted_age_years"))) is not None]
    elapsed = [value for row in ok_rows if (value := parse_float(row.get("elapsed_seconds"))) is not None]

    paired: list[tuple[float, float]] = []
    for row in ok_rows:
        chronological_age = parse_float(row.get("chronological_age_years"))
        predicted_age = parse_float(row.get("predicted_age_years"))
        if chronological_age is not None and predicted_age is not None:
            paired.append((chronological_age, predicted_age))

    chronological = [chronological_age for chronological_age, _ in paired]
    predicted_paired = [predicted_age for _, predicted_age in paired]
    deltas = [predicted_age - chronological_age for chronological_age, predicted_age in paired]
    abs_errors = [abs(delta) for delta in deltas]
    squared_errors = [delta * delta for delta in deltas]
    slope, intercept = slope_intercept(chronological, predicted_paired)

    return {
        "summary_id": summary_id,
        "group_columns": "|".join(group_columns),
        "group_values": "|".join(group_values),
        "n_manifest": str(len(rows)),
        "n_ok": str(len(ok_rows)),
        "n_failed": str(len(failed_rows)),
        "chronological_age_available": "yes" if paired else "no",
        "chronological_age_min_years": fmt(min(chronological) if chronological else None),
        "chronological_age_max_years": fmt(max(chronological) if chronological else None),
        "predicted_age_mean_years": fmt(mean(pred)),
        "predicted_age_sd_years": fmt(sd(pred)),
        "predicted_age_min_years": fmt(min(pred) if pred else None),
        "predicted_age_max_years": fmt(max(pred) if pred else None),
        "brain_age_delta_mean_years": fmt(mean(deltas)),
        "brain_age_delta_sd_years": fmt(sd(deltas)),
        "mae_years": fmt(mean(abs_errors)),
        "median_absolute_error_years": fmt(statistics.median(abs_errors) if abs_errors else None),
        "rmse_years": fmt(math.sqrt(statistics.fmean(squared_errors)) if squared_errors else None),
        "pearson_r": fmt(pearson(chronological, predicted_paired)),
        "slope_predicted_vs_chronological": fmt(slope),
        "intercept_predicted_vs_chronological": fmt(intercept),
        "elapsed_mean_seconds": fmt(mean(elapsed)),
        "elapsed_min_seconds": fmt(min(elapsed) if elapsed else None),
        "elapsed_max_seconds": fmt(max(elapsed) if elapsed else None),
        "claim_level": claim_level,
        "interpretation": interpretation,
    }


def group_rows(rows: list[dict[str, str]], group_columns: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    if not group_columns:
        return {(): rows}
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(column, "") for column in group_columns)].append(row)
    return dict(sorted(grouped.items()))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-id", required=True)
    parser.add_argument("--group-cols", nargs="*", default=[])
    parser.add_argument("--claim-level", required=True)
    parser.add_argument("--interpretation", required=True)
    args = parser.parse_args()

    rows = read_rows(args.predictions_csv)
    summary_rows = [
        summarize_group(
            group,
            summary_id=args.summary_id,
            group_columns=args.group_cols,
            group_values=key,
            claim_level=args.claim_level,
            interpretation=args.interpretation,
        )
        for key, group in group_rows(rows, args.group_cols).items()
    ]
    write_rows(args.output_csv, summary_rows)
    print(f"Wrote {len(summary_rows)} summary rows to {args.output_csv}")


if __name__ == "__main__":
    main()
