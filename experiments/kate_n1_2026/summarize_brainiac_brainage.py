#!/usr/bin/env python3
"""Summarize exploratory BrainIAC brain-age outputs."""

from __future__ import annotations

import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def median(values: list[float]) -> float:
    return float(stats.median(values)) if values else float("nan")


def mean(values: list[float]) -> float:
    return float(stats.fmean(values)) if values else float("nan")


def summarize_group(rows: list[dict[str, str]], group_cols: list[str]) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[tuple(row.get(col, "") for col in group_cols)].append(row)

    output: list[dict[str, str | float | int]] = []
    for key, group_rows in sorted(grouped.items()):
        raw = [as_float(row["raw_model_output"]) for row in group_rows]
        app_years = [as_float(row["predicted_age_years_if_months"]) for row in group_rows]
        raw_years = [as_float(row["predicted_age_years_if_raw_years"]) for row in group_rows]
        chronological = [as_float(row.get("chronological_age_years", "")) for row in group_rows]
        raw_vals = [x for x in raw if x is not None]
        app_vals = [x for x in app_years if x is not None]
        raw_year_vals = [x for x in raw_years if x is not None]
        chronological_vals = [x for x in chronological if x is not None]

        out: dict[str, str | float | int] = dict(zip(group_cols, key))
        out.update(
            {
                "n": len(group_rows),
                "raw_model_output_min": min(raw_vals),
                "raw_model_output_median": median(raw_vals),
                "raw_model_output_max": max(raw_vals),
                "app_years_min": min(app_vals),
                "app_years_median": median(app_vals),
                "app_years_max": max(app_vals),
                "raw_as_years_min": min(raw_year_vals),
                "raw_as_years_median": median(raw_year_vals),
                "raw_as_years_max": max(raw_year_vals),
            }
        )
        if chronological_vals:
            deltas_app = [app - chronological_vals[idx] for idx, app in enumerate(app_vals)]
            deltas_raw_years = [raw_year - chronological_vals[idx] for idx, raw_year in enumerate(raw_year_vals)]
            out.update(
                {
                    "chronological_age_years_min": min(chronological_vals),
                    "chronological_age_years_median": median(chronological_vals),
                    "chronological_age_years_max": max(chronological_vals),
                    "app_years_delta_median": median(deltas_app),
                    "raw_as_years_delta_median": median(deltas_raw_years),
                    "raw_as_years_delta_mean": mean(deltas_raw_years),
                }
            )
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kate-predictions", required=True, type=Path)
    parser.add_argument("--simon-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    kate_rows = read_rows(args.kate_predictions)
    simon_rows = read_rows(args.simon_predictions)
    all_rows = kate_rows + simon_rows

    write_csv(args.output_dir / "brainiac_brainage_branch_summary.csv", summarize_group(all_rows, ["dataset", "branch"]))
    write_csv(args.output_dir / "brainiac_brainage_simon_session_summary.csv", summarize_group(simon_rows, ["session"]))
    write_csv(args.output_dir / "brainiac_brainage_kate_scan_summary.csv", summarize_group(kate_rows, ["scan_id", "branch"]))

    print(f"Wrote summaries under: {args.output_dir}")


if __name__ == "__main__":
    main()
