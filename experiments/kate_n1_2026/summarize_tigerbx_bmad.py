#!/usr/bin/env python3
"""Summarize TIGERBx bmad volume and QC outputs for the Kate n=1 application."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


QC_RE = re.compile(r"^(?P<scan_id>.+)_qc-(?P<score>\d+)\.log$")


def read_volume_rows(path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(row["label"])
            row["volume_ml"] = float(row["volume_ml"])
            rows.append(row)
    return rows


def read_qc_scores(qc_dir: Path) -> dict[str, int]:
    scores: dict[str, int] = {}
    for path in sorted(qc_dir.glob("*_qc-*.log")):
        match = QC_RE.match(path.name)
        if match:
            scores[match.group("scan_id")] = int(match.group("score"))
    return scores


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def write_scan_summary(
    rows: list[dict[str, str | float | int]],
    qc_scores: dict[str, int],
    output_csv: Path,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str | float | int]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scan_id"]), str(row["output_type"]))].append(row)

    out_rows = []
    for (scan_id, output_type), group in sorted(grouped.items()):
        volumes = [float(row["volume_ml"]) for row in group]
        out_rows.append(
            {
                "scan_id": scan_id,
                "output_type": output_type,
                "n_labels": len(volumes),
                "total_ml": f"{sum(volumes):.6f}",
                "median_label_ml": f"{statistics.median(volumes):.6f}",
                "qc_score": qc_scores.get(scan_id, ""),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scan_id",
                "output_type",
                "n_labels",
                "total_ml",
                "median_label_ml",
                "qc_score",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def write_pairwise_summary(
    rows: list[dict[str, str | float | int]],
    output_csv: Path,
) -> None:
    output_types = sorted({str(row["output_type"]) for row in rows})
    out_rows = []
    for output_type in output_types:
        by_scan: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            if str(row["output_type"]) != output_type:
                continue
            by_scan[str(row["scan_id"])][int(row["label"])] = float(row["volume_ml"])

        for scan_a, scan_b in combinations(sorted(by_scan), 2):
            common = sorted(set(by_scan[scan_a]) & set(by_scan[scan_b]))
            diffs = []
            for label in common:
                vol_a = by_scan[scan_a][label]
                vol_b = by_scan[scan_b][label]
                mean_volume = (vol_a + vol_b) / 2.0
                if mean_volume > 0:
                    diffs.append(abs(vol_a - vol_b) / mean_volume * 100.0)
            total_a = sum(by_scan[scan_a].values())
            total_b = sum(by_scan[scan_b].values())
            total_mean = (total_a + total_b) / 2.0
            total_abs_rel = abs(total_a - total_b) / total_mean * 100.0 if total_mean else ""
            out_rows.append(
                {
                    "output_type": output_type,
                    "scan_a": scan_a,
                    "scan_b": scan_b,
                    "n_common_labels": len(common),
                    "median_abs_rel_diff_pct": f"{statistics.median(diffs):.6f}" if diffs else "",
                    "p90_abs_rel_diff_pct": f"{percentile(diffs, 0.9):.6f}" if diffs else "",
                    "max_abs_rel_diff_pct": f"{max(diffs):.6f}" if diffs else "",
                    "total_a_ml": f"{total_a:.6f}",
                    "total_b_ml": f"{total_b:.6f}",
                    "total_abs_rel_diff_pct": f"{total_abs_rel:.6f}" if total_abs_rel != "" else "",
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_type",
                "scan_a",
                "scan_b",
                "n_common_labels",
                "median_abs_rel_diff_pct",
                "p90_abs_rel_diff_pct",
                "max_abs_rel_diff_pct",
                "total_a_ml",
                "total_b_ml",
                "total_abs_rel_diff_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-csv", required=True, type=Path)
    parser.add_argument("--qc-dir", required=True, type=Path)
    parser.add_argument("--scan-summary-csv", required=True, type=Path)
    parser.add_argument("--pairwise-csv", required=True, type=Path)
    args = parser.parse_args()

    rows = read_volume_rows(args.volume_csv)
    qc_scores = read_qc_scores(args.qc_dir)
    write_scan_summary(rows, qc_scores, args.scan_summary_csv)
    write_pairwise_summary(rows, args.pairwise_csv)
    print(f"Wrote scan summary: {args.scan_summary_csv}")
    print(f"Wrote pairwise summary: {args.pairwise_csv}")


if __name__ == "__main__":
    main()
