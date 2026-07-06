#!/usr/bin/env python3
"""Join NeuroFM outputs with input metadata and write compact summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        result = float(value)
        if math.isfinite(result):
            return result
    except ValueError:
        pass
    return None


def summarize(rows: list[dict[str, str]], summary_id: str, interpretation: str) -> dict[str, str]:
    pairs = []
    predictions = []
    for row in rows:
        pred = as_float(row.get("predicted_brain_age_years", ""))
        chron = as_float(row.get("chronological_age_years", ""))
        if pred is not None:
            predictions.append(pred)
        if pred is not None and chron is not None:
            pairs.append((pred, chron))

    out = {
        "summary_id": summary_id,
        "n_rows": str(len(rows)),
        "n_predictions": str(len(predictions)),
        "n_with_chronological_age": str(len(pairs)),
        "mean_predicted_brain_age_years": "",
        "min_predicted_brain_age_years": "",
        "max_predicted_brain_age_years": "",
        "mae_years": "",
        "bias_years_pred_minus_chron": "",
        "rmse_years": "",
        "pearson_r": "",
        "claim_level": "application_branch_not_validation_claim",
        "interpretation": interpretation,
    }
    if predictions:
        arr = np.asarray(predictions, dtype=float)
        out["mean_predicted_brain_age_years"] = f"{float(arr.mean()):.6g}"
        out["min_predicted_brain_age_years"] = f"{float(arr.min()):.6g}"
        out["max_predicted_brain_age_years"] = f"{float(arr.max()):.6g}"
    if pairs:
        pred = np.asarray([p for p, _ in pairs], dtype=float)
        chron = np.asarray([c for _, c in pairs], dtype=float)
        diff = pred - chron
        out["mae_years"] = f"{float(np.mean(np.abs(diff))):.6g}"
        out["bias_years_pred_minus_chron"] = f"{float(np.mean(diff)):.6g}"
        out["rmse_years"] = f"{float(np.sqrt(np.mean(diff**2))):.6g}"
        if len(pairs) >= 2 and float(np.std(pred)) > 0 and float(np.std(chron)) > 0:
            out["pearson_r"] = f"{float(np.corrcoef(pred, chron)[0, 1]):.6g}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-summary", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--summary-id", required=True)
    parser.add_argument("--method", default="NeuroFM")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--variant", default="neurofm-s")
    parser.add_argument("--weights-cache-dir", default="")
    parser.add_argument("--interpretation", required=True)
    args = parser.parse_args()

    result_rows = read_rows(args.results_summary)
    manifest_rows = read_rows(args.input_manifest)
    manifest_by_input = {row["input"]: row for row in manifest_rows if row.get("input")}

    joined = []
    for result in result_rows:
        input_path = result.get("input", "")
        meta = manifest_by_input.get(input_path, {})
        row = dict(meta)
        row.update(
            {
                "method": args.method,
                "source_repo": args.source_repo,
                "source_commit": args.source_commit,
                "variant": args.variant,
                "neurofm_input": input_path,
                "predicted_brain_age_years": result.get("brain_age", ""),
                "predicted_sex_binary": result.get("sex", ""),
                "predicted_ventricle_volume_mm3": result.get("ventricle_volume", ""),
                "predicted_brain_volume_mm3": result.get("brain_volume", ""),
                "claim_level": "application_branch_not_validation_claim",
                "interpretation_guard": (
                    "NeuroFM output is a model estimate on this preprocessing branch; "
                    "it is not a segmentation, morphometry, clinical, or validation claim."
                ),
            }
        )
        pred = as_float(row.get("predicted_brain_age_years", ""))
        chron = as_float(row.get("chronological_age_years", ""))
        if pred is not None and chron is not None:
            row["brain_age_delta_years"] = f"{pred - chron:.6g}"
        else:
            row["brain_age_delta_years"] = ""
        joined.append(row)

    write_csv(args.output_csv, joined)
    write_csv(args.summary_csv, [summarize(joined, args.summary_id, args.interpretation)])
    metadata = {
        "method": args.method,
        "source_repo": args.source_repo,
        "source_commit": args.source_commit,
        "variant": args.variant,
        "weights_cache_dir": args.weights_cache_dir,
        "input_manifest": str(args.input_manifest),
        "results_summary": str(args.results_summary),
        "output_csv": str(args.output_csv),
        "summary_csv": str(args.summary_csv),
        "n_output_rows": len(joined),
        "critical_interpretation": (
            "NeuroFM brain-health outputs are application-branch model outputs. "
            "They do not prove segmentation quality, morphometric validity, or individual clinical brain health."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
