#!/usr/bin/env python3
"""Summarize NeuroFM SIMON HD-BET and perturbation QC outputs.

These summaries are application/QC diagnostics. They do not validate NeuroFM as
a calibrated brain-age model for SIMON or Kate.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "kate_n1_2026"

HDBET_PRED = DATA_DIR / "neurofm_simon_hdbet_predictions.csv"
HDBET_QC = DATA_DIR / "neurofm_simon_hdbet_qc_summary.csv"
STABILITY_PRED = DATA_DIR / "neurofm_simon_stability_perturbation_predictions.csv"
STABILITY_DELTAS = DATA_DIR / "neurofm_simon_stability_perturbation_deltas.csv"
STABILITY_SUMMARY = DATA_DIR / "neurofm_simon_stability_perturbation_delta_summary.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
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


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def summary_stats(values: list[float], prefix: str) -> dict[str, str]:
    if not values:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_sd": "",
            f"{prefix}_min": "",
            f"{prefix}_median": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_mean": fmt(statistics.mean(values)),
        f"{prefix}_sd": fmt(statistics.stdev(values) if len(values) > 1 else 0.0),
        f"{prefix}_min": fmt(min(values)),
        f"{prefix}_median": fmt(statistics.median(values)),
        f"{prefix}_max": fmt(max(values)),
    }


def summarize_age_rows(rows: list[dict[str, str]], group: str, notes: str) -> dict[str, str]:
    preds = [as_float(row, "predicted_brain_age_years") for row in rows]
    chrons = [as_float(row, "chronological_age_years") for row in rows]
    volumes = [as_float(row, "predicted_brain_volume_mm3") for row in rows]
    masks = [as_float(row, "mask_fraction") for row in rows]
    pred_pairs = [(pred, chron) for pred, chron in zip(preds, chrons) if pred is not None and chron is not None]
    pred_values = [pred for pred in preds if pred is not None]
    deltas = [pred - chron for pred, chron in pred_pairs]

    out = {
        "group": group,
        "n_rows": str(len(rows)),
        "n_predictions": str(len(pred_values)),
        "mae_years": "",
        "bias_years_pred_minus_chron": "",
        "rmse_years": "",
        "pearson_r_age_vs_prediction": "",
        "pearson_r_prediction_vs_neurofm_brain_volume": "",
        "pearson_r_prediction_vs_hdbet_mask_fraction": "",
        "sex_0_count": str(sum(1 for row in rows if row.get("predicted_sex_binary") == "0.0")),
        "sex_1_count": str(sum(1 for row in rows if row.get("predicted_sex_binary") == "1.0")),
        "interpretation": notes,
    }
    out.update(summary_stats(pred_values, "predicted_age_years"))
    out.update(summary_stats([v for v in volumes if v is not None], "predicted_brain_volume_mm3"))
    out.update(summary_stats([v for v in masks if v is not None], "hdbet_mask_fraction"))

    if deltas:
        out["mae_years"] = fmt(statistics.mean(abs(v) for v in deltas))
        out["bias_years_pred_minus_chron"] = fmt(statistics.mean(deltas))
        out["rmse_years"] = fmt(math.sqrt(statistics.mean(v * v for v in deltas)))
        out["pearson_r_age_vs_prediction"] = fmt(
            pearson([chron for _, chron in pred_pairs], [pred for pred, _ in pred_pairs])
        )

    volume_pairs = [
        (pred, volume)
        for pred, volume in zip(preds, volumes)
        if pred is not None and volume is not None
    ]
    mask_pairs = [
        (pred, mask)
        for pred, mask in zip(preds, masks)
        if pred is not None and mask is not None
    ]
    out["pearson_r_prediction_vs_neurofm_brain_volume"] = fmt(
        pearson([p for p, _ in volume_pairs], [v for _, v in volume_pairs])
    )
    out["pearson_r_prediction_vs_hdbet_mask_fraction"] = fmt(
        pearson([p for p, _ in mask_pairs], [m for _, m in mask_pairs])
    )
    return out


def summarize_hdbet() -> None:
    rows = read_rows(HDBET_PRED)
    conform = [row for row in rows if row.get("output_zooms_mm") == "1x1x1"]
    resampled = [row for row in rows if row.get("output_zooms_mm") != "1x1x1"]
    summary_rows = [
        summarize_age_rows(
            rows,
            "all_hdbet",
            "All 94 HD-BET skull-stripped SIMON FastSurfer orig.mgz derivatives; sanity/QC only.",
        ),
        summarize_age_rows(
            conform,
            "hdbet_1mm_inputs",
            "Inputs already at 1x1x1 mm after HD-BET; no NeuroFM voxel-size warning expected.",
        ),
        summarize_age_rows(
            resampled,
            "hdbet_non_1mm_inputs",
            "Inputs with non-1mm voxel sizes; NeuroFM internally resamples/reorients, which may affect accuracy.",
        ),
    ]
    write_csv(HDBET_QC, summary_rows)


def perturbation_family(name: str) -> str:
    if name.startswith("brain_size"):
        return "brain_size"
    if name.startswith("resample_roundtrip"):
        return "resample_roundtrip"
    if name.startswith("rotate_z"):
        return "rotation"
    return name


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_stability() -> None:
    rows = read_rows(STABILITY_PRED)
    baselines = {row.get("source_scan_id", ""): row for row in rows if row.get("perturbation") == "baseline"}
    delta_rows: list[dict[str, str]] = []

    for row in rows:
        perturbation = row.get("perturbation", "")
        source_scan_id = row.get("source_scan_id", "")
        baseline = baselines.get(source_scan_id)
        if perturbation == "baseline" or baseline is None:
            continue
        pred = as_float(row, "predicted_brain_age_years")
        base_pred = as_float(baseline, "predicted_brain_age_years")
        volume = as_float(row, "predicted_brain_volume_mm3")
        base_volume = as_float(baseline, "predicted_brain_volume_mm3")
        if pred is None or base_pred is None:
            continue
        volume_delta = None if volume is None or base_volume is None else volume - base_volume
        volume_delta_pct = None
        if volume_delta is not None and base_volume not in (None, 0):
            volume_delta_pct = 100 * volume_delta / base_volume
        delta_rows.append(
            {
                "source_scan_id": source_scan_id,
                "chronological_age_years": row.get("chronological_age_years", ""),
                "perturbation": perturbation,
                "perturbation_family": perturbation_family(perturbation),
                "perturbation_value": row.get("perturbation_value", ""),
                "baseline_predicted_age_years": fmt(base_pred),
                "perturbed_predicted_age_years": fmt(pred),
                "delta_years": fmt(pred - base_pred),
                "abs_delta_years": fmt(abs(pred - base_pred)),
                "baseline_predicted_sex_binary": baseline.get("predicted_sex_binary", ""),
                "perturbed_predicted_sex_binary": row.get("predicted_sex_binary", ""),
                "sex_flip": "1"
                if baseline.get("predicted_sex_binary", "") != row.get("predicted_sex_binary", "")
                else "0",
                "baseline_predicted_brain_volume_mm3": fmt(base_volume),
                "perturbed_predicted_brain_volume_mm3": fmt(volume),
                "brain_volume_delta_mm3": fmt(volume_delta),
                "brain_volume_delta_pct": fmt(volume_delta_pct),
                "claim_level": "stability_qc_not_validation_claim",
            }
        )

    write_csv(STABILITY_DELTAS, delta_rows)

    summary_rows: list[dict[str, str]] = []
    groups = sorted({row["perturbation"] for row in delta_rows}) + sorted(
        {row["perturbation_family"] for row in delta_rows}
    )
    for group in groups:
        if group in {row["perturbation_family"] for row in delta_rows}:
            members = [row for row in delta_rows if row["perturbation_family"] == group]
            group_type = "family"
        else:
            members = [row for row in delta_rows if row["perturbation"] == group]
            group_type = "perturbation"
        deltas = [float(row["delta_years"]) for row in members]
        abs_deltas = [float(row["abs_delta_years"]) for row in members]
        volume_pct = [float(row["brain_volume_delta_pct"]) for row in members if row["brain_volume_delta_pct"]]
        summary_rows.append(
            {
                "group": group,
                "group_type": group_type,
                "n": str(len(members)),
                "mean_delta_years": fmt(statistics.mean(deltas) if deltas else None),
                "mean_abs_delta_years": fmt(statistics.mean(abs_deltas) if abs_deltas else None),
                "median_abs_delta_years": fmt(statistics.median(abs_deltas) if abs_deltas else None),
                "p90_abs_delta_years": fmt(percentile(abs_deltas, 0.9)),
                "max_abs_delta_years": fmt(max(abs_deltas) if abs_deltas else None),
                "sex_flip_count": str(sum(1 for row in members if row["sex_flip"] == "1")),
                "mean_brain_volume_delta_pct": fmt(statistics.mean(volume_pct) if volume_pct else None),
                "claim_level": "stability_qc_not_validation_claim",
            }
        )
    write_csv(STABILITY_SUMMARY, summary_rows)


def main() -> None:
    summarize_hdbet()
    summarize_stability()
    print(f"Wrote {HDBET_QC}")
    print(f"Wrote {STABILITY_DELTAS}")
    print(f"Wrote {STABILITY_SUMMARY}")


if __name__ == "__main__":
    main()
