"""Adversarial ranking analysis for one-photo avatar baselines.

This script treats the MRI target as an uncertainty set, not as a fixed truth.
It asks whether a ranking claim survives all plausible MRI target candidates.

Definitions:
- gap_hd95 = HD95(MediaPipe -> MRI) - HD95(3DDFA -> MRI)
  Positive values favor 3DDFA; negative values favor MediaPipe.
- A ranking is robust only if the worst-case gap has the same sign and is
  larger than the MRI-target and method-disagreement uncertainty scale.

The goal is falsification: if a small change in target-cleaning destroys the
ranking, the ranking is not supported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_json(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def f(row: dict, key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def build_rows(qc_rows: list[dict], avatar_rows: list[dict], method_disagreement: dict) -> list[dict]:
    qc_by_id = {row["candidate_id"]: row for row in qc_rows if row.get("candidate_id")}
    avatar_by_id: dict[str, dict[str, dict]] = {}
    for row in avatar_rows:
        avatar_by_id.setdefault(row["candidate_id"], {})[row["method"]] = row

    method_hd95 = float(method_disagreement["boundary_hd95_mm"])
    method_assd = float(method_disagreement["boundary_assd_mm"])
    out = []
    for candidate_id, by_method in sorted(avatar_by_id.items()):
        if "3ddfa_v2" not in by_method or "mediapipe" not in by_method or candidate_id not in qc_by_id:
            continue
        qc = qc_by_id[candidate_id]
        if str(qc.get("pass_mri_only_gates", "")).lower() != "true":
            continue
        m3 = by_method["3ddfa_v2"]
        mp = by_method["mediapipe"]
        hd95_3 = f(m3, "boundary_hd95_mm")
        hd95_mp = f(mp, "boundary_hd95_mm")
        assd_3 = f(m3, "boundary_assd_mm")
        assd_mp = f(mp, "boundary_assd_mm")
        dice_3 = f(m3, "dice")
        dice_mp = f(mp, "dice")
        target_reference_hd95 = f(qc, "to_reference_hd95_mm")
        target_crop_hd95 = f(qc, "to_unsmoothed_crop_hd95_mm")
        target_uncertainty = max(target_reference_hd95, target_crop_hd95)
        uncertainty_scale = max(method_hd95, target_uncertainty)
        gap_hd95 = hd95_mp - hd95_3
        gap_assd = assd_mp - assd_3
        gap_dice = dice_3 - dice_mp
        out.append(
            {
                "candidate_id": candidate_id,
                "shape": qc.get("shape", ""),
                "margin_fraction": f(qc, "margin_fraction"),
                "taubin_iters": f(qc, "taubin_iters"),
                "subdivide_iters": f(qc, "subdivide_iters"),
                "mri_only_score": f(qc, "mri_only_score"),
                "roughness_median_mm": f(qc, "roughness_median_mm"),
                "stripe_highfreq_ratio": f(qc, "stripe_highfreq_ratio"),
                "nose_prominence_nearest_xz_mm": f(qc, "nose_prominence_nearest_xz_mm"),
                "target_reference_hd95_mm": target_reference_hd95,
                "target_crop_hd95_mm": target_crop_hd95,
                "target_uncertainty_hd95_mm": target_uncertainty,
                "method_disagreement_hd95_mm": method_hd95,
                "uncertainty_scale_hd95_mm": uncertainty_scale,
                "3ddfa_hd95_mm": hd95_3,
                "mediapipe_hd95_mm": hd95_mp,
                "gap_hd95_mp_minus_3ddfa_mm": gap_hd95,
                "gap_hd95_over_uncertainty": gap_hd95 / max(uncertainty_scale, 1e-12),
                "3ddfa_assd_mm": assd_3,
                "mediapipe_assd_mm": assd_mp,
                "gap_assd_mp_minus_3ddfa_mm": gap_assd,
                "gap_assd_over_method_disagreement": gap_assd / max(method_assd, 1e-12),
                "3ddfa_dice": dice_3,
                "mediapipe_dice": dice_mp,
                "gap_dice_3ddfa_minus_mp": gap_dice,
                "winner_by_hd95": "3ddfa_v2" if gap_hd95 > 0 else "mediapipe" if gap_hd95 < 0 else "tie",
                "winner_by_assd": "3ddfa_v2" if gap_assd > 0 else "mediapipe" if gap_assd < 0 else "tie",
                "winner_by_dice": "3ddfa_v2" if gap_dice > 0 else "mediapipe" if gap_dice < 0 else "tie",
            }
        )
    return out


def adversarial_objectives(rows: list[dict], lambdas: list[float]) -> dict:
    if not rows:
        return {"error": "No rows"}
    min_score = min(float(row["mri_only_score"]) for row in rows)
    out = {
        "n_feasible_targets_with_avatar_metrics": len(rows),
        "gap_hd95_min_mm": min(float(row["gap_hd95_mp_minus_3ddfa_mm"]) for row in rows),
        "gap_hd95_max_mm": max(float(row["gap_hd95_mp_minus_3ddfa_mm"]) for row in rows),
        "gap_hd95_median_mm": float(np.median([float(row["gap_hd95_mp_minus_3ddfa_mm"]) for row in rows])),
        "gap_hd95_over_uncertainty_max": max(float(row["gap_hd95_over_uncertainty"]) for row in rows),
        "gap_hd95_over_uncertainty_min": min(float(row["gap_hd95_over_uncertainty"]) for row in rows),
        "hd95_winner_counts": {
            "3ddfa_v2": sum(1 for row in rows if row["winner_by_hd95"] == "3ddfa_v2"),
            "mediapipe": sum(1 for row in rows if row["winner_by_hd95"] == "mediapipe"),
            "tie": sum(1 for row in rows if row["winner_by_hd95"] == "tie"),
        },
        "assd_winner_counts": {
            "3ddfa_v2": sum(1 for row in rows if row["winner_by_assd"] == "3ddfa_v2"),
            "mediapipe": sum(1 for row in rows if row["winner_by_assd"] == "mediapipe"),
            "tie": sum(1 for row in rows if row["winner_by_assd"] == "tie"),
        },
        "dice_winner_counts": {
            "3ddfa_v2": sum(1 for row in rows if row["winner_by_dice"] == "3ddfa_v2"),
            "mediapipe": sum(1 for row in rows if row["winner_by_dice"] == "mediapipe"),
            "tie": sum(1 for row in rows if row["winner_by_dice"] == "tie"),
        },
        "robust_claim_tests": {},
        "lambda_sweep": [],
    }

    worst_3ddfa_gap = out["gap_hd95_min_mm"]
    best_3ddfa_gap = out["gap_hd95_max_mm"]
    worst_normalized = out["gap_hd95_over_uncertainty_min"]
    best_normalized = out["gap_hd95_over_uncertainty_max"]
    out["robust_claim_tests"] = {
        "3ddfa_better_by_hd95": {
            "survives_sign_test": bool(worst_3ddfa_gap > 0),
            "survives_uncertainty_test": bool(worst_normalized > 1.0),
            "worst_case_gap_mm": worst_3ddfa_gap,
            "worst_case_gap_over_uncertainty": worst_normalized,
        },
        "mediapipe_better_by_hd95": {
            "survives_sign_test": bool(best_3ddfa_gap < 0),
            "survives_uncertainty_test": bool(best_normalized < -1.0),
            "best_case_against_3ddfa_gap_mm": best_3ddfa_gap,
            "best_case_against_3ddfa_gap_over_uncertainty": best_normalized,
        },
        "inconclusive_null": {
            "supported": bool(abs(best_normalized) < 1.0 or abs(worst_normalized) < 1.0),
            "reason": "Method-to-MRI gap is smaller than target/method uncertainty scale.",
        },
    }

    for lam in lambdas:
        def quality_regret(row: dict) -> float:
            return float(row["mri_only_score"]) - min_score

        best_for_3ddfa = max(
            rows,
            key=lambda row: float(row["gap_hd95_mp_minus_3ddfa_mm"]) - lam * quality_regret(row),
        )
        best_for_mp = max(
            rows,
            key=lambda row: -float(row["gap_hd95_mp_minus_3ddfa_mm"]) - lam * quality_regret(row),
        )
        out["lambda_sweep"].append(
            {
                "lambda_quality_penalty": lam,
                "best_candidate_supporting_3ddfa": best_for_3ddfa["candidate_id"],
                "support_3ddfa_objective": float(best_for_3ddfa["gap_hd95_mp_minus_3ddfa_mm"]) - lam * quality_regret(best_for_3ddfa),
                "support_3ddfa_raw_gap_mm": best_for_3ddfa["gap_hd95_mp_minus_3ddfa_mm"],
                "support_3ddfa_gap_over_uncertainty": best_for_3ddfa["gap_hd95_over_uncertainty"],
                "best_candidate_supporting_mediapipe": best_for_mp["candidate_id"],
                "support_mediapipe_objective": -float(best_for_mp["gap_hd95_mp_minus_3ddfa_mm"]) - lam * quality_regret(best_for_mp),
                "support_mediapipe_raw_gap_mm": best_for_mp["gap_hd95_mp_minus_3ddfa_mm"],
                "support_mediapipe_gap_over_uncertainty": best_for_mp["gap_hd95_over_uncertainty"],
            }
        )
    return out


def write_plot(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda row: float(row["mri_only_score"]))
    x = np.arange(len(rows_sorted))
    gaps = np.array([float(row["gap_hd95_mp_minus_3ddfa_mm"]) for row in rows_sorted])
    normalized = np.array([float(row["gap_hd95_over_uncertainty"]) for row in rows_sorted])
    scores = np.array([float(row["mri_only_score"]) for row in rows_sorted])
    labels = [row["candidate_id"].replace("_dec1p00", "") for row in rows_sorted]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)
    colors = ["#2563eb" if row["shape"] == "ellipse" else "#dc2626" for row in rows_sorted]
    axes[0].bar(x, gaps, color=colors, alpha=0.82)
    axes[0].axhline(0, color="black", linewidth=1.0)
    axes[0].set_ylabel("HD95 gap, mm\nMP - 3DDFA")
    axes[0].set_title("Adversarial target sensitivity: positive favors 3DDFA")
    axes[0].grid(axis="y", alpha=0.25)
    ax2 = axes[0].twinx()
    ax2.plot(x, scores, color="#111827", marker="o", markersize=3, linewidth=1.0, label="MRI-only score")
    ax2.set_ylabel("MRI-only score")

    axes[1].bar(x, normalized, color=colors, alpha=0.82)
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].axhline(1, color="#991b1b", linestyle="--", linewidth=1.0)
    axes[1].axhline(-1, color="#991b1b", linestyle="--", linewidth=1.0)
    axes[1].fill_between([-0.5, len(x) - 0.5], -1, 1, color="#fbbf24", alpha=0.12)
    axes[1].set_ylabel("Gap / uncertainty")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_lambdas(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-qc-csv", required=True, type=Path)
    parser.add_argument("--avatar-diagnostics-csv", required=True, type=Path)
    parser.add_argument("--method-disagreement-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lambdas", default="0,0.1,0.25,0.5,1,2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = read_csv(args.mri_qc_csv)
    avatar_rows = read_csv(args.avatar_diagnostics_csv)
    method_disagreement = json.loads(args.method_disagreement_json.read_text(encoding="utf-8"))
    rows = build_rows(qc_rows, avatar_rows, method_disagreement)
    rows = sorted(rows, key=lambda row: float(row["mri_only_score"]))

    write_csv_json(args.output_dir / "adversarial_gap_table.csv", rows)
    summary = adversarial_objectives(rows, parse_lambdas(args.lambdas))
    (args.output_dir / "adversarial_ranking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_plot(rows, args.output_dir / "adversarial_gap_plot.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
