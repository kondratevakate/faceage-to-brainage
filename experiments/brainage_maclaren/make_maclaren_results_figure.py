#!/usr/bin/env python3
"""Render the compact Maclaren NeuroFM repeatability result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "sub-01": "#176B87",
    "sub-02": "#D05A47",
    "sub-03": "#8A6D1D",
    "rotation": "#176B87",
    "resolution": "#D05A47",
    "scale": "#8A6D1D",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/brainage"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("docs/brainage/figures/maclaren_neurofm_robustness"),
    )
    args = parser.parse_args()

    predictions = read_csv(args.data_dir / "maclaren_neurofm_predictions.csv")
    perturbations = read_csv(
        args.data_dir / "maclaren_perturbation_predictions.csv"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    age_by_subject = {
        subject: float(
            next(
                row["chronological_age_years"]
                for row in predictions
                if row["participant_id"] == subject
            )
        )
        for subject in COLORS
        if subject.startswith("sub-")
    }
    for subject, chronological_age in age_by_subject.items():
        rows = sorted(
            (row for row in predictions if row["participant_id"] == subject),
            key=lambda row: int(row["run_index"]),
        )
        runs = np.asarray([int(row["run_index"]) for row in rows])
        ages = np.asarray([float(row["predicted_brain_age_years"]) for row in rows])
        axes[0].plot(
            runs,
            ages,
            marker="o",
            markersize=2.8,
            linewidth=1.1,
            color=COLORS[subject],
            label=f"{subject}: true {chronological_age:.0f}, mean {ages.mean():.1f}",
        )
        axes[0].axhline(
            chronological_age,
            color=COLORS[subject],
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            alpha=0.7,
        )
    axes[0].set_title("A  Short-interval repeated acquisitions")
    axes[0].set_xlabel("Acquisition run")
    axes[0].set_ylabel("NeuroFM-S predicted age (years)")
    axes[0].set_xlim(0, 41)
    axes[0].set_ylim(24, 61)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False, loc="center right", fontsize=8)

    families = ["rotation", "resolution", "scale"]
    for family_index, family in enumerate(families):
        rows = [
            row
            for row in perturbations
            if row["perturbation_family"] == family and row["status"] == "ok"
        ]
        values = np.asarray([abs(float(row["brain_age_delta_years"])) for row in rows])
        offsets = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else [0.0]
        axes[1].scatter(
            family_index + np.asarray(offsets),
            values,
            s=20,
            color=COLORS[family],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        if values.size:
            axes[1].plot(
                [family_index - 0.24, family_index + 0.24],
                [np.median(values)] * 2,
                color="#222222",
                linewidth=1.6,
                zorder=4,
            )
        failed = sum(
            row["perturbation_family"] == family and row["status"] == "failed"
            for row in perturbations
        )
        if failed:
            axes[1].text(
                family_index,
                5.55,
                f"{failed} failures",
                color=COLORS[family],
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    axes[1].axhline(
        2.0,
        color="#444444",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        label="Predeclared 2-year margin",
    )
    axes[1].set_title("B  Numerical perturbation sensitivity")
    axes[1].set_ylabel("Absolute age-output delta (years)")
    axes[1].set_xticks(range(len(families)), [name.title() for name in families])
    axes[1].set_ylim(0, 6.1)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)

    figure.suptitle(
        "Official NeuroFM-S on Maclaren ds000239: robustness, not age validation",
        fontsize=12,
        fontweight="bold",
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    figure.savefig(args.output_prefix.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    main()
