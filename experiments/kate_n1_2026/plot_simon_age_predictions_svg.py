#!/usr/bin/env python3
"""Plot SIMON chronological age vs tested brain-age model predictions.

The figure is intentionally descriptive. It compares local application branches
and does not turn any model output into a validation or clinical claim.
"""

from __future__ import annotations

import csv
import hashlib
import html
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "kate_n1_2026"
FIG_DIR = REPO_ROOT / "docs" / "kate_n1_2026" / "figures"

LONG_CSV = DATA_DIR / "simon_age_predictions_by_model_long.csv"
SVG_OUT = FIG_DIR / "simon_age_predictions_by_model.svg"


BRANCHES = [
    {
        "label": "BrainIAC app-style years (/12)",
        "source": DATA_DIR / "brainiac_brainage_predictions_simon_fastsurfer_orig.csv",
        "pred_col": "predicted_age_years_if_months",
        "note": "BrainIAC Space display-style conversion; unit ambiguity branch.",
        "color": "#7c3aed",
    },
    {
        "label": "BrainIAC raw output as years",
        "source": DATA_DIR / "brainiac_brainage_predictions_simon_fastsurfer_orig.csv",
        "pred_col": "predicted_age_years_if_raw_years",
        "note": "Diagnostic raw-output-as-years view; conflicts with Space display convention.",
        "color": "#a855f7",
    },
    {
        "label": "MIDIBrainAge stratified-12",
        "source": DATA_DIR / "midi_brainage_simon_stratified12_predictions.csv",
        "pred_col": "predicted_age_years",
        "note": "12-case stratified sanity subset.",
        "color": "#0f766e",
    },
    {
        "label": "MIDIBrainAge all-orig",
        "source": DATA_DIR / "midi_brainage_simon_all_orig_predictions.csv",
        "pred_col": "predicted_age_years",
        "note": "All visible FastSurfer orig.mgz derivative branch.",
        "color": "#14b8a6",
    },
    {
        "label": "NeuroFM FastSurfer mask",
        "source": DATA_DIR / "neurofm_simon_fastsurfer_mask_predictions.csv",
        "pred_col": "predicted_brain_age_years",
        "note": "Mask-derived skull-stripped branch; application/QC only.",
        "color": "#ea580c",
    },
    {
        "label": "NeuroFM raw-orig all-94",
        "source": DATA_DIR / "neurofm_simon_raw_orig_predictions.csv",
        "pred_col": "predicted_brain_age_years",
        "note": "Non-skull-stripped all-orig stress branch; application/QC only.",
        "color": "#f97316",
    },
]


def as_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_jitter(text: str, width: float = 0.16) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:2], "big") / 65535
    return (value - 0.5) * 2 * width


def collect_points() -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    for branch in BRANCHES:
        for row in read_rows(branch["source"]):
            if row.get("status", "ok") not in ("", "ok"):
                continue
            chron = as_float(row.get("chronological_age_years", ""))
            pred = as_float(row.get(branch["pred_col"], ""))
            if chron is None or pred is None:
                continue
            scan_id = row.get("scan_id", "") or row.get("session", "") or row.get("path", "")
            points.append(
                {
                    "model_branch": branch["label"],
                    "scan_id": scan_id,
                    "chronological_age_years": f"{chron:.6g}",
                    "predicted_age_years": f"{pred:.6g}",
                    "prediction_minus_chronological_years": f"{pred - chron:.6g}",
                    "source_file": str(branch["source"].relative_to(REPO_ROOT)),
                    "interpretation_note": branch["note"],
                }
            )
    return points


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    xs = [float(r["chronological_age_years"]) for r in rows]
    ys = [float(r["predicted_age_years"]) for r in rows]
    diffs = [y - x for x, y in zip(xs, ys)]
    return {
        "n": len(rows),
        "mae": sum(abs(d) for d in diffs) / len(diffs),
        "bias": sum(diffs) / len(diffs),
        "rmse": math.sqrt(sum(d * d for d in diffs) / len(diffs)),
        "r": pearson(xs, ys),
    }


def write_long_csv(points: list[dict[str, str]]) -> None:
    LONG_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_branch",
        "scan_id",
        "chronological_age_years",
        "predicted_age_years",
        "prediction_minus_chronological_years",
        "source_file",
        "interpretation_note",
    ]
    with LONG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def svg_text(x: float, y: float, text: str, size: int = 12, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#111827">{html.escape(text)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'
    )


def write_svg(points: list[dict[str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_branch = {branch["label"]: [] for branch in BRANCHES}
    for row in points:
        by_branch[row["model_branch"]].append(row)

    width, height = 1420, 920
    panel_w, panel_h = 410, 300
    left, top = 78, 92
    gap_x, gap_y = 42, 78
    plot_pad_left, plot_pad_right, plot_pad_top, plot_pad_bottom = 54, 18, 34, 46
    x_min, x_max = 28.0, 48.0
    y_min, y_max = 0.0, 95.0
    x_ticks = [30, 35, 40, 45]
    y_ticks = [0, 20, 40, 60, 80]

    def sx(value: float, col: int) -> float:
        x0 = left + col * (panel_w + gap_x) + plot_pad_left
        usable = panel_w - plot_pad_left - plot_pad_right
        return x0 + (value - x_min) / (x_max - x_min) * usable

    def sy(value: float, row: int) -> float:
        y0 = top + row * (panel_h + gap_y) + plot_pad_top
        usable = panel_h - plot_pad_top - plot_pad_bottom
        return y0 + (y_max - value) / (y_max - y_min) * usable

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(48, 42, "SIMON: chronological age vs model-predicted brain age", 24, "700"),
        svg_text(
            48,
            66,
            "Each panel is an application branch; y=x is the ideal line. These are QC/domain-risk outputs, not validation claims.",
            13,
        ),
    ]

    for idx, branch in enumerate(BRANCHES):
        row_i, col_i = divmod(idx, 3)
        px = left + col_i * (panel_w + gap_x)
        py = top + row_i * (panel_h + gap_y)
        plot_x0 = px + plot_pad_left
        plot_y0 = py + plot_pad_top
        plot_x1 = px + panel_w - plot_pad_right
        plot_y1 = py + panel_h - plot_pad_bottom
        branch_rows = by_branch[branch["label"]]
        m = metrics(branch_rows)

        out.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="4" fill="white" stroke="#cbd5e1"/>')
        out.append(svg_text(px + 16, py + 23, branch["label"], 14, "700"))
        r_text = "" if m["r"] is None else f", r={m['r']:.2f}"
        out.append(svg_text(px + 16, py + 43, f"n={m['n']}, MAE={m['mae']:.1f}, bias={m['bias']:+.1f}{r_text}", 12))

        for tick in y_ticks:
            y = sy(tick, row_i)
            out.append(line(plot_x0, y, plot_x1, y, "#e5e7eb", 1))
            out.append(svg_text(plot_x0 - 8, y + 4, str(tick), 10, anchor="end"))
        for tick in x_ticks:
            x = sx(tick, col_i)
            out.append(line(x, plot_y0, x, plot_y1, "#eef2f7", 1))
            out.append(svg_text(x, plot_y1 + 18, str(tick), 10, anchor="middle"))

        out.append(line(plot_x0, plot_y1, plot_x1, plot_y1, "#334155", 1.2))
        out.append(line(plot_x0, plot_y0, plot_x0, plot_y1, "#334155", 1.2))
        out.append(line(sx(x_min, col_i), sy(x_min, row_i), sx(x_max, col_i), sy(x_max, row_i), "#64748b", 1.5, "5 5"))

        for point in branch_rows:
            chron = float(point["chronological_age_years"])
            pred = float(point["predicted_age_years"])
            x = sx(chron + stable_jitter(point["scan_id"] + branch["label"]), col_i)
            y = sy(pred, row_i)
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{branch["color"]}" '
                f'fill-opacity="0.68" stroke="white" stroke-width="0.45"><title>'
                f'{html.escape(point["scan_id"])}: true {chron:.2f}, predicted {pred:.2f}'
                "</title></circle>"
            )

        out.append(svg_text((plot_x0 + plot_x1) / 2, py + panel_h - 10, "Chronological age, years", 11, anchor="middle"))
        if col_i == 0:
            out.append(
                f'<text x="{px + 14}" y="{(plot_y0 + plot_y1) / 2:.1f}" font-size="11" '
                'text-anchor="middle" transform="rotate(-90 '
                f'{px + 14} {(plot_y0 + plot_y1) / 2:.1f})" fill="#111827">'
                "Predicted age, years</text>"
            )

    out.append(svg_text(48, height - 28, f"Long-format data: {LONG_CSV.relative_to(REPO_ROOT)}", 12))
    out.append("</svg>")
    SVG_OUT.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    points = collect_points()
    write_long_csv(points)
    write_svg(points)
    print(f"Wrote {LONG_CSV}")
    print(f"Wrote {SVG_OUT}")


if __name__ == "__main__":
    main()
