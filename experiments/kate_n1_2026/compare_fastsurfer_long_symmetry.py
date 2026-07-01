from __future__ import annotations

import csv
import datetime as dt
import math
import statistics
from pathlib import Path


WORKSPACE = Path(r"C:\Users\Lenovo\Documents\Codex\2026-06-11\prior-conversation-with-codex-conversation-role")
OUT_DIR = WORKSPACE / "outputs"

DATA_ROOT = Path(r"D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years")
SYM_ROOT = DATA_ROOT / "reprocessed_2026" / "symmetry"
CROSS_ROOT = SYM_ROOT / "fastsurfer"
LONG_ROOT = SYM_ROOT / "fastsurfer_long_v2"
LOG = SYM_ROOT / "logs" / "fastsurfer_long_symmetry_v2.log"

SUBJECTS = ("sym_rotpos", "sym_rotneg")
ASEG_FILE = "aseg+DKT.VINN.stats"
HYP_FILE = "hypothalamus.HypVINN.stats"
APARC_FILE_TEMPLATE = "{hemi}.aparc.DKTatlas.mapped.stats"
SURFACE_METRICS = ["ThickAvg", "GrayVol", "SurfArea"]

SUBCORTICAL = [
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Thalamus",
    "Right-Thalamus",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Pallidum",
    "Right-Pallidum",
]

GLOBAL_MEASURES = [
    "BrainSegVol",
    "BrainSegVolNotVent",
    "SupraTentorialVol",
    "SupraTentorialVolNotVent",
    "SubCortGrayVol",
    "lhCerebralWhiteMatterVol",
    "rhCerebralWhiteMatterVol",
    "CerebralWhiteMatterVol",
]


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_stats_table(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in read_lines(path):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue
        try:
            values[parts[4]] = float(parts[3])
        except ValueError:
            continue
    return values


def parse_measures(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in read_lines(path):
        if not line.startswith("# Measure "):
            continue
        fields = [field.strip() for field in line[len("# Measure ") :].split(",")]
        if len(fields) < 4:
            continue
        try:
            values[fields[1]] = float(fields[3])
        except ValueError:
            continue
    return values


def parse_aparc_stats(path: Path, hemi: str) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for line in read_lines(path):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            values[f"{hemi}-{parts[0]}"] = {
                "NumVert": float(parts[1]),
                "SurfArea": float(parts[2]),
                "GrayVol": float(parts[3]),
                "ThickAvg": float(parts[4]),
                "ThickStd": float(parts[5]),
                "MeanCurv": float(parts[6]),
                "GausCurv": float(parts[7]),
                "FoldInd": float(parts[8]),
                "CurvInd": float(parts[9]),
            }
        except ValueError:
            continue
    return values


def pair_cv(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    total = a + b
    if total == 0:
        return 0.0
    return 100.0 * abs(a - b) / total


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return None
    return statistics.median(clean)


def load_pair(root: Path, stats_name: str) -> dict[str, dict[str, float]]:
    return {
        subject: parse_stats_table(root / subject / "stats" / stats_name)
        for subject in SUBJECTS
    }


def load_measure_pair(root: Path) -> dict[str, dict[str, float]]:
    return {
        subject: parse_measures(root / subject / "stats" / ASEG_FILE)
        for subject in SUBJECTS
    }


def load_surface_pair(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    pair: dict[str, dict[str, dict[str, float]]] = {}
    for subject in SUBJECTS:
        merged: dict[str, dict[str, float]] = {}
        for hemi in ("lh", "rh"):
            merged.update(
                parse_aparc_stats(
                    root / subject / "stats" / APARC_FILE_TEMPLATE.format(hemi=hemi),
                    hemi,
                )
            )
        pair[subject] = merged
    return pair


def compare_structures(
    category: str,
    structures: list[str],
    cross: dict[str, dict[str, float]],
    long: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in structures:
        c_pos = cross[SUBJECTS[0]].get(name)
        c_neg = cross[SUBJECTS[1]].get(name)
        l_pos = long[SUBJECTS[0]].get(name)
        l_neg = long[SUBJECTS[1]].get(name)
        c_cv = pair_cv(c_pos, c_neg)
        l_cv = pair_cv(l_pos, l_neg)
        ratio = None
        if c_cv is not None and l_cv is not None and c_cv > 0:
            ratio = l_cv / c_cv
        rows.append(
            {
                "category": category,
                "name": name,
                "cross_pos": c_pos,
                "cross_neg": c_neg,
                "cross_cv": c_cv,
                "long_pos": l_pos,
                "long_neg": l_neg,
                "long_cv": l_cv,
                "delta_cv": None if c_cv is None or l_cv is None else l_cv - c_cv,
                "ratio": ratio,
                "improved": None if c_cv is None or l_cv is None else l_cv < c_cv,
            }
        )
    return rows


def compare_long_only_surface(
    metric: str,
    long: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, object]]:
    structures = sorted(set(long[SUBJECTS[0]]) & set(long[SUBJECTS[1]]))
    rows: list[dict[str, object]] = []
    for name in structures:
        l_pos = long[SUBJECTS[0]][name].get(metric)
        l_neg = long[SUBJECTS[1]][name].get(metric)
        l_cv = pair_cv(l_pos, l_neg)
        rows.append(
            {
                "category": f"long-surface-{metric}",
                "name": name,
                "cross_pos": None,
                "cross_neg": None,
                "cross_cv": None,
                "long_pos": l_pos,
                "long_neg": l_neg,
                "long_cv": l_cv,
                "delta_cv": None,
                "ratio": None,
                "improved": None,
            }
        )
    return rows


def compare_measures(
    cross: dict[str, dict[str, float]],
    long: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in GLOBAL_MEASURES:
        c_pos = cross[SUBJECTS[0]].get(name)
        c_neg = cross[SUBJECTS[1]].get(name)
        l_pos = long[SUBJECTS[0]].get(name)
        l_neg = long[SUBJECTS[1]].get(name)
        c_cv = pair_cv(c_pos, c_neg)
        l_cv = pair_cv(l_pos, l_neg)
        rows.append(
            {
                "category": "global-measure",
                "name": name,
                "cross_pos": c_pos,
                "cross_neg": c_neg,
                "cross_cv": c_cv,
                "long_pos": l_pos,
                "long_neg": l_neg,
                "long_cv": l_cv,
                "delta_cv": None if c_cv is None or l_cv is None else l_cv - c_cv,
                "ratio": None if c_cv in (None, 0) or l_cv is None else l_cv / c_cv,
                "improved": None if c_cv is None or l_cv is None else l_cv < c_cv,
            }
        )
    return rows


def summary_for(rows: list[dict[str, object]]) -> dict[str, object]:
    cross_cvs = [row["cross_cv"] for row in rows]
    long_cvs = [row["long_cv"] for row in rows]
    comparable = [row for row in rows if row["improved"] is not None]
    improved = [row for row in comparable if row["improved"]]
    return {
        "n": len(comparable),
        "median_cross_cv": median(cross_cvs),
        "median_long_cv": median(long_cvs),
        "improved": len(improved),
    }


def long_only_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    long_cvs = [row["long_cv"] for row in rows]
    clean = [value for value in long_cvs if value is not None and not math.isnan(value)]
    return {
        "n": len(clean),
        "median_long_cv": median(clean),
        "max_long_cv": max(clean) if clean else None,
    }


def status_table() -> list[dict[str, str]]:
    rows = []
    for root_name, root in (("cross", CROSS_ROOT), ("long", LONG_ROOT)):
        for subject in SUBJECTS:
            stats = root / subject / "stats" / ASEG_FILE
            hyp = root / subject / "stats" / HYP_FILE
            recon_done = root / subject / "scripts" / "recon-surf.done"
            recon_error = root / subject / "scripts" / "recon-surf.error"
            rows.append(
                {
                    "stream": root_name,
                    "subject": subject,
                    "aseg_dkt": "yes" if stats.exists() and stats.stat().st_size > 0 else "no",
                    "hypvinn": "yes" if hyp.exists() and hyp.stat().st_size > 0 else "no",
                    "surf_done": "yes" if recon_done.exists() else "no",
                    "surf_error": "yes" if recon_error.exists() else "no",
                }
            )
    base_done = LONG_ROOT / "sym_fast_base" / "scripts" / "recon-surf.done"
    base_err = LONG_ROOT / "sym_fast_base" / "scripts" / "recon-surf.error"
    rows.append(
        {
            "stream": "long-base",
            "subject": "sym_fast_base",
            "aseg_dkt": "yes"
            if (LONG_ROOT / "sym_fast_base" / "stats" / ASEG_FILE).exists()
            else "no",
            "hypvinn": "n/a",
            "surf_done": "yes" if base_done.exists() else "no",
            "surf_error": "yes" if base_err.exists() else "no",
        }
    )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "category",
        "name",
        "cross_pos",
        "cross_neg",
        "cross_cv",
        "long_pos",
        "long_neg",
        "long_cv",
        "delta_cv",
        "ratio",
        "improved",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def row_to_md(row: dict[str, object]) -> list[str]:
    ratio = row["ratio"]
    return [
        str(row["name"]),
        fmt(row["cross_pos"], 1),
        fmt(row["cross_neg"], 1),
        fmt(row["long_pos"], 1),
        fmt(row["long_neg"], 1),
        fmt(row["cross_cv"]),
        fmt(row["long_cv"]),
        fmt(row["delta_cv"]),
        "" if ratio is None else f"{float(ratio):.2f}x",
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cross_aseg = load_pair(CROSS_ROOT, ASEG_FILE)
    long_aseg = load_pair(LONG_ROOT, ASEG_FILE)
    cross_hyp = load_pair(CROSS_ROOT, HYP_FILE)
    long_hyp = load_pair(LONG_ROOT, HYP_FILE)
    cross_measures = load_measure_pair(CROSS_ROOT)
    long_measures = load_measure_pair(LONG_ROOT)
    long_surface = load_surface_pair(LONG_ROOT)

    cortical = sorted(
        set(cross_aseg[SUBJECTS[0]])
        & set(cross_aseg[SUBJECTS[1]])
        & set(long_aseg[SUBJECTS[0]])
        & set(long_aseg[SUBJECTS[1]])
    )
    cortical = [name for name in cortical if name.startswith("ctx-")]

    hyp_structures = sorted(
        set(cross_hyp[SUBJECTS[0]])
        & set(cross_hyp[SUBJECTS[1]])
        & set(long_hyp[SUBJECTS[0]])
        & set(long_hyp[SUBJECTS[1]])
    )

    sub_rows = compare_structures("subcortical-volume", SUBCORTICAL, cross_aseg, long_aseg)
    cortical_rows = compare_structures("cortical-dkt-volume", cortical, cross_aseg, long_aseg)
    hyp_rows = compare_structures("hypvinn-volume", hyp_structures, cross_hyp, long_hyp)
    measure_rows = compare_measures(cross_measures, long_measures)
    surface_rows_by_metric = {
        metric: compare_long_only_surface(metric, long_surface)
        for metric in SURFACE_METRICS
    }
    surface_rows = [row for rows in surface_rows_by_metric.values() for row in rows]
    all_rows = measure_rows + sub_rows + cortical_rows + hyp_rows + surface_rows

    csv_path = OUT_DIR / "fastsurfer_long_symmetry_volume_cv.csv"
    report_path = OUT_DIR / "fastsurfer_long_symmetry_report.md"
    write_csv(all_rows, csv_path)

    sub_summary = summary_for(sub_rows)
    cortical_summary = summary_for(cortical_rows)
    hyp_summary = summary_for(hyp_rows)
    measure_summary = summary_for(measure_rows)
    surface_summaries = {
        metric: long_only_summary(rows)
        for metric, rows in surface_rows_by_metric.items()
    }

    status_rows = [
        [
            row["stream"],
            row["subject"],
            row["aseg_dkt"],
            row["hypvinn"],
            row["surf_done"],
            row["surf_error"],
        ]
        for row in status_table()
    ]

    best = sorted(
        [row for row in cortical_rows + sub_rows if row["delta_cv"] is not None],
        key=lambda row: float(row["delta_cv"]),
    )[:8]
    worst = sorted(
        [row for row in cortical_rows + sub_rows if row["delta_cv"] is not None],
        key=lambda row: float(row["delta_cv"]),
        reverse=True,
    )[:8]

    log_note = ""
    if LOG.exists():
        text = "\n".join(read_lines(LOG)[-200:])
        if (
            "Full longitudinal pipeline finished" in text
            or "Full longitudinal processing" in text
            or "ALL DONE" in text
        ):
            log_note = "The wrapper log appears complete."
        elif "ERROR" in text:
            log_note = "The recent wrapper log contains `ERROR`; inspect the log before treating surface outputs as final."
        else:
            log_note = "The wrapper log does not yet show final `ALL DONE`; volume results can be read, but surface completion may still be pending."

    report = f"""# FastSurfer Long Symmetry Consistency Report

Date: {dt.date.today().isoformat()}

Dataset: `{DATA_ROOT}`

Cross-sectional FastSurfer: `{CROSS_ROOT}`

FastSurfer Long v2: `{LONG_ROOT}`

## Status

{md_table(["stream", "subject", "aseg+DKT", "HypVINN", "surf done", "surf error"], status_rows)}

{log_note}

## Method

This compares the same 2018 T1 after opposite synthetic rotations (`sym_rotpos` vs `sym_rotneg`). Pairwise CV is:

`100 * abs(rotpos - rotneg) / (rotpos + rotneg)`

Lower CV means better repeatability under this rotation perturbation. It is an internal consistency / method-floor check, not proof of biological accuracy.

## Summary

| Metric family | n | cross median CV% | long median CV% | improved |
|---|---:|---:|---:|---:|
| Global aseg measures | {measure_summary["n"]} | {fmt(measure_summary["median_cross_cv"])} | {fmt(measure_summary["median_long_cv"])} | {measure_summary["improved"]}/{measure_summary["n"]} |
| Subcortical volumes | {sub_summary["n"]} | {fmt(sub_summary["median_cross_cv"])} | {fmt(sub_summary["median_long_cv"])} | {sub_summary["improved"]}/{sub_summary["n"]} |
| Cortical DKT parcel volumes | {cortical_summary["n"]} | {fmt(cortical_summary["median_cross_cv"])} | {fmt(cortical_summary["median_long_cv"])} | {cortical_summary["improved"]}/{cortical_summary["n"]} |
| HypVINN volumes | {hyp_summary["n"]} | {fmt(hyp_summary["median_cross_cv"])} | {fmt(hyp_summary["median_long_cv"])} | {hyp_summary["improved"]}/{hyp_summary["n"]} |

Long-only surface symmetry, because the available cross-sectional FastSurfer folders are segmentation-only:

| Surface metric | n | long median CV% | long max CV% |
|---|---:|---:|---:|
| ThickAvg | {surface_summaries["ThickAvg"]["n"]} | {fmt(surface_summaries["ThickAvg"]["median_long_cv"])} | {fmt(surface_summaries["ThickAvg"]["max_long_cv"])} |
| GrayVol | {surface_summaries["GrayVol"]["n"]} | {fmt(surface_summaries["GrayVol"]["median_long_cv"])} | {fmt(surface_summaries["GrayVol"]["max_long_cv"])} |
| SurfArea | {surface_summaries["SurfArea"]["n"]} | {fmt(surface_summaries["SurfArea"]["median_long_cv"])} | {fmt(surface_summaries["SurfArea"]["max_long_cv"])} |

## Subcortical Volumes

{md_table(["Region", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"], [row_to_md(row) for row in sub_rows])}

## Global Measures

{md_table(["Measure", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"], [row_to_md(row) for row in measure_rows])}

## Largest CV Improvements

{md_table(["Region", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"], [row_to_md(row) for row in best])}

## Largest CV Worsenings

{md_table(["Region", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"], [row_to_md(row) for row in worst])}

## Files

- CSV table: `{csv_path}`
- Main FastSurfer Long log: `{LOG}`
"""

    if hyp_rows:
        report += "\n## HypVINN Volumes\n\n"
        report += md_table(
            ["Region", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"],
            [row_to_md(row) for row in hyp_rows],
        )
        report += "\n"

    thick_rows = surface_rows_by_metric["ThickAvg"]
    if thick_rows:
        highest_thick = sorted(
            thick_rows,
            key=lambda row: -1.0 if row["long_cv"] is None else float(row["long_cv"]),
            reverse=True,
        )[:12]
        report += "\n## Long-Only Surface Thickness\n\n"
        report += md_table(
            ["Region", "cross pos", "cross neg", "long pos", "long neg", "cross CV%", "long CV%", "delta", "ratio"],
            [row_to_md(row) for row in highest_thick],
        )
        report += "\n"

    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    print(csv_path)
    print(
        "subcortical",
        fmt(sub_summary["median_cross_cv"]),
        fmt(sub_summary["median_long_cv"]),
        f'{sub_summary["improved"]}/{sub_summary["n"]}',
    )
    print(
        "cortical-dkt-volume",
        fmt(cortical_summary["median_cross_cv"]),
        fmt(cortical_summary["median_long_cv"]),
        f'{cortical_summary["improved"]}/{cortical_summary["n"]}',
    )


if __name__ == "__main__":
    main()
