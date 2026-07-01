#!/usr/bin/env python3
"""Build SIMON FastSurfer `_orig.mgz` segmentation input manifests.

These `_orig.mgz` files are useful standardized inputs for external segmenter
smoke tests and repeatability analysis. They are FreeSurfer/FastSurfer internal
source images: not segmentation outputs, but potentially converted/conformed
copies of the original input rather than untouched scanner DICOM/NIfTI.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PHENO = Path("data/SIMON_pheno.csv")
DEFAULT_ORIG_ROOT = Path("D:/data/fastserfer_simon")
DEFAULT_INPUT_OUT = Path("data/kate_n1_2026/simon_fastsurfer_orig_segmentation_inputs.csv")
DEFAULT_PAIR_OUT = Path("data/kate_n1_2026/simon_fastsurfer_orig_segmentation_run1_pairs.csv")
DEFAULT_STATUS_OUT = Path("data/kate_n1_2026/simon_orig_input_benchmark_status.csv")

ORIG_RE = re.compile(
    r"^ses-(?P<session>\d{3})"
    r"(?:_acq-(?P<acquisition>[^_]+))?"
    r"(?:_run-(?P<run>\d+))?"
    r"(?:_(?P<modality>T1w))?"
    r"_orig\.mgz$"
)


@dataclass(frozen=True)
class OrigInput:
    dataset: str
    subject_id: str
    session_id: str
    session_number: int
    run: str
    acquisition: str
    modality: str
    scan_id: str
    age: str
    acquisition_date: str
    institution_name: str
    manufacturer: str
    scanner_model: str
    input_path: Path
    file_size_bytes: int


def read_pheno(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["Session"]): row for row in rows}


def to_wsl_path(path: Path) -> str:
    text = path.as_posix()
    if len(text) >= 3 and text[1:3] == ":/":
        drive = text[0].lower()
        return f"/mnt/{drive}{text[2:]}"
    return text


def build_inputs(pheno_by_session: dict[int, dict[str, str]], orig_root: Path) -> list[OrigInput]:
    inputs: list[OrigInput] = []
    for path in sorted(orig_root.glob("*_orig.mgz")):
        match = ORIG_RE.match(path.name)
        if match is None:
            continue
        session_number = int(match.group("session"))
        run_number = match.group("run")
        acquisition = match.group("acquisition") or ""
        modality = match.group("modality") or "unknown"
        pheno = pheno_by_session.get(session_number, {})
        session_id = f"ses-{session_number:03d}"
        run = f"run-{run_number}" if run_number else "run-unspecified"
        scan_parts = [session_id]
        if acquisition:
            scan_parts.append(f"acq-{acquisition}")
        if run_number:
            scan_parts.append(run)
        if modality != "unknown":
            scan_parts.append(modality)
        inputs.append(
            OrigInput(
                dataset="SIMON",
                subject_id="sub-032633",
                session_id=session_id,
                session_number=session_number,
                run=run,
                acquisition=acquisition,
                modality=modality,
                scan_id="_".join(scan_parts),
                age=pheno.get("Age", ""),
                acquisition_date=pheno.get("Acquisition_date", ""),
                institution_name=pheno.get("institution_name", ""),
                manufacturer=pheno.get("manufacturer", ""),
                scanner_model=pheno.get("man_model_name", ""),
                input_path=path,
                file_size_bytes=path.stat().st_size,
            )
        )
    return inputs


def write_inputs(path: Path, inputs: list[OrigInput]) -> None:
    rows = []
    for item in inputs:
        rows.append(
            {
                "dataset": item.dataset,
                "subject_id": item.subject_id,
                "session_id": item.session_id,
                "session_number": item.session_number,
                "run": item.run,
                "acquisition": item.acquisition,
                "modality": item.modality,
                "scan_id": item.scan_id,
                "age": item.age,
                "acquisition_date": item.acquisition_date,
                "institution_name": item.institution_name,
                "manufacturer": item.manufacturer,
                "scanner_model": item.scanner_model,
                "input_path_windows": str(item.input_path),
                "input_path_wsl": to_wsl_path(item.input_path),
                "file_size_bytes": item.file_size_bytes,
                "preprocessing_level": "fastsurfer_orig_mgz_internal_source_possibly_conformed",
                "evidence_role": "standardized_internal_source_input_benchmark",
                "limitation": "FreeSurfer/FastSurfer internal source image; not a segmentation derivative, but not guaranteed untouched scanner-native DICOM/NIfTI.",
            }
        )
    write_csv(path, rows)


def write_run1_pairs(path: Path, inputs: list[OrigInput]) -> list[dict[str, object]]:
    run1_candidates = [
        item
        for item in inputs
        if item.run == "run-1" and item.modality == "T1w" and item.acquisition == ""
    ]
    run1 = {item.session_number: item for item in run1_candidates}
    rows: list[dict[str, object]] = []
    for first_number in sorted(run1)[:-1]:
        first = run1[first_number]
        second = run1.get(first_number + 1)
        if second is None:
            continue
        rows.append(
            {
                "dataset": "SIMON",
                "subject_id": "sub-032633",
                "pair_id": f"{first.session_id}_run-1_vs_{second.session_id}_run-1",
                "session1": first.session_id,
                "session2": second.session_id,
                "age1": first.age,
                "age2": second.age,
                "institution1": first.institution_name,
                "institution2": second.institution_name,
                "manufacturer1": first.manufacturer,
                "manufacturer2": second.manufacturer,
                "scanner_model1": first.scanner_model,
                "scanner_model2": second.scanner_model,
                "input1_path_windows": str(first.input_path),
                "input2_path_windows": str(second.input_path),
                "input1_path_wsl": to_wsl_path(first.input_path),
                "input2_path_wsl": to_wsl_path(second.input_path),
                "usable_for_standardized_input_repeatability": 1,
                "limitation": "Consecutive FastSurfer-orig internal-source pair; scanner/time/aging/conform effects are mixed.",
            }
        )
    write_csv(path, rows)
    return rows


def write_status(path: Path, inputs: list[OrigInput], pairs: list[dict[str, object]]) -> None:
    n_sessions = len({item.session_number for item in inputs})
    n_run1 = sum(item.run == "run-1" for item in inputs)
    n_t1w = sum(item.modality == "T1w" for item in inputs)
    n_unknown = sum(item.modality == "unknown" for item in inputs)
    rows = [
        {
            "dataset": "SIMON",
            "branch": "FastSurfer orig internal-source segmentation input benchmark",
            "status": "manifest_built_internal_source_input",
            "n_orig_inputs": len(inputs),
            "n_unique_sessions": n_sessions,
            "n_t1w_inputs": n_t1w,
            "n_unknown_modality_inputs": n_unknown,
            "n_run1_inputs": n_run1,
            "n_run1_consecutive_pairs": len(pairs),
            "input_manifest": str(DEFAULT_INPUT_OUT),
            "pair_manifest": str(DEFAULT_PAIR_OUT),
            "current_interpretation": "Can be used as the available standardized source-image dataset for segmenter smoke tests and repeatability; distinguish it from untouched scanner-native DICOM/NIfTI.",
            "next_action": "Run SynthSeg/BrainChop/SIAM-compatible wrappers on a small run-1 subset first, then scale if runtime and QC pass.",
        }
    ]
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheno", type=Path, default=DEFAULT_PHENO)
    parser.add_argument("--orig-root", type=Path, default=DEFAULT_ORIG_ROOT)
    parser.add_argument("--input-out", type=Path, default=DEFAULT_INPUT_OUT)
    parser.add_argument("--pair-out", type=Path, default=DEFAULT_PAIR_OUT)
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    args = parser.parse_args()

    pheno_by_session = read_pheno(args.pheno)
    inputs = build_inputs(pheno_by_session, args.orig_root)
    write_inputs(args.input_out, inputs)
    pairs = write_run1_pairs(args.pair_out, inputs)
    write_status(args.status_out, inputs, pairs)

    print(f"Wrote {len(inputs)} inputs to {args.input_out}")
    print(f"Wrote {len(pairs)} run-1 consecutive pairs to {args.pair_out}")
    print(f"Wrote status to {args.status_out}")


if __name__ == "__main__":
    main()
