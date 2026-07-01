#!/usr/bin/env python3
"""Build SIMON derivative-level segmentation test-retest manifests.

This does not process raw MRI. It records which locally available FreeSurfer8
derivative label maps can be used for the first repeatability scaffold.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHENO = Path("data/SIMON_pheno.csv")
DEFAULT_FS8_ROOT = Path("D:/data/freesurfer8_simon")
DEFAULT_SESSION_OUT = Path("data/kate_n1_2026/simon_freesurfer8_segmentation_sessions.csv")
DEFAULT_PAIR_OUT = Path("data/kate_n1_2026/simon_freesurfer8_segmentation_pairs.csv")
DEFAULT_STATUS_OUT = Path("data/kate_n1_2026/simon_segmentation_test_retest_status.csv")


@dataclass(frozen=True)
class SessionRecord:
    dataset: str
    subject_id: str
    session_id: str
    session_number: int
    age: str
    acquisition_date: str
    institution_name: str
    manufacturer: str
    scanner_model: str
    fs8_subject_dir: Path
    dkt_aseg_path: Path
    aseg_path: Path
    orig_path: Path
    has_dkt_aseg: bool
    has_aparc_aseg: bool
    has_orig: bool


def read_pheno(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def session_id(session_number: int) -> str:
    return f"ses-{session_number:03d}"


def build_sessions(pheno_rows: list[dict[str, str]], fs8_root: Path) -> list[SessionRecord]:
    out: list[SessionRecord] = []
    for row in pheno_rows:
        number = int(row["Session"])
        sid = session_id(number)
        subject_dir = fs8_root / sid
        dkt_aseg = subject_dir / "mri" / "aparc.DKTatlas+aseg.mgz"
        aseg = subject_dir / "mri" / "aparc+aseg.mgz"
        orig = subject_dir / "mri" / "orig.mgz"
        out.append(
            SessionRecord(
                dataset="SIMON",
                subject_id="sub-032633",
                session_id=sid,
                session_number=number,
                age=row.get("Age", ""),
                acquisition_date=row.get("Acquisition_date", ""),
                institution_name=row.get("institution_name", ""),
                manufacturer=row.get("manufacturer", ""),
                scanner_model=row.get("man_model_name", ""),
                fs8_subject_dir=subject_dir,
                dkt_aseg_path=dkt_aseg,
                aseg_path=aseg,
                orig_path=orig,
                has_dkt_aseg=dkt_aseg.exists(),
                has_aparc_aseg=aseg.exists(),
                has_orig=orig.exists(),
            )
        )
    return out


def write_sessions(path: Path, sessions: list[SessionRecord]) -> None:
    rows = []
    for session in sessions:
        rows.append(
            {
                "dataset": session.dataset,
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "session_number": session.session_number,
                "age": session.age,
                "acquisition_date": session.acquisition_date,
                "institution_name": session.institution_name,
                "manufacturer": session.manufacturer,
                "scanner_model": session.scanner_model,
                "fs8_subject_dir": str(session.fs8_subject_dir),
                "dkt_aseg_path": str(session.dkt_aseg_path),
                "aseg_path": str(session.aseg_path),
                "orig_path": str(session.orig_path),
                "has_dkt_aseg": int(session.has_dkt_aseg),
                "has_aparc_aseg": int(session.has_aparc_aseg),
                "has_orig": int(session.has_orig),
                "evidence_role": "derivative_repeatability_secondary",
                "limitation": "FreeSurfer8 derivative exists locally; raw T1 source is not represented in this manifest.",
            }
        )
    write_csv(path, rows)


def write_pairs(path: Path, sessions: list[SessionRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_number = {session.session_number: session for session in sessions}
    for first_number in sorted(by_number)[:-1]:
        first = by_number[first_number]
        second = by_number.get(first_number + 1)
        if second is None:
            continue
        usable = first.has_dkt_aseg and second.has_dkt_aseg
        rows.append(
            {
                "dataset": "SIMON",
                "subject_id": "sub-032633",
                "pair_id": f"{first.session_id}_vs_{second.session_id}",
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
                "label1_path": str(first.dkt_aseg_path),
                "label2_path": str(second.dkt_aseg_path),
                "usable_for_derivative_spatial_repeatability": int(usable),
                "evidence_role": "derivative_repeatability_secondary",
                "limitation": "Consecutive-session FreeSurfer8 derivative pair; scanner/time/aging effects are mixed.",
            }
        )
    write_csv(path, rows)
    return rows


def write_status(path: Path, sessions: list[SessionRecord], pairs: list[dict[str, object]]) -> None:
    n_sessions = len(sessions)
    n_dkt = sum(session.has_dkt_aseg for session in sessions)
    n_usable_pairs = sum(int(row["usable_for_derivative_spatial_repeatability"]) for row in pairs)
    rows = [
        {
            "dataset": "SIMON",
            "branch": "FreeSurfer8 derivative segmentation repeatability",
            "status": "manifest_built_secondary_derivative_evidence",
            "n_pheno_sessions": n_sessions,
            "n_sessions_with_dkt_aseg": n_dkt,
            "n_consecutive_pairs": len(pairs),
            "n_usable_consecutive_pairs": n_usable_pairs,
            "session_manifest": str(DEFAULT_SESSION_OUT),
            "pair_manifest": str(DEFAULT_PAIR_OUT),
            "current_interpretation": "Can support derivative-level repeatability scaffolding; does not prove raw-input SOTA robustness.",
            "next_action": "Run or map method outputs for at least SynthSeg and one comparator on accessible raw or derivative inputs, then connect TTA uncertainty with pairwise repeatability error.",
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
    parser.add_argument("--fs8-root", type=Path, default=DEFAULT_FS8_ROOT)
    parser.add_argument("--session-out", type=Path, default=DEFAULT_SESSION_OUT)
    parser.add_argument("--pair-out", type=Path, default=DEFAULT_PAIR_OUT)
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    args = parser.parse_args()

    pheno_rows = read_pheno(args.pheno)
    sessions = build_sessions(pheno_rows, args.fs8_root)
    write_sessions(args.session_out, sessions)
    pairs = write_pairs(args.pair_out, sessions)
    write_status(args.status_out, sessions, pairs)

    print(f"Wrote {len(sessions)} sessions to {args.session_out}")
    print(f"Wrote {len(pairs)} consecutive pairs to {args.pair_out}")
    print(f"Wrote status to {args.status_out}")


if __name__ == "__main__":
    main()
