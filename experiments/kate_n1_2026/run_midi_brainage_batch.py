#!/usr/bin/env python3
"""Run MIDIBrainAge one case at a time with resume-friendly CSV output."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path


FIELDNAMES_EXTRA = [
    "status",
    "error",
    "predicted_age_years",
    "brain_age_delta_years",
    "project_name",
    "elapsed_seconds",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_existing(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], set()
    rows = read_rows(path)
    done = {row["scan_id"] for row in rows if row.get("status") == "ok"}
    return rows, done


def append_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_project_name(prefix: str, scan_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scan_id)
    return f"{prefix}_{safe}"[:180]


def read_prediction(output_csv: Path) -> tuple[str, str]:
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one output row in {output_csv}, got {len(rows)}")
    pred = rows[0].get("Predicted_age (years)", "")
    chron = rows[0].get("Chronological age", "")
    return pred, chron


def run_case(row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    started = time.time()
    out = dict(row)
    out.update({key: "" for key in FIELDNAMES_EXTRA})
    out["status"] = "failed"
    project_name = safe_project_name(args.project_prefix, row["scan_id"])
    out["project_name"] = project_name
    case_csv = args.work_dir / f"{project_name}_input.csv"
    metrics_output_csv = args.work_dir / f"{project_name}_brain_age_output.csv"
    project_dir = args.work_dir / project_name
    prediction_output_csv = project_dir / "brain_age_output.csv"

    try:
        if not Path(row["path"]).exists():
            raise FileNotFoundError(row["path"])
        input_fieldnames = ["ID", "file_name"]
        case_row = {"ID": row["scan_id"], "file_name": row["path"]}
        if args.return_metrics:
            input_fieldnames.append("Age")
            case_row["Age"] = row.get("chronological_age_years", "")
        with case_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=input_fieldnames)
            writer.writeheader()
            writer.writerow(case_row)

        cmd = [
            args.python,
            str(args.work_dir / "run_inference.py"),
            "--csv_file",
            str(case_csv),
            "--project_name",
            project_name,
            "--sequence",
            "t1",
            "--ensemble",
            "--skull_strip",
        ]
        if args.return_metrics:
            cmd.append("--return_metrics")
        env = dict(os.environ)
        python_bin = Path(args.python).expanduser().parent
        linux_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env["PATH"] = f"{python_bin}{os.pathsep}{linux_path}"
        env.setdefault("VIRTUAL_ENV", str(python_bin.parent))
        result = subprocess.run(
            cmd,
            cwd=str(args.work_dir),
            text=True,
            capture_output=True,
            timeout=args.case_timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr + "\n" + result.stdout)[-4000:])
        output_csv = metrics_output_csv if args.return_metrics else prediction_output_csv
        try:
            pred, chron = read_prediction(output_csv)
        except Exception as exc:
            output_tail = (result.stderr + "\n" + result.stdout)[-4000:]
            raise RuntimeError(f"{exc}\nSTDOUT_STDERR_TAIL:\n{output_tail}") from exc
        out["predicted_age_years"] = pred
        chron_for_delta = chron or row.get("chronological_age_years", "")
        if pred != "" and chron_for_delta != "":
            out["brain_age_delta_years"] = f"{float(pred) - float(chron_for_delta):.6g}"
        out["status"] = "ok"
    except Exception as exc:
        out["error"] = repr(exc)
    finally:
        out["elapsed_seconds"] = f"{time.time() - started:.3f}"
        if not args.keep_work:
            case_csv.unlink(missing_ok=True)
            metrics_output_csv.unlink(missing_ok=True)
            shutil.rmtree(project_dir, ignore_errors=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/home/kate/midi_brainage_work"))
    parser.add_argument("--python", default="/home/kate/.venvs/midi_brainage_py311/bin/python")
    parser.add_argument("--project-prefix", default="midi_simon")
    parser.add_argument("--case-timeout-seconds", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--return-metrics", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    existing, done = load_existing(args.output_csv) if args.resume else ([], set())
    fieldnames = list(rows[0].keys()) + FIELDNAMES_EXTRA
    if existing and not args.output_csv.exists():
        raise RuntimeError("Internal resume state error")

    for index, row in enumerate(rows, start=1):
        if args.resume and row["scan_id"] in done:
            print(f"[{index}/{len(rows)}] skip {row['scan_id']}", flush=True)
            continue
        result = run_case(row, args)
        append_row(args.output_csv, fieldnames, result)
        print(
            f"[{index}/{len(rows)}] {row['scan_id']} {result['status']} "
            f"pred={result.get('predicted_age_years', '')} elapsed={result['elapsed_seconds']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
