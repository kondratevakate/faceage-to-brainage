#!/usr/bin/env python3
"""Run BrainChop CLI on a manifest with per-scan timeouts.

This wrapper is intentionally conservative: it writes only small logs and a CSV
summary to the output root. NIfTI outputs are written outside git under the
local data root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/mnt/d/YandexDisk/kondratevakate/01_insidekatesbrain/01_my_brain_years")
DEFAULT_MANIFEST = Path(__file__).with_name("brainchop_inputs.csv")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reprocessed_2026" / "brainchop" / "brainchop_0.2.5"


@dataclass
class RunRecord:
    scan_id: str
    session: str
    model: str
    input_path: str
    output_path: str
    status: str
    elapsed_sec: float
    returncode: int | str
    log_path: str
    note: str


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one BrainChop model must be provided.")
    return models


def parse_optional_csv(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_id", "session", "relative_path", "brainchop_candidate"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return rows


def write_summary(path: Path, rows: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "all"


def run_one(
    *,
    brainchop_bin: Path,
    data_root: Path,
    output_root: Path,
    row: dict[str, str],
    model: str,
    timeout_sec: int,
    force: bool,
    dry_run: bool,
) -> RunRecord:
    scan_id = row["scan_id"]
    input_path = data_root / row["relative_path"]
    model_dir = output_root / model
    output_path = model_dir / f"{scan_id}_{model}.nii.gz"
    log_path = model_dir / "logs" / f"{scan_id}_{model}.log"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return RunRecord(scan_id, row["session"], model, str(input_path), str(output_path), "missing_input", 0.0, "NA", str(log_path), "Input NIfTI not found.")

    if output_path.exists() and not force:
        return RunRecord(scan_id, row["session"], model, str(input_path), str(output_path), "skipped_existing", 0.0, "NA", str(log_path), "Output already exists.")

    cmd = [
        str(brainchop_bin),
        str(input_path),
        "-m",
        model,
        "--inverse-conform",
        "--no-optimize",
        "-o",
        str(output_path),
    ]
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return RunRecord(scan_id, row["session"], model, str(input_path), str(output_path), "dry_run", 0.0, "NA", str(log_path), "Command not executed.")

    env = os.environ.copy()
    env["PATH"] = str(brainchop_bin.parent) + os.pathsep + env.get("PATH", "")
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(output + f"\nTIMEOUT after {timeout_sec} seconds\n", encoding="utf-8")
        return RunRecord(scan_id, row["session"], model, str(input_path), str(output_path), "timeout", round(elapsed, 3), "timeout", str(log_path), "BrainChop exceeded per-scan timeout.")

    elapsed = time.monotonic() - start
    log_path.write_text(completed.stdout, encoding="utf-8")
    status = "done" if completed.returncode == 0 and output_path.exists() else "failed"
    note = "Completed." if status == "done" else "Nonzero exit code or output file missing."
    return RunRecord(scan_id, row["session"], model, str(input_path), str(output_path), status, round(elapsed, 3), completed.returncode, str(log_path), note)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BrainChop CLI on Kate n=1 manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--brainchop-bin", type=Path, default=Path.home() / ".venvs" / "brainchop" / "bin" / "brainchop")
    parser.add_argument("--models", default="subcortical", help="Comma-separated BrainChop model names.")
    parser.add_argument("--scan-ids", default="", help="Optional comma-separated scan_id filter.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on candidate scans per model; 0 means all.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [row for row in read_manifest(args.manifest) if row.get("brainchop_candidate", "").strip() in {"1", "true", "True", "yes"}]
    scan_ids = parse_optional_csv(args.scan_ids)
    if scan_ids:
        rows = [row for row in rows if row["scan_id"] in scan_ids]
    if args.limit > 0:
        rows = rows[: args.limit]
    models = parse_models(args.models)

    summary: list[RunRecord] = []
    for model in models:
        for row in rows:
            record = run_one(
                brainchop_bin=args.brainchop_bin,
                data_root=args.data_root,
                output_root=args.output_root,
                row=row,
                model=model,
                timeout_sec=args.timeout_seconds,
                force=args.force,
                dry_run=args.dry_run,
            )
            summary.append(record)
            print(json.dumps(record.__dict__, ensure_ascii=True), flush=True)

    model_tag = safe_name("_".join(models))
    scan_tag = safe_name("_".join(sorted(scan_ids)) if scan_ids else "all_scans")
    write_summary(args.output_root / "brainchop_run_summary.csv", summary)
    write_summary(args.output_root / f"brainchop_run_summary_{model_tag}_{scan_tag}.csv", summary)
    metadata = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "models": models,
        "scan_ids": sorted(scan_ids),
        "timeout_seconds": args.timeout_seconds,
        "dry_run": args.dry_run,
    }
    (args.output_root / "brainchop_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_root / f"brainchop_run_metadata_{model_tag}_{scan_tag}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    failed = [row for row in summary if row.status in {"failed", "timeout", "missing_input"}]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
