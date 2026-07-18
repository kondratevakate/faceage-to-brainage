#!/usr/bin/env python3
"""Download and hash the official FCP-INDI BNU1 T1 test-retest subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


S3_ORIGIN = "https://fcp-indi.s3.amazonaws.com"
S3_PREFIX = "data/Projects/CORR/RawDataBIDS/BNU_1/"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/d/data/faceage-to-brainage/sourcedata/bnu1_corr/s3_2026-07-18/BNU_1"
)
EXPECTED_T1_FILES = 107
EXPECTED_SESSION_TABLES = 57
XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def request(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "faceage-to-brainage/1.0"}),
        timeout=120,
    )


def list_objects() -> list[dict[str, object]]:
    parameters = {"list-type": "2", "prefix": S3_PREFIX, "max-keys": "1000"}
    rows: list[dict[str, object]] = []
    while True:
        url = f"{S3_ORIGIN}/?{urllib.parse.urlencode(parameters)}"
        with request(url) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", XML_NAMESPACE):
            rows.append(
                {
                    "key": item.findtext("s3:Key", namespaces=XML_NAMESPACE),
                    "size_bytes": int(
                        item.findtext("s3:Size", namespaces=XML_NAMESPACE) or "0"
                    ),
                    "etag": (
                        item.findtext("s3:ETag", namespaces=XML_NAMESPACE) or ""
                    ).strip('"'),
                    "last_modified": item.findtext(
                        "s3:LastModified", namespaces=XML_NAMESPACE
                    ),
                }
            )
        if root.findtext("s3:IsTruncated", namespaces=XML_NAMESPACE) != "true":
            break
        continuation = root.findtext(
            "s3:NextContinuationToken", namespaces=XML_NAMESPACE
        )
        if not continuation:
            raise RuntimeError("S3 listing was truncated without a continuation token")
        parameters["continuation-token"] = continuation
    return rows


def select_objects(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in objects
        if str(row["key"]).endswith("_T1w.nii.gz")
        or str(row["key"]).endswith("_sessions.tsv")
        or str(row["key"]) in {f"{S3_PREFIX}participants.tsv", f"{S3_PREFIX}T1w.json"}
    ]
    t1_count = sum(str(row["key"]).endswith("_T1w.nii.gz") for row in selected)
    session_count = sum(str(row["key"]).endswith("_sessions.tsv") for row in selected)
    if t1_count != EXPECTED_T1_FILES or session_count != EXPECTED_SESSION_TABLES:
        raise ValueError(
            "Official S3 inventory changed: expected "
            f"{EXPECTED_T1_FILES} T1 and {EXPECTED_SESSION_TABLES} session tables, "
            f"found {t1_count} and {session_count}"
        )
    if len({str(row["key"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate key in selected S3 inventory")
    return sorted(selected, key=lambda row: str(row["key"]))


def file_hash(path: Path, algorithm: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(key: str) -> Path:
    if not key.startswith(S3_PREFIX):
        raise ValueError(f"Unexpected S3 key: {key}")
    path = Path(key[len(S3_PREFIX) :])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe S3 key: {key}")
    return path


def download_one(
    row: dict[str, object], output_root: Path, overwrite: bool
) -> dict[str, object]:
    key = str(row["key"])
    expected_size = int(row["size_bytes"])
    etag = str(row["etag"])
    if not etag or "-" in etag:
        raise ValueError(f"Cannot use multipart or missing ETag as MD5: {key} ({etag})")
    destination = output_root / relative_path(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "downloaded"
    if destination.is_file() and not overwrite:
        if destination.stat().st_size == expected_size and file_hash(destination, "md5") == etag:
            status = "verified_existing"
        else:
            status = "replaced_invalid_existing"
    if status != "verified_existing":
        temporary = destination.with_name(f"{destination.name}.part")
        url = f"{S3_ORIGIN}/{urllib.parse.quote(key, safe='/')}"
        try:
            with request(url) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size != expected_size:
                raise ValueError(f"Downloaded size mismatch: {key}")
            if file_hash(temporary, "md5") != etag:
                raise ValueError(f"Downloaded ETag/MD5 mismatch: {key}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "s3_key": key,
        "relative_path": relative_path(key).as_posix(),
        "size_bytes": expected_size,
        "s3_etag_md5": etag,
        "sha256": file_hash(destination, "sha256"),
        "s3_last_modified": row["last_modified"],
        "status": status,
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")

    selected = select_objects(list_objects())
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, args.output_root, args.overwrite): row
            for row in selected
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(result)
            print(
                f"[{index}/{len(selected)}] {result['status']}: "
                f"{result['relative_path']}",
                flush=True,
            )
    completed.sort(key=lambda row: str(row["relative_path"]))

    provenance_dir = args.output_root / "_provenance"
    provenance_dir.mkdir(exist_ok=True)
    manifest_path = provenance_dir / "download_manifest.csv"
    write_manifest(manifest_path, completed)
    t1_rows = [row for row in completed if str(row["relative_path"]).endswith("_T1w.nii.gz")]
    participants_with_session_1 = {
        str(row["relative_path"]).split("/")[0]
        for row in t1_rows
        if "/ses-1/" in f"/{row['relative_path']}"
    }
    participants_with_session_2 = {
        str(row["relative_path"]).split("/")[0]
        for row in t1_rows
        if "/ses-2/" in f"/{row['relative_path']}"
    }
    metadata = {
        "dataset": "BNU1 / CoRR",
        "dataset_doi": "10.15387/fcp_indi.corr.bnu1",
        "source_prefix": f"s3://fcp-indi/{S3_PREFIX}",
        "source_documentation": (
            "https://fcon_1000.projects.nitrc.org/indi/CoRR/html/bnu_1.html"
        ),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "selected_object_count": len(completed),
        "t1_file_count": len(t1_rows),
        "t1_bytes": sum(int(row["size_bytes"]) for row in t1_rows),
        "participants_session_1_t1": len(participants_with_session_1),
        "participants_session_2_t1": len(participants_with_session_2),
        "complete_t1_pairs": len(
            participants_with_session_1 & participants_with_session_2
        ),
        "missing_session_2_t1": sorted(
            participants_with_session_1 - participants_with_session_2
        ),
        "integrity": "S3 ETag/MD5 plus local SHA-256",
        "use_policy": (
            "FCP/INDI states unrestricted usage for non-commercial research; "
            "verify institutional requirements before analysis or redistribution."
        ),
        "analysis_status": (
            "Acquisition only. Inclusion and QC must be locked before model outputs."
        ),
    }
    (provenance_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified BNU1 T1 acquisition: {manifest_path}")


if __name__ == "__main__":
    main()
