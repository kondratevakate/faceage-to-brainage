#!/usr/bin/env python3
"""
Batch SFCN inference driven by a local JSON config.

This script is intended for the first "usable baseline" milestone:
run the official pretrained SFCN model on the cleaned IXI/SIMON manifests
and write project-ready prediction CSVs with a stable schema.
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "local" / "brain_age_runtime.json"


def get_required(mapping, key, context):
    value = mapping.get(key)
    if value in (None, ""):
        raise KeyError(f"Missing required key '{key}' in {context}")
    return value


def get_optional_row_value(row, column_name):
    if not column_name:
        return ""
    if column_name not in row.index:
        return ""
    value = row[column_name]
    if pd.isna(value):
        return ""
    return value


def to_float_or_nan(value):
    if value in ("", None) or pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Batch SFCN inference from manifest CSVs")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to local JSON config. Defaults to config/local/brain_age_runtime.json",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "ixi", "simon"),
        default="all",
        help="Which dataset block to run from the config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute preprocessed inputs and overwrite output CSVs",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Copy config/brain_age_runtime.example.json into config/local/brain_age_runtime.json and fill in local paths."
        )

    config = load_config(config_path)
    runtime_cfg = config.get("runtime", {})
    sfcn_cfg = config.get("sfcn", {})
    datasets_cfg = config.get("datasets", {})

    device = runtime_cfg.get("device", "cpu")
    age_bins_cfg = sfcn_cfg.get("age_bins", {})
    model_dir = Path(get_required(sfcn_cfg, "model_dir", "sfcn"))
    weight_path = Path(get_required(sfcn_cfg, "weight_path", "sfcn"))
    skullstrip_command = sfcn_cfg.get("skullstrip_command", "mri_synthstrip")
    skip_skullstrip = bool(sfcn_cfg.get("skip_skullstrip", False))
    keep_skullstripped = bool(sfcn_cfg.get("keep_skullstripped", False))
    age_bin_start = float(age_bins_cfg.get("start", 42.0))
    age_bin_step = float(age_bins_cfg.get("step", 1.0))
    age_bin_count = int(age_bins_cfg.get("count", 40))

    sys.path.insert(0, str(REPO_ROOT))
    from src.brain_age import prepare_sfcn_input, predict_sfcn

    selected = ["ixi", "simon"] if args.dataset == "all" else [args.dataset]
    for dataset_name in selected:
        dataset_cfg = datasets_cfg.get(dataset_name)
        if dataset_cfg is None:
            raise KeyError(f"Missing dataset config block: {dataset_name}")

        manifest_csv = Path(get_required(dataset_cfg, "manifest_csv", dataset_name))
        output_csv = Path(get_required(dataset_cfg, "output_csv", dataset_name))
        preproc_dir = Path(get_required(dataset_cfg, "preproc_dir", dataset_name))
        input_path_column = get_required(dataset_cfg, "input_path_column", dataset_name)
        chron_age_column = get_required(dataset_cfg, "chron_age_column", dataset_name)
        subject_id_column = get_required(dataset_cfg, "subject_id_column", dataset_name)
        session_id_column = dataset_cfg.get("session_id_column", "")
        scan_id_column = dataset_cfg.get("scan_id_column", "")
        run_column = dataset_cfg.get("run_column", "")
        acquisition_label_column = dataset_cfg.get("acquisition_label_column", "")

        frame = pd.read_csv(manifest_csv)
        if args.limit is not None:
            frame = frame.head(args.limit).copy()

        preproc_dir.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"SFCN {dataset_name}"):
            input_path = Path(get_required(row, input_path_column, f"{dataset_name} row"))
            stem = input_path.name.replace(".nii.gz", "").replace(".nii", "").replace(".mgz", "")
            preproc_path = preproc_dir / f"{stem}_sfcn_input.nii.gz"

            subject_id = get_optional_row_value(row, subject_id_column)
            session_id = get_optional_row_value(row, session_id_column)
            scan_id = get_optional_row_value(row, scan_id_column)
            run = get_optional_row_value(row, run_column)
            acquisition_label = get_optional_row_value(row, acquisition_label_column)
            chron_age = to_float_or_nan(get_optional_row_value(row, chron_age_column))

            record = {
                "dataset": dataset_name.upper(),
                "subject_id": subject_id,
                "session_id": session_id,
                "scan_id": scan_id,
                "run": run,
                "acquisition_label": acquisition_label,
                "chron_age": chron_age,
                "model_name": "SFCN",
                "predicted_age": float("nan"),
                "brain_age_gap": float("nan"),
                "input_path": str(input_path),
                "preproc_path": str(preproc_path),
                "status": "error",
                "error": "",
            }

            try:
                if args.overwrite or not preproc_path.exists():
                    prepare_sfcn_input(
                        nifti_path=input_path,
                        out_path=preproc_path,
                        skullstrip=not skip_skullstrip,
                        skullstrip_command=skullstrip_command,
                        keep_skullstripped=keep_skullstripped,
                    )

                predicted_age = predict_sfcn(
                    nifti_path=preproc_path,
                    model_dir=model_dir,
                    weight_path=weight_path,
                    device=device,
                    age_bin_start=age_bin_start,
                    age_bin_step=age_bin_step,
                    age_bin_count=age_bin_count,
                )
                record["predicted_age"] = predicted_age
                if not math.isnan(chron_age):
                    record["brain_age_gap"] = predicted_age - chron_age
                record["status"] = "ok"
            except Exception as exc:
                record["error"] = str(exc)

            records.append(record)

        out_df = pd.DataFrame(records)
        out_df.to_csv(output_csv, index=False)
        ok_count = int((out_df["status"] == "ok").sum())
        log.info("Saved %s predictions to %s (%s ok / %s total)", dataset_name, output_csv, ok_count, len(out_df))


if __name__ == "__main__":
    main()
