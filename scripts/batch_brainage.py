#!/usr/bin/env python3
"""
Batch brain age inference for multiple models: SFCN, SynthBA, MIDIBrainAge.

Usage examples:
  python scripts/batch_brainage.py --model sfcn   --dataset ixi --tta
  python scripts/batch_brainage.py --model synthba --dataset ixi
  python scripts/batch_brainage.py --model midi   --dataset ixi
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
DEFAULT_CONFIG = REPO_ROOT / "config" / "local" / "brain_age_local.json"


def get_required(mapping, key, context):
    value = mapping.get(key)
    if value in (None, ""):
        raise KeyError(f"Missing required key '{key}' in {context}")
    return value


def get_optional(row, col):
    if not col or col not in row.index:
        return ""
    v = row[col]
    return "" if pd.isna(v) else v


def to_float_or_nan(value):
    if value in ("", None) or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Batch brain age inference")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", choices=("sfcn", "synthba", "midi"), required=True)
    parser.add_argument("--dataset", choices=("all", "ixi", "simon"), default="ixi")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation (L-R flip + voxel shifts for SFCN; L-R flip for SynthBA)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text())

    runtime_cfg = config.get("runtime", {})
    sfcn_cfg = config.get("sfcn", {})
    datasets_cfg = config.get("datasets", {})

    device = runtime_cfg.get("device", "cpu")

    sys.path.insert(0, str(REPO_ROOT))
    from src.brain_age import (
        predict_midi_brainage,
        predict_sfcn,
        predict_sfcn_tta,
        predict_synthba,
        predict_synthba_tta,
        prepare_sfcn_input,
    )

    # SFCN-specific config
    age_bins_cfg = sfcn_cfg.get("age_bins", {})
    model_dir = Path(sfcn_cfg.get("model_dir", str(REPO_ROOT / "vendor" / "SFCN")))
    weight_path = Path(sfcn_cfg.get("weight_path", ""))
    skullstrip_command = sfcn_cfg.get("skullstrip_command", "deepbet")
    skip_skullstrip = bool(sfcn_cfg.get("skip_skullstrip", False))
    keep_skullstripped = bool(sfcn_cfg.get("keep_skullstripped", False))
    n4_correct = bool(sfcn_cfg.get("n4_correct", True))
    register_mni = bool(sfcn_cfg.get("register_mni", True))
    age_bin_start = float(age_bins_cfg.get("start", 42.0))
    age_bin_step = float(age_bins_cfg.get("step", 1.0))
    age_bin_count = int(age_bins_cfg.get("count", 40))

    midi_dir = REPO_ROOT / "vendor" / "MIDIBrainAge"
    model_label = args.model.upper() + ("+TTA" if args.tta else "")

    selected = ["ixi", "simon"] if args.dataset == "all" else [args.dataset]

    for dataset_name in selected:
        dataset_cfg = datasets_cfg.get(dataset_name)
        if dataset_cfg is None:
            raise KeyError(f"Missing dataset block: {dataset_name}")

        manifest_csv = Path(get_required(dataset_cfg, "manifest_csv", dataset_name))
        input_col = get_required(dataset_cfg, "input_path_column", dataset_name)
        age_col = get_required(dataset_cfg, "chron_age_column", dataset_name)
        subject_col = get_required(dataset_cfg, "subject_id_column", dataset_name)
        session_col = dataset_cfg.get("session_id_column", "")
        scan_col = dataset_cfg.get("scan_id_column", "")
        run_col = dataset_cfg.get("run_column", "")
        acq_col = dataset_cfg.get("acquisition_label_column", "")

        # output paths per model
        base_out = Path(dataset_cfg.get("output_csv", "")).parent
        output_csv = base_out / f"{args.model}_predictions{'_tta' if args.tta else ''}.csv"
        preproc_dir = Path(dataset_cfg.get("preproc_dir", str(REPO_ROOT / "results" / "preproc" / dataset_name)))
        preproc_dir.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        frame = pd.read_csv(manifest_csv)
        if args.limit:
            frame = frame.head(args.limit).copy()

        records = []
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"{model_label} {dataset_name}"):
            input_path = Path(get_required(row, input_col, dataset_name))
            stem = input_path.name.replace(".nii.gz", "").replace(".nii", "").replace(".mgz", "")
            preproc_path = preproc_dir / f"{stem}_sfcn_input.nii.gz"

            record = {
                "dataset": dataset_name.upper(),
                "subject_id": get_optional(row, subject_col),
                "session_id": get_optional(row, session_col),
                "scan_id": get_optional(row, scan_col),
                "run": get_optional(row, run_col),
                "acquisition_label": get_optional(row, acq_col),
                "chron_age": to_float_or_nan(get_optional(row, age_col)),
                "model_name": model_label,
                "predicted_age": float("nan"),
                "predicted_age_std": float("nan"),
                "n_aug": 1,
                "brain_age_gap": float("nan"),
                "input_path": str(input_path),
                "status": "error",
                "error": "",
            }

            try:
                if args.model == "sfcn":
                    if args.overwrite or not preproc_path.exists():
                        prepare_sfcn_input(
                            nifti_path=input_path,
                            out_path=preproc_path,
                            skullstrip=not skip_skullstrip,
                            skullstrip_command=skullstrip_command,
                            keep_skullstripped=keep_skullstripped,
                            n4_correct=n4_correct,
                            register_mni=register_mni,
                        )
                    if args.tta:
                        r = predict_sfcn_tta(preproc_path, model_dir, device=device,
                                             weight_path=weight_path,
                                             age_bin_start=age_bin_start,
                                             age_bin_step=age_bin_step,
                                             age_bin_count=age_bin_count)
                        predicted_age = r["mean"]
                        record["predicted_age_std"] = r["std"]
                        record["n_aug"] = r["n_aug"]
                    else:
                        predicted_age = predict_sfcn(preproc_path, model_dir, device=device,
                                                     weight_path=weight_path,
                                                     age_bin_start=age_bin_start,
                                                     age_bin_step=age_bin_step,
                                                     age_bin_count=age_bin_count)

                elif args.model == "synthba":
                    if args.tta:
                        r = predict_synthba_tta(input_path, device=device)
                        predicted_age = r["mean"]
                        record["predicted_age_std"] = r["std"]
                        record["n_aug"] = r["n_aug"]
                    else:
                        predicted_age = predict_synthba(input_path, device=device)

                elif args.model == "midi":
                    predicted_age = predict_midi_brainage(
                        input_path, midi_dir=midi_dir, device=device)

                record["predicted_age"] = predicted_age
                chron = record["chron_age"]
                if not math.isnan(chron):
                    record["brain_age_gap"] = predicted_age - chron
                record["status"] = "ok"

            except Exception as exc:
                log.warning("Error on %s: %s", input_path.name, exc)
                record["error"] = str(exc)

            records.append(record)

        out_df = pd.DataFrame(records)
        out_df.to_csv(output_csv, index=False)
        ok = int((out_df["status"] == "ok").sum())
        log.info("%s %s → %s  (%d ok / %d total)", model_label, dataset_name, output_csv, ok, len(out_df))


if __name__ == "__main__":
    main()
