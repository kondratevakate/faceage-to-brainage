#!/usr/bin/env python3
"""Run HD-BET with bounded worker counts for low-memory CPU environments.

HD-BET 2.0.1 hard-codes four preprocessing and eight export workers in its
CLI. This wrapper uses the same package predictor and checkpoint while making
those worker counts explicit and applying the predicted masks sequentially.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import nifti_files

from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    parser.add_argument("--export-workers", type=int, default=1)
    args = parser.parse_args()

    if args.preprocessing_workers < 1 or args.export_workers < 1:
        raise ValueError("Worker counts must be positive")
    inputs = [Path(path) for path in sorted(nifti_files(str(args.input_folder)))]
    if not inputs:
        raise ValueError(f"No NIfTI inputs in {args.input_folder}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_folder / path.name for path in inputs]
    masks = [args.output_folder / f"{path.name[:-7]}_bet.nii.gz" for path in inputs]

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    predictor.predict_from_files(
        [[str(path)] for path in inputs],
        [str(path) for path in masks],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=args.preprocessing_workers,
        num_processes_segmentation_export=args.export_workers,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    for source, mask, output in zip(inputs, masks, outputs):
        apply_bet(str(source), str(mask), str(output))

    for metadata_name in (
        "dataset.json",
        "plans.json",
        "predict_from_raw_data_args.json",
    ):
        metadata_path = args.output_folder / metadata_name
        if metadata_path.exists():
            metadata_path.unlink()

    print(f"HD-BET low-memory wrapper completed {len(inputs)} scan(s)")


if __name__ == "__main__":
    main()
