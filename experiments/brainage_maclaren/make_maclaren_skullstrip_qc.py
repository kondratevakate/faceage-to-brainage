#!/usr/bin/env python3
"""Render deterministic external HD-BET QC montages for locked Maclaren scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


DEFAULT_STATUS = Path(
    "/mnt/d/data/faceage-to-brainage/derivatives/hdbet/2.0.1/"
    "maclaren_ds000239/R1.0.1/preprocessing_status.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/d/data/faceage-to-brainage/derivatives/skullstrip_qc/"
    "maclaren_ds000239/hdbet_2.0.1"
)
PARTICIPANTS = ["sub-01", "sub-02", "sub-03"]
RUNS = [1, 20, 40]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def oriented_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(data, index, axis=axis))


def mask_topology(row: dict[str, str]) -> dict[str, object]:
    mask_path = Path(row["mask_path"])
    mask_data = np.asanyarray(nib.load(str(mask_path)).dataobj)
    mask = mask_data > 0
    unique_mask = np.unique(mask_data)
    if not set(unique_mask.tolist()).issubset({0, 1}) or not np.any(mask):
        raise ValueError(f"Mask is empty or non-binary: {mask_path}")
    labels, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    counts = np.bincount(labels.ravel())[1:]
    counts.sort()
    coordinates = np.argwhere(mask)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    widths = upper - lower + 1
    touches_border = any(
        lower[axis] == 0 or upper[axis] == mask.shape[axis] - 1 for axis in range(3)
    )
    return {
        "participant_id": row["participant_id"],
        "run_index": row["run_index"],
        "relative_path": row["relative_path"],
        "mask_sha256": row["mask_sha256"],
        "mask_nonzero_voxels": int(mask.sum()),
        "mask_fraction": row["mask_fraction"],
        "components_26_connected": int(n_components),
        "largest_component_voxels": int(counts[-1]),
        "largest_component_fraction": f"{float(counts[-1] / mask.sum()):.8g}",
        "second_component_voxels": int(counts[-2]) if counts.size > 1 else 0,
        "bbox_width_axis0": int(widths[0]),
        "bbox_width_axis1": int(widths[1]),
        "bbox_width_axis2": int(widths[2]),
        "touches_volume_border": int(touches_border),
        "claim_level": "skullstrip_input_qc_not_segmentation_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--compact-output-dir", type=Path, default=Path("data/brainage")
    )
    args = parser.parse_args()

    rows = read_csv(args.status)
    successful = [row for row in rows if row["status"] == "ok"]
    if len(successful) != 120:
        raise ValueError(f"Expected 120 successful masks, found {len(successful)}")
    selected = [
        row
        for row in successful
        if row["participant_id"] in PARTICIPANTS
        and int(row["run_index"]) in RUNS
    ]
    selected.sort(key=lambda row: (row["participant_id"], int(row["run_index"])))
    expected = {(participant, run) for participant in PARTICIPANTS for run in RUNS}
    actual = {(row["participant_id"], int(row["run_index"])) for row in selected}
    if actual != expected or len(selected) != 9:
        raise ValueError(f"Expected nine successful locked QC scans, found {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology_rows = [mask_topology(row) for row in successful]
    topology_rows.sort(
        key=lambda row: (str(row["participant_id"]), int(row["run_index"]))
    )
    write_csv(args.output_dir / "all_mask_topology_qc.csv", topology_rows)
    args.compact_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.compact_output_dir / "maclaren_skullstrip_topology_qc.csv",
        topology_rows,
    )
    qc_rows: list[dict[str, object]] = []
    axis_labels = ["voxel axis 0", "voxel axis 1", "voxel axis 2"]

    for participant in PARTICIPANTS:
        participant_rows = [row for row in selected if row["participant_id"] == participant]
        figure, axes = plt.subplots(3, 3, figsize=(10.5, 10.5), constrained_layout=True)
        for row_index, row in enumerate(participant_rows):
            source_path = Path(row["source_image_runtime"])
            mask_path = Path(row["mask_path"])
            source_image = nib.load(str(source_path))
            mask_image = nib.load(str(mask_path))
            source = source_image.get_fdata(dtype=np.float32)
            mask_data = np.asanyarray(mask_image.dataobj)
            mask = mask_data > 0
            if source.shape != mask.shape or not np.allclose(
                source_image.affine, mask_image.affine, rtol=1e-5, atol=1e-4
            ):
                raise ValueError(f"Source/mask geometry mismatch: {mask_path}")
            unique_mask = np.unique(mask_data)
            if not set(unique_mask.tolist()).issubset({0, 1}) or not np.any(mask):
                raise ValueError(f"Mask is empty or non-binary: {mask_path}")
            center = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
            brain_values = source[mask]
            vmin, vmax = np.percentile(brain_values, [0.5, 99.5])
            if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
                raise ValueError(f"Invalid display range: {source_path}")

            for axis in range(3):
                panel = axes[row_index, axis]
                image_slice = oriented_slice(source, axis, int(center[axis]))
                mask_slice = oriented_slice(mask, axis, int(center[axis]))
                panel.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)
                panel.contour(
                    mask_slice.astype(float),
                    levels=[0.5],
                    colors=["#e53935"],
                    linewidths=0.8,
                )
                panel.set_title(
                    f"run-{int(row['run_index']):02d} | {axis_labels[axis]}",
                    fontsize=9,
                )
                panel.axis("off")

            qc_rows.append(
                {
                    "participant_id": participant,
                    "run_index": row["run_index"],
                    "source_sha256": row["source_sha256"],
                    "mask_sha256": row["mask_sha256"],
                    "mask_fraction": row["mask_fraction"],
                    "mask_center_voxel_axis0": int(center[0]),
                    "mask_center_voxel_axis1": int(center[1]),
                    "mask_center_voxel_axis2": int(center[2]),
                    "source_mask_shape_match": 1,
                    "source_mask_affine_match": 1,
                    "mask_binary": 1,
                }
            )

        figure.suptitle(
            f"Maclaren {participant}: HD-BET contour on source T1w",
            fontsize=13,
        )
        output_path = args.output_dir / f"{participant}_runs-01-20-40_qc.png"
        figure.savefig(output_path, dpi=180, facecolor="white")
        plt.close(figure)

    summary_path = args.output_dir / "selected_skullstrip_qc.csv"
    write_csv(summary_path, qc_rows)
    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "selection_rule": "run_index in {1,20,40} for each participant",
        "n_scans": len(qc_rows),
        "n_topology_scans": len(topology_rows),
        "n_multicomponent_masks": sum(
            int(row["components_26_connected"]) > 1 for row in topology_rows
        ),
        "n_masks_largest_component_fraction_below_0p995": sum(
            float(row["largest_component_fraction"]) < 0.995
            for row in topology_rows
        ),
        "n_masks_touching_volume_border": sum(
            int(row["touches_volume_border"]) for row in topology_rows
        ),
        "mask_cleanup_applied": False,
        "cleanup_decision": (
            "Retain official HD-BET masks. Largest-component filtering would only "
            "remove islands of at most 21 voxels and would not address connected "
            "anterior inclusions; no outcome-informed cleanup is introduced."
        ),
        "views": axis_labels,
        "slice_rule": "rounded center of mass of the binary HD-BET mask",
        "overlay": "HD-BET binary-mask contour on source T1w",
        "status_sha256": sha256_file(args.status),
        "interpretation": (
            "This montage supports visual skull-strip QC only. It is not ground-truth "
            "segmentation validation or morphometric validation."
        ),
    }
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    (args.output_dir / "skullstrip_qc_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )
    (args.compact_output_dir / "maclaren_skullstrip_qc_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )
    print(f"Wrote external skull-strip QC to {args.output_dir}")


if __name__ == "__main__":
    main()
