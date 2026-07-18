#!/usr/bin/env python3
"""Compare two skull-strip masks on the same grid for preprocessing QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def percentile_surface_distance(
    first: np.ndarray,
    second: np.ndarray,
    voxel_size: tuple[float, float, float],
    percentile: float,
) -> float:
    first_surface = surface(first)
    second_surface = surface(second)
    distance_to_second = ndimage.distance_transform_edt(~second_surface, sampling=voxel_size)
    distance_to_first = ndimage.distance_transform_edt(~first_surface, sampling=voxel_size)
    distances = np.concatenate(
        [distance_to_second[first_surface], distance_to_first[second_surface]]
    )
    return float(np.percentile(distances, percentile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference_image = nib.load(str(args.reference))
    candidate_image = nib.load(str(args.candidate))
    if reference_image.shape != candidate_image.shape:
        raise ValueError(
            f"Mask shape mismatch: {reference_image.shape} vs {candidate_image.shape}"
        )
    if not np.allclose(reference_image.affine, candidate_image.affine, atol=1e-4):
        raise ValueError("Mask affine mismatch")

    reference = np.asanyarray(reference_image.dataobj) > 0
    candidate = np.asanyarray(candidate_image.dataobj) > 0
    if not reference.any() or not candidate.any():
        raise ValueError("At least one mask is empty")

    intersection = int(np.count_nonzero(reference & candidate))
    union = int(np.count_nonzero(reference | candidate))
    reference_n = int(np.count_nonzero(reference))
    candidate_n = int(np.count_nonzero(candidate))
    voxel_size = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
    voxel_volume = float(np.prod(voxel_size))
    reference_centroid = np.asarray(ndimage.center_of_mass(reference), dtype=float)
    candidate_centroid = np.asarray(ndimage.center_of_mass(candidate), dtype=float)

    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference_image.shape),
        "voxel_size_mm": list(voxel_size),
        "reference_voxels": reference_n,
        "candidate_voxels": candidate_n,
        "reference_volume_mm3": reference_n * voxel_volume,
        "candidate_volume_mm3": candidate_n * voxel_volume,
        "candidate_to_reference_volume_ratio": candidate_n / reference_n,
        "dice": 2.0 * intersection / (reference_n + candidate_n),
        "jaccard": intersection / union,
        "candidate_only_fraction_of_reference": int(np.count_nonzero(candidate & ~reference))
        / reference_n,
        "reference_only_fraction_of_reference": int(np.count_nonzero(reference & ~candidate))
        / reference_n,
        "centroid_distance_mm": float(
            np.linalg.norm((candidate_centroid - reference_centroid) * np.asarray(voxel_size))
        ),
        "symmetric_surface_distance_p95_mm": percentile_surface_distance(
            reference, candidate, voxel_size, 95.0
        ),
        "interpretation": (
            "Agreement between skull-strip methods is preprocessing QC only; "
            "neither mask is treated as anatomical ground truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
