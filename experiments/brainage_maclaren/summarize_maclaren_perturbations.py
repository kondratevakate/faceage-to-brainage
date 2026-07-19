#!/usr/bin/env python3
"""Summarize the locked three-subject NeuroFM perturbation screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DECLARED_LATENT_DIMENSIONS = 161
EXECUTED_LATENT_DIMENSIONS = 160


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(value)
    return result


def text(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.8g}"


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else math.nan


def summarize_group(group: list[dict[str, str]], label: str, value: str) -> dict[str, object]:
    successful = [row for row in group if row["status"] == "ok"]
    age_delta = np.asarray(
        [number(row["brain_age_delta_years"]) for row in successful], dtype=float
    )
    ventricle_pct = np.asarray(
        [number(row["ventricle_volume_delta_fraction"]) for row in successful],
        dtype=float,
    )
    brain_pct = np.asarray(
        [number(row["brain_volume_delta_fraction"]) for row in successful], dtype=float
    )
    latent_cosine = np.asarray(
        [number(row["latent_cosine_to_baseline"]) for row in successful], dtype=float
    )
    attempted = len(group)
    result: dict[str, object] = {
        "grouping": label,
        "group": value,
        "n_attempted": attempted,
        "n_successful": len(successful),
        "n_failed": attempted - len(successful),
        "failure_rate": text((attempted - len(successful)) / attempted),
        "n_participants": len({row["participant_id"] for row in group}),
        "mean_signed_brain_age_delta_years": "",
        "mean_abs_brain_age_delta_years": "",
        "p95_abs_brain_age_delta_years": "",
        "max_abs_brain_age_delta_years": "",
        "fraction_attempted_age_delta_within_2_year_margin": text(
            float(np.count_nonzero(np.abs(age_delta) <= 2.0) / attempted)
        ),
        "mean_abs_ventricle_volume_delta_fraction": "",
        "max_abs_ventricle_volume_delta_fraction": "",
        "fraction_attempted_ventricle_delta_within_5pct_margin": text(
            float(np.count_nonzero(np.abs(ventricle_pct) <= 0.05) / attempted)
        ),
        "mean_abs_brain_volume_delta_fraction": "",
        "max_abs_brain_volume_delta_fraction": "",
        "fraction_attempted_brain_volume_delta_within_5pct_margin": text(
            float(np.count_nonzero(np.abs(brain_pct) <= 0.05) / attempted)
        ),
        "sex_class_flips": sum(int(row["sex_class_flip"]) for row in successful),
        "median_latent_cosine_to_baseline": "",
        "equivalence_claim": "not_permitted_n_participants_3",
        "claim_level": "numerical_robustness_probe_only",
    }
    if successful:
        result.update(
            {
                "mean_signed_brain_age_delta_years": text(float(age_delta.mean())),
                "mean_abs_brain_age_delta_years": text(
                    float(np.mean(np.abs(age_delta)))
                ),
                "p95_abs_brain_age_delta_years": text(
                    float(np.percentile(np.abs(age_delta), 95))
                ),
                "max_abs_brain_age_delta_years": text(
                    float(np.max(np.abs(age_delta)))
                ),
                "mean_abs_ventricle_volume_delta_fraction": text(
                    float(np.mean(np.abs(ventricle_pct)))
                ),
                "max_abs_ventricle_volume_delta_fraction": text(
                    float(np.max(np.abs(ventricle_pct)))
                ),
                "mean_abs_brain_volume_delta_fraction": text(
                    float(np.mean(np.abs(brain_pct)))
                ),
                "max_abs_brain_volume_delta_fraction": text(
                    float(np.max(np.abs(brain_pct)))
                ),
                "median_latent_cosine_to_baseline": text(
                    float(np.median(latent_cosine))
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--results-summary", required=True, type=Path)
    parser.add_argument("--latent-array", required=True, type=Path)
    parser.add_argument("--latent-index", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = read_csv(args.input_manifest)
    results = read_csv(args.results_summary)
    index_rows = read_csv(args.latent_index)
    baseline_predictions = read_csv(args.baseline_predictions)
    embeddings = np.asarray(np.load(args.latent_array), dtype=float)
    if len(manifest) != 72:
        raise ValueError(
            f"Expected 72 manifest rows, found {len(manifest)}"
        )
    if len(results) != len(index_rows):
        raise ValueError(
            f"Results/index row mismatch: {len(results)}/{len(index_rows)}"
        )
    if embeddings.shape != (
        len(index_rows),
        EXECUTED_LATENT_DIMENSIONS,
    ) or not np.isfinite(embeddings).all():
        raise ValueError(f"Unexpected perturbation latent array: {embeddings.shape}")

    manifest_by_input = {row["input"]: row for row in manifest}
    results_by_input = {row["input"]: row for row in results}
    embedding_by_input = {
        row["input"]: embeddings[index] for index, row in enumerate(index_rows)
    }
    if len(manifest_by_input) != 72 or len(results_by_input) != len(results):
        raise ValueError("Duplicate input path in perturbation manifest or results")
    if len(embedding_by_input) != len(index_rows):
        raise ValueError("Duplicate input path in perturbation latent index")
    if set(results_by_input) != set(embedding_by_input):
        raise ValueError("Perturbation results and latent index do not align")
    if not set(results_by_input).issubset(manifest_by_input):
        raise ValueError("Perturbation outputs contain an input outside the manifest")

    baseline_results: dict[str, dict[str, str]] = {}
    baseline_embeddings: dict[str, np.ndarray] = {}
    for input_path, row in manifest_by_input.items():
        if row["perturbation"] == "baseline":
            baseline_results[row["participant_id"]] = results_by_input[input_path]
            baseline_embeddings[row["participant_id"]] = embedding_by_input[input_path]
    if set(baseline_results) != {"sub-01", "sub-02", "sub-03"}:
        raise ValueError("Missing locked baseline for at least one participant")
    if any(np.linalg.norm(value) <= 0 for value in embedding_by_input.values()):
        raise ValueError("Perturbation embeddings contain a zero-norm row")

    baseline_reference = {
        row["participant_id"]: row
        for row in baseline_predictions
        if int(row["run_index"]) == 20
    }
    if set(baseline_reference) != {"sub-01", "sub-02", "sub-03"}:
        raise ValueError("Baseline prediction table has no complete locked run-20 reference")
    reference_fields = {
        "brain_age": "predicted_brain_age_years",
        "sex": "predicted_sex_binary",
        "ventricle_volume": "predicted_ventricle_volume_mm3",
        "brain_volume": "predicted_brain_volume_mm3",
    }
    reference_differences: list[float] = []
    for participant, rerun in baseline_results.items():
        reference = baseline_reference[participant]
        for result_field, reference_field in reference_fields.items():
            difference = abs(number(rerun[result_field]) - number(reference[reference_field]))
            reference_differences.append(difference)
            if not np.isclose(
                number(rerun[result_field]),
                number(reference[reference_field]),
                rtol=1e-6,
                atol=1e-5,
            ):
                raise ValueError(
                    f"Perturbation-batch baseline differs from locked baseline: "
                    f"{participant} {result_field} ({difference})"
                )

    output_rows: list[dict[str, str]] = []
    for input_path, source in manifest_by_input.items():
        if input_path not in results_by_input:
            output_rows.append(
                {
                    "participant_id": source["participant_id"],
                    "run_index": source["run_index"],
                    "relative_path": source["relative_path"],
                    "base_input_sha256": source["base_input_sha256"],
                    "perturbation": source["perturbation"],
                    "perturbation_family": source["perturbation_family"],
                    "perturbation_level": source["perturbation_level"],
                    "perturbation_axis": source["perturbation_axis"],
                    "perturbation_unit": source["perturbation_unit"],
                    "input_sha256": source["input_sha256"],
                    "status": "failed",
                    "failure_reason": "official_neurofm_no_summary_output",
                    "predicted_brain_age_years": "",
                    "brain_age_delta_years": "",
                    "predicted_sex_binary": "",
                    "sex_class_flip": "",
                    "predicted_ventricle_volume_mm3": "",
                    "ventricle_volume_delta_mm3": "",
                    "ventricle_volume_delta_fraction": "",
                    "predicted_brain_volume_mm3": "",
                    "brain_volume_delta_mm3": "",
                    "brain_volume_delta_fraction": "",
                    "latent_cosine_to_baseline": "",
                    "claim_level": "numerical_robustness_probe_only",
                }
            )
            continue
        result = results_by_input[input_path]
        participant = source["participant_id"]
        baseline = baseline_results[participant]
        baseline_latent = baseline_embeddings[participant]
        age = number(result["brain_age"])
        baseline_age = number(baseline["brain_age"])
        ventricle = number(result["ventricle_volume"])
        baseline_ventricle = number(baseline["ventricle_volume"])
        brain = number(result["brain_volume"])
        baseline_brain = number(baseline["brain_volume"])
        if baseline_ventricle == 0 or baseline_brain == 0:
            raise ValueError(f"Zero baseline volume for {participant}")
        sex = number(result["sex"])
        baseline_sex = number(baseline["sex"])
        output_rows.append(
            {
                "participant_id": participant,
                "run_index": source["run_index"],
                "relative_path": source["relative_path"],
                "base_input_sha256": source["base_input_sha256"],
                "perturbation": source["perturbation"],
                "perturbation_family": source["perturbation_family"],
                "perturbation_level": source["perturbation_level"],
                "perturbation_axis": source["perturbation_axis"],
                "perturbation_unit": source["perturbation_unit"],
                "input_sha256": source["input_sha256"],
                "status": "ok",
                "failure_reason": "",
                "predicted_brain_age_years": text(age),
                "brain_age_delta_years": text(age - baseline_age),
                "predicted_sex_binary": text(sex),
                "sex_class_flip": "1" if sex != baseline_sex else "0",
                "predicted_ventricle_volume_mm3": text(ventricle),
                "ventricle_volume_delta_mm3": text(ventricle - baseline_ventricle),
                "ventricle_volume_delta_fraction": text(
                    (ventricle - baseline_ventricle) / baseline_ventricle
                ),
                "predicted_brain_volume_mm3": text(brain),
                "brain_volume_delta_mm3": text(brain - baseline_brain),
                "brain_volume_delta_fraction": text((brain - baseline_brain) / baseline_brain),
                "latent_cosine_to_baseline": text(
                    cosine(embedding_by_input[input_path], baseline_latent)
                ),
                "claim_level": "numerical_robustness_probe_only",
            }
        )
    output_rows.sort(
        key=lambda row: (row["perturbation_family"], row["perturbation"], row["participant_id"])
    )

    nonbaseline = [row for row in output_rows if row["perturbation_family"] != "baseline"]
    by_perturbation: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nonbaseline:
        by_perturbation[row["perturbation"]].append(row)
        by_family[row["perturbation_family"]].append(row)
    summary_rows = [
        summarize_group(group, "perturbation", key)
        for key, group in sorted(by_perturbation.items())
    ] + [
        summarize_group(group, "family", key) for key, group in sorted(by_family.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "maclaren_perturbation_predictions.csv", output_rows)
    write_csv(args.output_dir / "maclaren_perturbation_summary.csv", summary_rows)
    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "model": "rockNroll87q/NeuroFM neurofm-s",
        "source_commit": "d4e3c463910d939a681d24ebdeb26d44dea6878f",
        "n_rows": len(output_rows),
        "n_nonbaseline_rows": len(nonbaseline),
        "n_successful": sum(row["status"] == "ok" for row in output_rows),
        "n_failed": sum(row["status"] == "failed" for row in output_rows),
        "failed_cases": [
            {
                "participant_id": row["participant_id"],
                "relative_path": row["relative_path"],
                "perturbation": row["perturbation"],
                "reason": row["failure_reason"],
            }
            for row in output_rows
            if row["status"] == "failed"
        ],
        "observed_failure_note": (
            "All three resolution_1mm inputs were 236x270x270 at 1 mm RAS. "
            "The official NeuroFM fast conform path attempted padding/reorientation, "
            "could not crop axes larger than 256, and therefore supplied an "
            "incompatible shape to the model."
        ),
        "n_participants": 3,
        "latent_shape": list(embeddings.shape),
        "latent_dimensions_declared_by_registry": DECLARED_LATENT_DIMENSIONS,
        "latent_dimensions_observed_at_multihead_output": EXECUTED_LATENT_DIMENSIONS,
        "latent_dimension_contract_note": (
            "At commit d4e3c46 NeuroFM-S declares 161 dimensions but executes a "
            "160-dimensional multihead_output layer; the upstream model and weights "
            "were not modified."
        ),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "results_summary_sha256": sha256_file(args.results_summary),
        "latent_array_sha256": sha256_file(args.latent_array),
        "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
        "baseline_rerun_max_absolute_scalar_difference": max(reference_differences),
        "baseline_rerun_comparison_tolerance": {"rtol": 1e-6, "atol": 1e-5},
        "equivalence_test_permitted": False,
        "critical_interpretation": (
            "Observed perturbation deltas are a three-person numerical sensitivity "
            "screen. They do not establish equivalence, biological response, age "
            "accuracy, segmentation validity, morphometry, or clinical utility."
        ),
    }
    (args.output_dir / "maclaren_perturbation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote perturbation summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
