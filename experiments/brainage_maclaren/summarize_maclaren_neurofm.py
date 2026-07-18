#!/usr/bin/env python3
"""Summarize locked Maclaren NeuroFM repeatability and feature stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


PARTICIPANTS = ["sub-01", "sub-02", "sub-03"]
RUNS = list(range(1, 41))
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260718
EXPECTED_WEIGHTS_SHA256 = "8015a0552214b87e43b5462b6c183f8d0da2d957d7ae11ed09a2e3355f5e991f"
SCALAR_OUTPUTS = [
    ("predicted_brain_age_years", "years"),
    ("predicted_ventricle_volume_mm3", "mm3"),
    ("predicted_brain_volume_mm3", "mm3"),
]


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


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def format_number(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.8g}"


def pooled_within_sd(matrix: np.ndarray) -> float:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    denominator = matrix.size - matrix.shape[0]
    return float(np.sqrt(np.sum(centered**2) / denominator))


def icc(matrix: np.ndarray, agreement: str) -> float:
    n_subjects, n_sessions = matrix.shape
    grand_mean = float(matrix.mean())
    subject_means = matrix.mean(axis=1)
    session_means = matrix.mean(axis=0)
    ms_subject = n_sessions * float(np.sum((subject_means - grand_mean) ** 2)) / (
        n_subjects - 1
    )
    ms_session = n_subjects * float(np.sum((session_means - grand_mean) ** 2)) / (
        n_sessions - 1
    )
    residual = matrix - subject_means[:, None] - session_means[None, :] + grand_mean
    ms_error = float(np.sum(residual**2)) / ((n_subjects - 1) * (n_sessions - 1))
    if agreement == "absolute":
        denominator = (
            ms_subject
            + (n_sessions - 1) * ms_error
            + n_sessions * (ms_session - ms_error) / n_subjects
        )
    elif agreement == "consistency":
        denominator = ms_subject + (n_sessions - 1) * ms_error
    else:
        raise ValueError(agreement)
    return float((ms_subject - ms_error) / denominator) if denominator != 0 else math.nan


def bootstrap_interval(
    matrix: np.ndarray,
    metric,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        value = float(metric(matrix[:, indices]))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def matrix_for(predictions: list[dict[str, str]], field: str) -> np.ndarray:
    by_key = {
        (row["participant_id"], int(row["run_index"])): finite_float(row[field])
        for row in predictions
    }
    expected = {(subject, run) for subject in PARTICIPANTS for run in RUNS}
    if set(by_key) != expected:
        raise ValueError(f"Incomplete participant/run matrix for {field}")
    return np.asarray(
        [[by_key[(subject, run)] for run in RUNS] for subject in PARTICIPANTS],
        dtype=float,
    )


def pairwise_absolute_differences(matrix: np.ndarray) -> np.ndarray:
    values = [
        abs(float(row[first] - row[second]))
        for row in matrix
        for first, second in combinations(range(matrix.shape[1]), 2)
    ]
    return np.asarray(values, dtype=float)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else math.nan


def build_predictions(
    preprocessing: list[dict[str, str]],
    result_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    preproc_by_input = {row["input"]: row for row in preprocessing if row["status"] == "ok"}
    if len(preproc_by_input) != 120:
        raise ValueError(
            f"Expected 120 successful preprocessing rows, found {len(preproc_by_input)}"
        )
    if len(result_rows) != 120:
        raise ValueError(f"Expected 120 NeuroFM results, found {len(result_rows)}")

    predictions: list[dict[str, str]] = []
    for result in result_rows:
        input_path = result["input"]
        source = preproc_by_input.get(input_path)
        if source is None:
            raise ValueError(f"NeuroFM result has no preprocessing row: {input_path}")
        predicted_age = finite_float(result["brain_age"])
        chronological_age = finite_float(source["chronological_age_years"])
        predictions.append(
            {
                "dataset_id": source["dataset_id"],
                "release": source["release"],
                "participant_id": source["participant_id"],
                "run_index": source["run_index"],
                "chronological_age_years": source["chronological_age_years"],
                "reported_sex": source["reported_sex"],
                "relative_path": source["relative_path"],
                "source_sha256": source["source_sha256"],
                "skullstrip_output_sha256": source["output_sha256"],
                "skullstrip_mask_sha256": source["mask_sha256"],
                "skullstrip_mask_fraction": source["mask_fraction"],
                "predicted_brain_age_years": format_number(predicted_age),
                "predicted_minus_chronological_years": format_number(
                    predicted_age - chronological_age
                ),
                "predicted_sex_binary": format_number(finite_float(result["sex"])),
                "predicted_ventricle_volume_mm3": format_number(
                    finite_float(result["ventricle_volume"])
                ),
                "predicted_brain_volume_mm3": format_number(
                    finite_float(result["brain_volume"])
                ),
                "age_accuracy_eligible": "0",
                "neurofm_age_range_status": "below_documented_40_to_90_range",
                "claim_level": "B2_test_retest_robustness_only",
            }
        )
    predictions.sort(key=lambda row: (row["participant_id"], int(row["run_index"])))
    if len({(row["participant_id"], row["run_index"]) for row in predictions}) != 120:
        raise ValueError("Duplicate participant/run output")
    return predictions


def summarize_scalars(
    predictions: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    participant_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for field, unit in SCALAR_OUTPUTS:
        matrix = matrix_for(predictions, field)
        for subject_index, subject in enumerate(PARTICIPANTS):
            values = matrix[subject_index]
            participant_rows.append(
                {
                    "participant_id": subject,
                    "output": field,
                    "unit": unit,
                    "n_runs": values.size,
                    "mean": format_number(float(values.mean())),
                    "sd": format_number(float(values.std(ddof=1))),
                    "median": format_number(float(np.median(values))),
                    "mad": format_number(float(np.median(np.abs(values - np.median(values))))),
                    "iqr": format_number(
                        float(np.percentile(values, 75) - np.percentile(values, 25))
                    ),
                    "min": format_number(float(values.min())),
                    "max": format_number(float(values.max())),
                    "range": format_number(float(np.ptp(values))),
                }
            )

        within_sd = pooled_within_sd(matrix)
        within_ci = bootstrap_interval(matrix, pooled_within_sd, rng)
        icc_absolute = icc(matrix, "absolute")
        icc_absolute_ci = bootstrap_interval(
            matrix, lambda values: icc(values, "absolute"), rng
        )
        icc_consistency = icc(matrix, "consistency")
        icc_consistency_ci = bootstrap_interval(
            matrix, lambda values: icc(values, "consistency"), rng
        )
        pairwise = pairwise_absolute_differences(matrix)
        summary_rows.append(
            {
                "output": field,
                "unit": unit,
                "n_participants": 3,
                "n_runs_per_participant": 40,
                "n_outputs": matrix.size,
                "pooled_within_subject_sd": format_number(within_sd),
                "pooled_within_subject_sd_bootstrap_low": format_number(within_ci[0]),
                "pooled_within_subject_sd_bootstrap_high": format_number(within_ci[1]),
                "repeatability_coefficient_95": format_number(1.96 * math.sqrt(2) * within_sd),
                "within_subject_pairwise_abs_diff_p50": format_number(
                    float(np.percentile(pairwise, 50))
                ),
                "within_subject_pairwise_abs_diff_p95": format_number(
                    float(np.percentile(pairwise, 95))
                ),
                "within_subject_pairwise_abs_diff_max": format_number(float(pairwise.max())),
                "icc_2_1_absolute": format_number(icc_absolute),
                "icc_2_1_session_bootstrap_low": format_number(icc_absolute_ci[0]),
                "icc_2_1_session_bootstrap_high": format_number(icc_absolute_ci[1]),
                "icc_3_1_consistency": format_number(icc_consistency),
                "icc_3_1_session_bootstrap_low": format_number(icc_consistency_ci[0]),
                "icc_3_1_session_bootstrap_high": format_number(icc_consistency_ci[1]),
                "ci_scope": "session_bootstrap_conditional_on_three_participants",
                "claim_level": "B2_test_retest_robustness_only",
            }
        )

    for subject in PARTICIPANTS:
        values = np.asarray(
            [
                finite_float(row["predicted_sex_binary"])
                for row in predictions
                if row["participant_id"] == subject
            ]
        )
        classes, counts = np.unique(values, return_counts=True)
        modal_index = int(np.argmax(counts))
        participant_rows.append(
            {
                "participant_id": subject,
                "output": "predicted_sex_binary",
                "unit": "class",
                "n_runs": values.size,
                "mean": format_number(float(values.mean())),
                "modal_class": format_number(float(classes[modal_index])),
                "n_nonmodal": int(values.size - counts[modal_index]),
                "fraction_nonmodal": format_number(
                    float((values.size - counts[modal_index]) / values.size)
                ),
                "claim_level": "model_audit_not_health_score",
            }
        )
    return participant_rows, summary_rows


def summarize_features(
    predictions: list[dict[str, str]],
    latent_array_path: Path,
    latent_index_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    embeddings = np.asarray(np.load(latent_array_path), dtype=float)
    index_rows = read_csv(latent_index_path)
    if embeddings.shape != (120, 161) or len(index_rows) != 120:
        raise ValueError(
            f"Expected latent shape (120, 161) and 120 index rows, found "
            f"{embeddings.shape} and {len(index_rows)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Latent embeddings contain non-finite values")

    ordered_metadata: list[dict[str, str]] = []
    preprocessing_inputs = {
        Path(row["relative_path"]).name: row for row in predictions
    }
    for index_row in index_rows:
        name = Path(index_row["input"]).name
        row = preprocessing_inputs.get(name)
        if row is None:
            raise ValueError(f"Latent index has no prediction metadata: {index_row['input']}")
        ordered_metadata.append(row)

    labels = np.asarray([row["participant_id"] for row in ordered_metadata])
    feature_mean = embeddings.mean(axis=0)
    feature_sd = embeddings.std(axis=0, ddof=1)
    nonconstant = feature_sd > 0
    standardized = np.zeros_like(embeddings)
    standardized[:, nonconstant] = (
        embeddings[:, nonconstant] - feature_mean[nonconstant]
    ) / feature_sd[nonconstant]

    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(embedding_norms <= 0):
        raise ValueError("Latent embeddings contain a zero-norm row")
    n_nonconstant = int(np.count_nonzero(nonconstant))
    if n_nonconstant == 0:
        raise ValueError("All latent dimensions are constant")
    normalized = embeddings / embedding_norms
    cosine_matrix = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(cosine_matrix, -np.inf)
    nearest_indices = np.argmax(cosine_matrix, axis=1)
    nearest_match = labels[nearest_indices] == labels

    scan_rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered_metadata):
        own = labels == labels[index]
        centroid = embeddings[own].mean(axis=0)
        standardized_centroid = standardized[own].mean(axis=0)
        scan_rows.append(
            {
                "participant_id": row["participant_id"],
                "run_index": row["run_index"],
                "relative_path": row["relative_path"],
                "cosine_similarity_to_subject_centroid": format_number(
                    cosine_similarity(embeddings[index], centroid)
                ),
                "standardized_l2_to_subject_centroid_per_sqrt_dim": format_number(
                    float(
                        np.linalg.norm(standardized[index] - standardized_centroid)
                        / math.sqrt(n_nonconstant)
                    )
                ),
                "nearest_neighbor_participant": labels[nearest_indices[index]],
                "nearest_neighbor_identity_match": "1" if nearest_match[index] else "0",
                "claim_level": "feature_stability_qc_only",
            }
        )

    within_distances: list[float] = []
    between_distances: list[float] = []
    for first, second in combinations(range(embeddings.shape[0]), 2):
        distance = 1.0 - float(normalized[first] @ normalized[second])
        if labels[first] == labels[second]:
            within_distances.append(distance)
        else:
            between_distances.append(distance)

    indices_by_subject = {
        subject: sorted(
            [
                index
                for index, row in enumerate(ordered_metadata)
                if row["participant_id"] == subject
            ],
            key=lambda index: int(ordered_metadata[index]["run_index"]),
        )
        for subject in PARTICIPANTS
    }
    if any(len(indices) != 40 for indices in indices_by_subject.values()):
        raise ValueError("Latent embeddings do not form a complete 3 x 40 design")

    dimension_rows: list[dict[str, object]] = []
    dimension_icc_values: list[float] = []
    for dimension in range(embeddings.shape[1]):
        values = np.vstack(
            [embeddings[indices_by_subject[subject], dimension] for subject in PARTICIPANTS]
        )
        absolute_icc = icc(values, "absolute")
        if math.isfinite(absolute_icc):
            dimension_icc_values.append(absolute_icc)
        dimension_rows.append(
            {
                "dimension": dimension,
                "icc_2_1_absolute": format_number(absolute_icc),
                "icc_3_1_consistency": format_number(icc(values, "consistency")),
                "overall_sd": format_number(float(values.std(ddof=1))),
                "pooled_within_subject_sd": format_number(pooled_within_sd(values)),
                "claim_level": "feature_stability_qc_only",
            }
        )

    dimension_icc = np.asarray(dimension_icc_values, dtype=float)
    if dimension_icc.size == 0:
        raise ValueError("No finite per-dimension ICC values")
    cosine_to_centroid = np.asarray(
        [finite_float(str(row["cosine_similarity_to_subject_centroid"])) for row in scan_rows]
    )
    standardized_l2 = np.asarray(
        [
            finite_float(str(row["standardized_l2_to_subject_centroid_per_sqrt_dim"]))
            for row in scan_rows
        ]
    )
    summary_rows = [
        {
            "n_scans": embeddings.shape[0],
            "n_participants": 3,
            "latent_dimensions": embeddings.shape[1],
            "nonconstant_dimensions": n_nonconstant,
            "dimension_icc_2_1_median": format_number(float(np.median(dimension_icc))),
            "dimension_icc_2_1_p10": format_number(float(np.percentile(dimension_icc, 10))),
            "dimension_icc_2_1_p90": format_number(float(np.percentile(dimension_icc, 90))),
            "dimension_icc_2_1_fraction_above_0p75": format_number(
                float(np.mean(dimension_icc > 0.75))
            ),
            "cosine_to_subject_centroid_median": format_number(
                float(np.median(cosine_to_centroid))
            ),
            "standardized_l2_to_centroid_median": format_number(float(np.median(standardized_l2))),
            "nearest_neighbor_identity_retention": format_number(float(np.mean(nearest_match))),
            "within_subject_cosine_distance_median": format_number(
                float(np.median(within_distances))
            ),
            "between_subject_cosine_distance_median": format_number(
                float(np.median(between_distances))
            ),
            "within_subject_cosine_distance_p95": format_number(
                float(np.percentile(within_distances, 95))
            ),
            "between_subject_cosine_distance_p05": format_number(
                float(np.percentile(between_distances, 5))
            ),
            "claim_level": "feature_stability_qc_only_not_downstream_validation",
        }
    ]
    return scan_rows, dimension_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing-status", required=True, type=Path)
    parser.add_argument("--results-summary", required=True, type=Path)
    parser.add_argument("--latent-array", required=True, type=Path)
    parser.add_argument("--latent-index", required=True, type=Path)
    parser.add_argument("--schema-validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    schema = json.loads(args.schema_validation.read_text(encoding="utf-8"))
    if schema.get("schema_status") != "pass":
        raise ValueError("NeuroFM schema validation did not pass")
    if schema.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Schema test used an unexpected NeuroFM-S weight")

    preprocessing = read_csv(args.preprocessing_status)
    results = read_csv(args.results_summary)
    predictions = build_predictions(preprocessing, results)
    participant_rows, repeatability_rows = summarize_scalars(predictions)
    feature_scan_rows, feature_dimension_rows, feature_summary_rows = summarize_features(
        predictions, args.latent_array, args.latent_index
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "maclaren_neurofm_predictions.csv", predictions)
    write_csv(
        args.output_dir / "maclaren_repeatability_by_participant.csv", participant_rows
    )
    write_csv(args.output_dir / "maclaren_repeatability_summary.csv", repeatability_rows)
    write_csv(
        args.output_dir / "maclaren_feature_stability_by_scan.csv", feature_scan_rows
    )
    write_csv(
        args.output_dir / "maclaren_feature_dimension_icc.csv", feature_dimension_rows
    )
    write_csv(
        args.output_dir / "maclaren_feature_stability_summary.csv", feature_summary_rows
    )

    metadata = {
        "dataset_id": "maclaren_ds000239",
        "release": "R1.0.1",
        "model": "rockNroll87q/NeuroFM neurofm-s",
        "source_commit": "d4e3c463910d939a681d24ebdeb26d44dea6878f",
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "skullstrip": "HD-BET 2.0.1 CPU disable_tta low-memory wrapper",
        "n_predictions": len(predictions),
        "latent_shape": list(np.load(args.latent_array, mmap_mode="r").shape),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "schema_validation": schema,
        "results_summary_sha256": sha256_file(args.results_summary),
        "latent_array_sha256": sha256_file(args.latent_array),
        "latent_index_sha256": sha256_file(args.latent_index),
        "permitted_claim": (
            "B2_test_retest_robustness_conditional_on_three_out_of_range_participants"
        ),
        "critical_interpretation": (
            "Ages 26-31 are below NeuroFM's documented 40-90 range. No age-accuracy, "
            "calibration, biological-age, segmentation, morphometry, diagnostic, or "
            "clinical claim is permitted. Feature stability is technical QC only."
        ),
    }
    (args.output_dir / "maclaren_neurofm_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Maclaren NeuroFM compact outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
