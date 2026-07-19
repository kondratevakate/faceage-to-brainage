"""Unit tests for the locked Maclaren repeatability estimators."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "brainage_maclaren"
    / "summarize_maclaren_neurofm.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_maclaren_neurofm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PERTURBATION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "brainage_maclaren"
    / "prepare_maclaren_perturbations.py"
)
PERTURBATION_SPEC = importlib.util.spec_from_file_location(
    "prepare_maclaren_perturbations", PERTURBATION_SCRIPT
)
assert PERTURBATION_SPEC is not None and PERTURBATION_SPEC.loader is not None
PERTURBATION_MODULE = importlib.util.module_from_spec(PERTURBATION_SPEC)
PERTURBATION_SPEC.loader.exec_module(PERTURBATION_MODULE)

PERTURBATION_SUMMARY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "brainage_maclaren"
    / "summarize_maclaren_perturbations.py"
)
PERTURBATION_SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_maclaren_perturbations", PERTURBATION_SUMMARY_SCRIPT
)
assert (
    PERTURBATION_SUMMARY_SPEC is not None
    and PERTURBATION_SUMMARY_SPEC.loader is not None
)
PERTURBATION_SUMMARY_MODULE = importlib.util.module_from_spec(
    PERTURBATION_SUMMARY_SPEC
)
PERTURBATION_SUMMARY_SPEC.loader.exec_module(PERTURBATION_SUMMARY_MODULE)


class RepeatabilityEstimatorTests(unittest.TestCase):
    def test_perfect_repeated_agreement_has_icc_one_and_zero_within_sd(self):
        matrix = np.asarray([[10.0] * 40, [20.0] * 40, [30.0] * 40])
        self.assertAlmostEqual(MODULE.pooled_within_sd(matrix), 0.0)
        self.assertAlmostEqual(MODULE.icc(matrix, "absolute"), 1.0)
        self.assertAlmostEqual(MODULE.icc(matrix, "consistency"), 1.0)

    def test_common_session_shift_reduces_absolute_but_not_consistency_icc(self):
        subject = np.asarray([10.0, 20.0, 30.0])[:, None]
        session = np.linspace(-3.0, 3.0, 40)[None, :]
        matrix = subject + session
        self.assertAlmostEqual(MODULE.icc(matrix, "consistency"), 1.0)
        self.assertLess(MODULE.icc(matrix, "absolute"), 1.0)

    def test_pooled_within_sd_uses_residual_degrees_of_freedom(self):
        matrix = np.asarray([[0.0, 2.0], [10.0, 12.0], [20.0, 22.0]])
        self.assertAlmostEqual(MODULE.pooled_within_sd(matrix), np.sqrt(2.0))

    def test_pairwise_differences_stay_within_subject(self):
        matrix = np.asarray([[0.0, 1.0, 3.0], [10.0, 12.0, 15.0]])
        values = MODULE.pairwise_absolute_differences(matrix)
        self.assertEqual(values.size, 6)
        np.testing.assert_allclose(
            np.sort(values), [1.0, 2.0, 2.0, 3.0, 3.0, 5.0]
        )

    def test_interpolated_perturbations_preserve_float_data(self):
        header = nib.Nifti1Header()
        header.set_data_dtype(np.int16)
        image = nib.Nifti1Image(
            np.arange(9**3, dtype=np.int16).reshape((9, 9, 9)),
            np.eye(4),
            header,
        )
        specs = [
            {"family": "rotation", "axis": "0", "level": "1"},
            {"family": "scale", "axis": "isotropic", "level": "1.05"},
            {"family": "resolution", "axis": "isotropic", "level": "1.2"},
        ]
        for spec in specs:
            with self.subTest(family=spec["family"]):
                result = PERTURBATION_MODULE.create_perturbation(image, spec)
                self.assertEqual(result.get_data_dtype(), np.dtype(np.float32))
                self.assertTrue(np.isfinite(result.get_fdata()).all())

    def test_failed_perturbations_remain_in_summary_denominator(self):
        failed_rows = [
            {
                "participant_id": f"sub-0{index}",
                "status": "failed",
                "sex_class_flip": "",
            }
            for index in range(1, 4)
        ]
        result = PERTURBATION_SUMMARY_MODULE.summarize_group(
            failed_rows, "perturbation", "resolution_1mm"
        )
        self.assertEqual(result["n_attempted"], 3)
        self.assertEqual(result["n_successful"], 0)
        self.assertEqual(result["failure_rate"], "1")
        self.assertEqual(
            result["fraction_attempted_age_delta_within_2_year_margin"], "0"
        )
        self.assertEqual(result["mean_abs_brain_age_delta_years"], "")


if __name__ == "__main__":
    unittest.main()
