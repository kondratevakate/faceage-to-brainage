"""
Unit tests for src/brain_age.py — no model weights, no GPU, no ANTs required.

Run with:
    pytest tests/test_brain_age_unit.py -v
"""
import sys
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# Make sure `src` is importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.brain_age import (
    _crop_center,
    build_sfcn_age_bins,
    decode_sfcn_output,
    n4_bias_correction,
    predict_synthba_tta,
    reset_synthba_import_state,
    suppress_synthba_import_warnings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(shape=(64, 64, 64), value_range=(0.0, 1000.0), seed=0) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    data = rng.uniform(*value_range, size=shape).astype(np.float32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return nib.Nifti1Image(data, affine)


def _save_tmp_nifti(img: nib.Nifti1Image) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        path = Path(f.name)
    nib.save(img, str(path))
    return path


# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

class TestCropCenter:
    def test_exact_size_is_identity(self):
        data = np.ones((10, 10, 10), dtype=np.float32)
        out = _crop_center(data, (10, 10, 10))
        np.testing.assert_array_equal(out, data)

    def test_crop_larger_volume(self):
        data = np.ones((200, 200, 200), dtype=np.float32)
        target = (160, 192, 160)
        out = _crop_center(data, target)
        assert out.shape == target

    def test_pad_smaller_volume(self):
        data = np.ones((50, 50, 50), dtype=np.float32)
        target = (160, 192, 160)
        out = _crop_center(data, target)
        assert out.shape == target
        # original data should be somewhere in the center; zeros at borders
        assert out[0, 0, 0] == 0.0

    def test_center_is_preserved_on_crop(self):
        # place a peak at center; after crop it should still be present
        data = np.zeros((200, 200, 200), dtype=np.float32)
        data[100, 100, 100] = 99.0
        out = _crop_center(data, (160, 192, 160))
        assert 99.0 in out

    def test_mixed_crop_and_pad(self):
        # axis 0 needs cropping (200 > 160), axis 2 needs padding (50 < 160)
        data = np.ones((200, 192, 50), dtype=np.float32)
        out = _crop_center(data, (160, 192, 160))
        assert out.shape == (160, 192, 160)

    def test_output_dtype_preserved(self):
        data = np.zeros((10, 10, 10), dtype=np.float64)
        out = _crop_center(data, (10, 10, 10))
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# build_sfcn_age_bins
# ---------------------------------------------------------------------------

class TestBuildSfcnAgeBins:
    def test_default_bins(self):
        bins = build_sfcn_age_bins()
        assert bins.shape == (40,)
        assert bins[0] == pytest.approx(42.0)
        assert bins[-1] == pytest.approx(81.0)

    def test_custom_bins(self):
        bins = build_sfcn_age_bins(start=20.0, step=2.0, count=10)
        assert bins.shape == (10,)
        assert bins[0] == pytest.approx(20.0)
        assert bins[1] == pytest.approx(22.0)
        assert bins[-1] == pytest.approx(38.0)

    def test_step_1(self):
        bins = build_sfcn_age_bins(start=0.0, step=1.0, count=5)
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(bins, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# decode_sfcn_output
# ---------------------------------------------------------------------------

class TestDecodeSfcnOutput:
    def _make_logits(self, n_bins=40):
        """Create a torch tensor of logits for testing."""
        import torch
        logits = torch.zeros(1, n_bins)
        return logits

    def test_uniform_logits_predict_midpoint(self):
        import torch
        n = 40
        logits = torch.zeros(1, n)  # softmax → uniform → E[age] = mean(bins)
        bins = build_sfcn_age_bins()
        age = decode_sfcn_output(logits, age_bins=bins)
        assert age == pytest.approx(float(bins.mean()), abs=0.1)

    def test_one_hot_logits_predict_bin_age(self):
        import torch
        n = 40
        logits = torch.full((1, n), -1e9)
        logits[0, 10] = 0.0  # all probability mass on bin 10
        bins = build_sfcn_age_bins()
        age = decode_sfcn_output(logits, age_bins=bins)
        assert age == pytest.approx(float(bins[10]), abs=0.01)

    def test_accepts_list_wrapping(self):
        import torch
        logits = torch.zeros(1, 40)
        age = decode_sfcn_output([logits])
        assert isinstance(age, float)

    def test_wrong_bin_count_raises(self):
        import torch
        logits = torch.zeros(1, 30)  # default bins = 40
        with pytest.raises(ValueError, match="dimension"):
            decode_sfcn_output(logits)


# ---------------------------------------------------------------------------
# n4_bias_correction
# ---------------------------------------------------------------------------

class TestN4BiasCorrection:
    def test_output_file_created(self, tmp_path):
        img = _make_nifti((32, 32, 32))
        in_path = tmp_path / "input.nii.gz"
        out_path = tmp_path / "corrected.nii.gz"
        nib.save(img, str(in_path))

        result = n4_bias_correction(in_path, out_path)

        assert result == out_path
        assert out_path.exists()

    def test_output_same_shape(self, tmp_path):
        img = _make_nifti((32, 32, 32))
        in_path = tmp_path / "input.nii.gz"
        out_path = tmp_path / "corrected.nii.gz"
        nib.save(img, str(in_path))
        n4_bias_correction(in_path, out_path)

        corrected = nib.load(str(out_path))
        assert corrected.shape == img.shape

    def test_output_values_differ_from_input(self, tmp_path):
        # N4 correction changes intensities
        img = _make_nifti((32, 32, 32))
        in_path = tmp_path / "input.nii.gz"
        out_path = tmp_path / "corrected.nii.gz"
        nib.save(img, str(in_path))
        n4_bias_correction(in_path, out_path)

        orig = img.get_fdata()
        corr = nib.load(str(out_path)).get_fdata()
        # Values should not be identical (N4 changes something)
        assert not np.allclose(orig, corr)


# ---------------------------------------------------------------------------
# prepare_sfcn_input (lightweight: no skull strip, no MNI reg)
# ---------------------------------------------------------------------------

class TestPrepareSfcnInput:
    def test_output_shape_and_file_created(self, tmp_path):
        from src.brain_age import prepare_sfcn_input

        img = _make_nifti((180, 200, 175))
        in_path = tmp_path / "raw.nii.gz"
        out_path = tmp_path / "sfcn_input.nii.gz"
        nib.save(img, str(in_path))

        result = prepare_sfcn_input(
            in_path, out_path,
            skullstrip=False,
            n4_correct=False,
            register_mni=False,
        )

        assert result == out_path
        assert out_path.exists()
        prepared = nib.load(str(out_path))
        assert prepared.shape == (160, 192, 160)

    def test_intermediates_cleaned_up(self, tmp_path):
        from src.brain_age import prepare_sfcn_input

        img = _make_nifti((64, 64, 64))
        in_path = tmp_path / "raw.nii.gz"
        out_path = tmp_path / "sfcn_input.nii.gz"
        nib.save(img, str(in_path))

        prepare_sfcn_input(
            in_path, out_path,
            skullstrip=False,
            n4_correct=True,
            register_mni=False,
        )

        # n4 intermediate should be deleted
        n4_path = out_path.with_name("raw_n4.nii.gz")
        assert not n4_path.exists(), "Intermediate N4 file should be cleaned up"


# ---------------------------------------------------------------------------
# predict_synthba_tta (mocked to avoid antspyx dependency)
# ---------------------------------------------------------------------------

class TestPredicSynthbaTta:
    def test_tta_returns_correct_schema(self, monkeypatch):
        """Mock predict_synthba to test the TTA aggregation logic."""
        import src.brain_age as ba

        call_count = {"n": 0}
        def mock_predict(path, device="cpu", mr_weighting="t1"):
            call_count["n"] += 1
            return 55.0 + call_count["n"]  # 56.0 then 57.0

        monkeypatch.setattr(ba, "predict_synthba", mock_predict)

        img = _make_nifti((32, 32, 32))
        path = _save_tmp_nifti(img)
        try:
            result = predict_synthba_tta(path, device="cpu")
        finally:
            path.unlink(missing_ok=True)

        assert result["n_aug"] == 2
        assert result["mean"] == pytest.approx(56.5)
        assert result["std"] == pytest.approx(0.5)
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# SynthBA import helpers
# ---------------------------------------------------------------------------

class TestSynthbaImportHelpers:
    def test_suppress_synthba_import_warnings_ignores_known_futurewarning(self):
        suppress_synthba_import_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn(
                "The cuda.cudart module is deprecated and will be removed in a future release.",
                FutureWarning,
            )
        assert caught == []

    def test_reset_synthba_import_state_clears_known_roots(self):
        sentinel = object()
        sys.modules["synthba"] = sentinel
        sys.modules["monai.transforms"] = sentinel
        sys.modules["cuda.cudart"] = sentinel
        sys.modules["skimage.segmentation"] = sentinel
        sys.modules["scipy.cluster"] = sentinel
        sys.modules["unrelated.module"] = sentinel

        cleared = reset_synthba_import_state()

        assert cleared >= 5
        assert "synthba" not in sys.modules
        assert "monai.transforms" not in sys.modules
        assert "cuda.cudart" not in sys.modules
        assert "skimage.segmentation" not in sys.modules
        assert "scipy.cluster" not in sys.modules
        assert sys.modules["unrelated.module"] is sentinel
        sys.modules.pop("unrelated.module", None)
