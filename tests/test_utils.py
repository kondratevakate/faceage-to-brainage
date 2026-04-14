"""
Unit tests for src/utils.py.

Run with:
    pytest tests/test_utils.py -v
"""
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import conform_1mm, load_vol, reorient_to_ras


def _make_nifti(shape=(32, 32, 32), voxel_size=1.0) -> nib.Nifti1Image:
    data = np.random.default_rng(42).uniform(0, 1000, size=shape).astype(np.float32)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    return nib.Nifti1Image(data, affine)


class TestLoadVol:
    def test_loads_nifti_gz(self, tmp_path):
        img = _make_nifti()
        p = tmp_path / "test.nii.gz"
        nib.save(img, str(p))
        loaded = load_vol(p)
        assert isinstance(loaded, nib.Nifti1Image)
        assert loaded.shape == img.shape

    def test_returns_nifti1image_type(self, tmp_path):
        img = _make_nifti()
        p = tmp_path / "test.nii.gz"
        nib.save(img, str(p))
        loaded = load_vol(str(p))  # also accept str
        assert isinstance(loaded, nib.Nifti1Image)


class TestReorientToRas:
    def test_already_ras_unchanged(self):
        # Identity affine is RAS
        img = _make_nifti()
        out = reorient_to_ras(img)
        assert out.shape == img.shape

    def test_non_ras_gets_reoriented(self):
        # LAS affine (flip x)
        data = np.ones((20, 20, 20), dtype=np.float32)
        affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        out = reorient_to_ras(img)
        # After reorientation, voxel count stays the same
        assert np.prod(out.shape[:3]) == np.prod(img.shape[:3])


class TestConform1mm:
    def test_output_is_1mm_isotropic(self):
        # Input at 2mm; output should be 1mm
        img = _make_nifti(shape=(64, 64, 64), voxel_size=2.0)
        out = conform_1mm(img)
        zooms = out.header.get_zooms()[:3]
        np.testing.assert_allclose(zooms, [1.0, 1.0, 1.0], atol=1e-3)

    def test_output_shape_is_256_cube(self):
        img = _make_nifti(shape=(64, 64, 64), voxel_size=2.0)
        out = conform_1mm(img)
        assert out.shape == (256, 256, 256)
