"""
Unit tests for MIDIBrainAge compatibility patching.

These tests lock down the real Colab failures we hit while validating MIDI on
SIMON:
- MONAI removed ``AddChannel`` and changed ``Spacing`` usage
- HD-BET no longer accepts numeric GPU device strings like ``0``
- reordered numpy arrays can carry negative strides into Torch/MONAI

Run with:
    pytest tests/test_midi_compat.py -v
"""
import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIDI_DIR = REPO_ROOT / "vendor" / "MIDIBrainAge"
PRE_PROCESS = MIDI_DIR / "pre_process.py"

sys.path.insert(0, str(REPO_ROOT))

from src.brain_age import (
    _patch_midi_preprocess_file,
    _patch_midi_preprocess_source,
    predict_midi_brainage,
)


LEGACY_PRE_PROCESS = dedent(
    """\
    import numpy as np
    import nibabel as nib
    import monai
    from monai.transforms import (
        AddChannel,
        Spacing,
        ResizeWithPadOrCrop
    )
    import os

    def preprocess(input_path, use_gpu, skull_strip, register, project_name):
        if skull_strip:
            orig_nii = nib.load(input_path)
            orig_arr, orig_affine = np.asarray(orig_nii.dataobj), orig_nii.affine
            reoriented_arr, reoriented_affine, *_ = reorder_voxels(orig_arr, orig_affine, 'RAS')
            new_image = nib.Nifti1Image(reoriented_arr, reoriented_affine)
            nib.save(new_image, "./tmp/reorient.nii.gz")
            if use_gpu:
                cmd = 'hd-bet -i {} -o {} -mode fast'.format(reoriented_path, stripped_path)
            else:
                cmd = 'hd-bet -i {} -o {} -mode fast -device cpu'.format(reoriented_path, stripped_path)
            os.system(cmd)

        orig_nii = nib.load(input_path)
        orig_arr, orig_affine = np.asarray(orig_nii.dataobj), orig_nii.affine
        reoriented_arr, reoriented_affine, *_ = reorder_voxels(orig_arr, orig_affine, 'RAS')
        reoriented_arr = AddChannel()(reoriented_arr)
        resampled_arr =  Spacing(pixdim=(1.4, 1.4, 1.4), mode='bilinear')(reoriented_arr, reoriented_affine)[0]
        return resampled_arr
    """
)


def _assert_no_addchannel_in_monai_import_block(src: str) -> None:
    in_block = False
    for line in src.splitlines():
        if line.strip().startswith("from monai.transforms import ("):
            in_block = True
        if in_block:
            assert "AddChannel" not in line, (
                f"AddChannel still present in monai import block: {line!r}"
            )
        if in_block and line.strip() == ")":
            in_block = False


class TestPatchMidiPreprocessSource:
    def test_legacy_source_is_rewritten_for_modern_colab(self):
        patched = _patch_midi_preprocess_source(LEGACY_PRE_PROCESS)

        ast.parse(patched)
        assert "class AddChannel" in patched
        assert patched.count("class AddChannel") == 1
        assert "from monai.data import MetaTensor" in patched
        assert "-mode fast" not in patched
        assert "-device 0" not in patched
        assert "-device cuda" in patched
        assert "-device cpu" in patched
        assert "np.ascontiguousarray(reoriented_arr)" in patched
        assert patched.count("np.ascontiguousarray(reoriented_arr)") == 2
        assert "Spacing(pixdim=(1.4, 1.4, 1.4), mode='bilinear')(reoriented_arr, reoriented_affine)[0]" not in patched
        assert "meta_img = MetaTensor(reoriented_arr, affine=reoriented_affine)" in patched
        assert "resampled_arr = Spacing(pixdim=(1.4, 1.4, 1.4), mode='bilinear')(meta_img)" in patched
        assert "resampled_arr = np.asarray(resampled_arr)" in patched
        _assert_no_addchannel_in_monai_import_block(patched)

    def test_patch_is_idempotent(self):
        patched_once = _patch_midi_preprocess_source(LEGACY_PRE_PROCESS)
        patched_twice = _patch_midi_preprocess_source(patched_once)
        assert patched_twice == patched_once

    def test_current_vendor_source_patches_cleanly(self):
        if not PRE_PROCESS.exists():
            pytest.skip("vendor/MIDIBrainAge not present")

        patched = _patch_midi_preprocess_source(PRE_PROCESS.read_text(encoding="utf-8"))
        ast.parse(patched)
        assert "-device 0" not in patched
        assert "-mode fast" not in patched
        assert "np.ascontiguousarray(reoriented_arr)" in patched
        assert "Spacing(pixdim=(1.4, 1.4, 1.4), mode='bilinear')(reoriented_arr, reoriented_affine)[0]" not in patched
        assert "resampled_arr = np.asarray(resampled_arr)" in patched


class TestPatchMidiPreprocessFile:
    def test_patch_file_is_in_place_and_idempotent(self, tmp_path):
        pp = tmp_path / "pre_process.py"
        pp.write_text(LEGACY_PRE_PROCESS, encoding="utf-8")

        changed = _patch_midi_preprocess_file(pp)
        first = pp.read_text(encoding="utf-8")
        changed_again = _patch_midi_preprocess_file(pp)
        second = pp.read_text(encoding="utf-8")

        assert changed is True
        assert changed_again is False
        assert first == second
        ast.parse(second)


class TestPredictMidiBrainagePatching:
    def test_predict_midi_brainage_patches_preprocess_before_subprocess(self, tmp_path, monkeypatch):
        midi_dir = tmp_path / "MIDIBrainAge"
        midi_dir.mkdir()
        pp = midi_dir / "pre_process.py"
        pp.write_text(LEGACY_PRE_PROCESS, encoding="utf-8")
        (midi_dir / "run_inference.py").write_text("# dummy\n", encoding="utf-8")

        nifti_path = tmp_path / "scan.nii.gz"
        nifti_path.write_bytes(b"")

        def fake_run(cmd, capture_output, text, cwd, timeout):
            project_name = cmd[cmd.index("--project_name") + 1]
            out_dir = Path(cwd) / project_name
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"ID": "subj", "Predicted_age (years)": 44.2}]
            ).to_csv(out_dir / "brain_age_output.csv", index=False)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr("src.brain_age.subprocess.run", fake_run)

        pred = predict_midi_brainage(
            nifti_path=nifti_path,
            midi_dir=midi_dir,
            device="cuda",
            sequence="t1",
            skull_strip=True,
        )

        patched = pp.read_text(encoding="utf-8")
        assert pred == pytest.approx(44.2)
        assert "-device cuda" in patched
        assert "-device 0" not in patched
        assert "resampled_arr = np.asarray(resampled_arr)" in patched
        assert "np.ascontiguousarray(reoriented_arr)" in patched
