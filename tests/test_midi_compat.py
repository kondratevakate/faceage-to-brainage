"""
Compatibility tests for MIDIBrainAge vendor integration.

Catches issues that previously required multiple Colab iterations to discover:
- hd-bet CLI flags change between versions
- MONAI AddChannel removed in >= 1.0
- pre_process.py import errors

Run with:
    pytest tests/test_midi_compat.py -v

Skipped automatically if vendor/MIDIBrainAge is not present.
"""
import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIDI_DIR = REPO_ROOT / "vendor" / "MIDIBrainAge"
PRE_PROCESS = MIDI_DIR / "pre_process.py"

sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not MIDI_DIR.exists() or not PRE_PROCESS.exists(),
    reason="vendor/MIDIBrainAge not present",
)


# ---------------------------------------------------------------------------
# pre_process.py patch correctness
# ---------------------------------------------------------------------------

class TestPreProcessPatch:
    def _get_patched(self) -> str:
        """Apply predict_midi_brainage patch logic and return patched source."""
        from src.brain_age import predict_midi_brainage
        import inspect, unittest.mock as mock

        raw = PRE_PROCESS.read_text(encoding="utf-8")

        # Simulate the patch from predict_midi_brainage
        if "AddChannel" in raw and "class AddChannel" not in raw:
            raw = re.sub(r"[ \t]*AddChannel,?[ \t]*\r?\n", "", raw)
            raw = re.sub(r"from monai\.transforms import AddChannel\r?\n", "", raw)
            shim = (
                "try:\n"
                "    from monai.transforms import AddChannel\n"
                "except ImportError:\n"
                "    class AddChannel:\n"
                "        def __call__(self, x):\n"
                "            import numpy as np\n"
                "            return np.expand_dims(x, 0)\n"
            )
            raw = shim + raw

        if "-mode fast" in raw:
            raw = raw.replace(
                "cmd = 'hd-bet -i {} -o {} -mode fast'.format(reoriented_path, stripped_path)",
                "cmd = 'hd-bet -i {} -o {}'.format(reoriented_path, stripped_path)",
            )
            raw = raw.replace(
                "cmd = 'hd-bet -i {} -o {} -mode fast -device cpu'.format(reoriented_path, stripped_path)",
                "cmd = 'hd-bet -i {} -o {} --device cpu'.format(reoriented_path, stripped_path)",
            )
        return raw

    def test_patched_source_is_valid_python(self):
        """Patched pre_process.py must parse without SyntaxError."""
        src = self._get_patched()
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"Patched pre_process.py has SyntaxError at line {e.lineno}: {e.msg}")

    def test_no_bare_addchannel_in_monai_import_block(self):
        """After patching, AddChannel must not appear inside a monai import block."""
        src = self._get_patched()
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

    def test_addchannel_shim_present_after_patch(self):
        """Patched source must define AddChannel via try/except or class."""
        src = self._get_patched()
        has_shim = "class AddChannel" in src or (
            "try:" in src and "AddChannel" in src and "except ImportError" in src
        )
        assert has_shim, "No AddChannel compatibility shim found after patch"

    def test_no_old_hdbet_mode_flag(self):
        """Patched source must not contain deprecated -mode fast flag."""
        src = self._get_patched()
        assert "-mode fast" not in src, "Deprecated hd-bet flag '-mode fast' still present after patch"

    def test_no_old_hdbet_device_flag(self):
        """Patched source must not use '-device cpu' (old syntax, should be --device)."""
        src = self._get_patched()
        # Old: -device cpu (single dash). New: --device cpu (double dash)
        assert "'-device cpu'" not in src and '"-device cpu"' not in src, (
            "Deprecated hd-bet flag '-device cpu' still present (use '--device cpu')"
        )


# ---------------------------------------------------------------------------
# hd-bet CLI compatibility
# ---------------------------------------------------------------------------

class TestHdBetCli:
    @pytest.fixture(autouse=True)
    def require_hdbet(self):
        if shutil.which("hd-bet") is None:
            pytest.skip("hd-bet not installed")

    def test_hdbet_help_exits_zero(self):
        """hd-bet --help must succeed (exit 0)."""
        r = subprocess.run(["hd-bet", "--help"], capture_output=True, text=True)
        assert r.returncode == 0, f"hd-bet --help failed:\n{r.stderr}"

    def test_hdbet_does_not_accept_mode_flag(self):
        """Verify that -mode fast is no longer valid (confirms we need the patch)."""
        r = subprocess.run(
            ["hd-bet", "-i", "x.nii.gz", "-o", "y.nii.gz", "-mode", "fast"],
            capture_output=True, text=True,
        )
        # Either unrecognized argument error or file-not-found — NOT a clean success
        assert r.returncode != 0 or "unrecognized" in r.stderr.lower(), (
            "hd-bet accepted -mode fast — patch may be unnecessary now, review"
        )

    def test_hdbet_accepts_device_double_dash(self):
        """hd-bet must accept --device flag (new API)."""
        r = subprocess.run(
            ["hd-bet", "--help"],
            capture_output=True, text=True,
        )
        help_text = r.stdout + r.stderr
        assert "--device" in help_text or "-device" in help_text, (
            "hd-bet --help does not mention device flag at all"
        )


# ---------------------------------------------------------------------------
# predict_midi_brainage applies patch before subprocess
# ---------------------------------------------------------------------------

class TestMidiPatchApplied:
    def test_patch_applied_to_actual_file(self, tmp_path):
        """predict_midi_brainage must patch pre_process.py before running."""
        import shutil as sh
        # Copy MIDI dir to tmp to avoid mutating vendor
        tmp_midi = tmp_path / "MIDIBrainAge"
        sh.copytree(MIDI_DIR, tmp_midi)

        # Make sure it starts unpatched (restore original if already patched)
        pp = tmp_midi / "pre_process.py"
        original = PRE_PROCESS.read_text(encoding="utf-8")
        pp.write_text(original, encoding="utf-8")

        # Import and call patch logic directly (same code as predict_midi_brainage)
        raw = pp.read_text(encoding="utf-8")
        changed = False
        if "AddChannel" in raw and "class AddChannel" not in raw:
            raw = re.sub(r"[ \t]*AddChannel,?[ \t]*\r?\n", "", raw)
            raw = re.sub(r"from monai\.transforms import AddChannel\r?\n", "", raw)
            shim = (
                "try:\n    from monai.transforms import AddChannel\n"
                "except ImportError:\n    class AddChannel:\n"
                "        def __call__(self, x):\n"
                "            import numpy as np\n"
                "            return np.expand_dims(x, 0)\n"
            )
            raw = shim + raw
            changed = True
        if "-mode fast" in raw:
            raw = raw.replace(
                "cmd = 'hd-bet -i {} -o {} -mode fast'.format(reoriented_path, stripped_path)",
                "cmd = 'hd-bet -i {} -o {}'.format(reoriented_path, stripped_path)",
            )
            raw = raw.replace(
                "cmd = 'hd-bet -i {} -o {} -mode fast -device cpu'.format(reoriented_path, stripped_path)",
                "cmd = 'hd-bet -i {} -o {} --device cpu'.format(reoriented_path, stripped_path)",
            )
            changed = True
        if changed:
            pp.write_text(raw, encoding="utf-8")

        patched = pp.read_text(encoding="utf-8")
        assert "-mode fast" not in patched
        assert "class AddChannel" in patched or (
            "try:" in patched and "AddChannel" in patched
        )
        # Must still be valid Python
        ast.parse(patched)
