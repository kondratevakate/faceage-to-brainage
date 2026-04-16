"""
Drop-in replacement for bioface3d/mvcnn/predict.py that adds:
  - MPS (Apple Silicon) device support via monkey-patching DeepMVLM._prepare_device
  - Automatic fix for hardcoded Windows model-weight paths in config JSONs

Called by src/landmarks.py as a subprocess. All standard predict.py arguments
(-c, -n, -o, -of, -pn, etc.) are forwarded unchanged; --device is the only addition.

Usage (same as predict.py, plus --device):
    python _mvcnn_driver.py -c <config.json> -n <mesh.ply> -o <out_dir> \
                            -of txt -pn 5 --device cpu|mps|cuda
"""
import argparse
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Resolve bioface3d/mvcnn/ and add to sys.path ─────────────────────────────
_SRC_DIR        = Path(__file__).resolve().parent
_BIOFACE3D_ROOT = _SRC_DIR.parents[1] / "bioface3d"
_MVCNN_DIR      = _BIOFACE3D_ROOT / "mvcnn"

if str(_MVCNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MVCNN_DIR))

# ── Python 3.12 compatibility: shim the removed `imp` module ─────────────────
# bioface3d/mvcnn/utils/__init__.py imports `imp` but does not use it.
# `imp` was removed in Python 3.12; inject a minimal stub so the import succeeds.
if "imp" not in sys.modules:
    import importlib
    import importlib.util
    import types as _types

    _imp_stub = _types.ModuleType("imp")
    _imp_stub.load_source       = lambda name, path: importlib.import_module(name)  # type: ignore[attr-defined]
    _imp_stub.find_module       = importlib.util.find_spec                           # type: ignore[attr-defined]
    _imp_stub.load_module       = lambda name: sys.modules.get(name)                # type: ignore[attr-defined]
    _imp_stub.new_module        = lambda name: _types.ModuleType(name)              # type: ignore[attr-defined]
    _imp_stub.PY_SOURCE         = 1                                                  # type: ignore[attr-defined]
    _imp_stub.PY_COMPILED       = 2                                                  # type: ignore[attr-defined]
    _imp_stub.C_EXTENSION       = 3                                                  # type: ignore[attr-defined]
    sys.modules["imp"] = _imp_stub

# ── Shim torch.utils.tensorboard.SummaryWriter (training-only, not needed for inference)
# torch/utils/tensorboard/__init__.py raises ImportError when tensorboard is absent.
# Replace the whole submodule with a no-op stub before anything imports it.
import types as _types2

_tb_mod = _types2.ModuleType("torch.utils.tensorboard")


class _FakeSummaryWriter:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, name): return lambda *a, **kw: None


_tb_mod.SummaryWriter = _FakeSummaryWriter  # type: ignore[attr-defined]
sys.modules["torch.utils.tensorboard"] = _tb_mod

# ── PyTorch 2.6 compatibility: weights_only default changed to True ───────────
# The BioFace3D checkpoint embeds ConfigParser objects; patch torch.load so it
# keeps weights_only=False (safe: we control the checkpoint source).
import torch as _torch

_orig_torch_load = _torch.load


def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)


_torch.load = _patched_torch_load


def _patch_config_model_path(cfg_path: str) -> str:
    """
    Load the MVCNN config JSON, replace the hardcoded Windows model path with
    the actual local model_best.pth sitting next to the config file, and write
    the patched version to a temporary file.  Returns the temp file path.
    """
    cfg_path = Path(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # The key may be 'model_pth_or_url' (older configs) or 'model' depending on version
    pred = cfg.get("predict", {})
    for key in ("model_pth_or_url", "model"):
        if key in pred:
            local_pth = cfg_path.parent / "model_best.pth"
            if not local_pth.exists():
                raise FileNotFoundError(
                    f"model_best.pth not found at {local_pth}. "
                    "Download the BioFace3D-20 weights before running."
                )
            pred[key] = str(local_pth)
            break

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(cfg_path.parent)
    )
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _apply_device_patch(device: str) -> None:
    """
    Monkey-patch DeepMVLM._prepare_device before the class is used so that
    'mps' and 'cpu' are honoured without touching BioFace3D source files.
    """
    import torch
    import deepmvlm

    if device == "mps":
        if not torch.backends.mps.is_available():
            print("[_mvcnn_driver] MPS not available, falling back to CPU.")
            device = "cpu"
        else:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    chosen = torch.device(device if device != "cuda" else "cuda:0")

    def _patched_prepare_device(self, n_gpu_use):  # noqa: ANN001
        return chosen, []

    deepmvlm.DeepMVLM._prepare_device = _patched_prepare_device


def main() -> None:
    # ── Parse our extra --device flag, pass everything else to ConfigParser ───
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    known, remaining = pre.parse_known_args()

    _apply_device_patch(known.device)

    # ── Patch the config to fix the hardcoded Windows model path ─────────────
    # Find -c / --config in remaining args and replace its value
    tmp_cfg_path = None
    for i, arg in enumerate(remaining):
        if arg in ("-c", "--config") and i + 1 < len(remaining):
            tmp_cfg_path = _patch_config_model_path(remaining[i + 1])
            remaining[i + 1] = tmp_cfg_path
            break

    # ── Hand off to the original predict.py main() ───────────────────────────
    # Re-inject remaining args so ConfigParser sees them via sys.argv
    sys.argv = [sys.argv[0]] + remaining

    from parse_config import ConfigParser
    from map import mapConfig as m
    import predict as _predict_module

    args = argparse.ArgumentParser(description="Deep-MVLM (driver)")
    args.add_argument("-c", "--config", default=None, type=str)
    args.add_argument("-d", "--device", default=None, type=str)
    args.add_argument("-n", "--name", default=None, type=str)
    args.add_argument("-pn", "--predict_num", default=None, type=str)
    args.add_argument("-pt", "--predict_tries", default=None, type=str)
    args.add_argument("-mr", "--max_ransac", default=None, type=str)
    args.add_argument("-rp", "--render_predict", default=None, type=str)
    args.add_argument("-si", "--save_img", default=None, type=str)
    args.add_argument("-o", "--output_path", default=None, type=str)
    args.add_argument("-of", "--output_format", default=None, type=str)
    args.add_argument("-ms", "--metadata_save", default=None, type=str)

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [CustomArgs(["-ng", "--n_gpu"], type=int, target=[m.ngpu])]

    try:
        config = ConfigParser(args, options)
        _predict_module.main(config)
    finally:
        if tmp_cfg_path and Path(tmp_cfg_path).exists():
            Path(tmp_cfg_path).unlink()


if __name__ == "__main__":
    main()
