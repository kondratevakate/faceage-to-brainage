# Cloud Runbook: Single-Shot Avatar Baselines

Purpose: run heavier one-shot avatar methods without changing the local project
layout or uploading private face photos to public demos.

## Decision

Use the same portable bundle everywhere:

```powershell
.\.venv\Scripts\python.exe scripts\photo_mri_avatar\prepare_cloud_avatar_bundle.py
```

Default bundle scope is privacy-minimal:

- only primary case crops with prefix `1_1`;
- no internal controls;
- no MRI surface files;
- output path: `data/avatar_2026_work/cloud_bundles/`.

Add internal controls or MRI only deliberately:

```powershell
.\.venv\Scripts\python.exe scripts\photo_mri_avatar\prepare_cloud_avatar_bundle.py --include-internal-controls
.\.venv\Scripts\python.exe scripts\photo_mri_avatar\prepare_cloud_avatar_bundle.py --include-mri-surface
```

## Lenovo

Install PyTorch locally only for CPU tests or if the laptop has an NVIDIA GPU.
The current preflight did not find `nvidia-smi`, so assume CPU-only until proven
otherwise.

CPU smoke test:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe scripts\photo_mri_avatar\sota_avatar_preflight.py
```

CUDA only makes sense if `nvidia-smi` works first. Use the current official
PyTorch selector for the exact install command.

## Google Colab

Good for a quick smoke test if GPU is available. Use a private notebook, upload
the case bundle, unzip it, clone the target repo, run inference, zip outputs.

Minimal notebook cells:

```python
!nvidia-smi
!python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

```python
from google.colab import files
uploaded = files.upload()  # upload avatar_case_1_1_*.zip
```

```python
import zipfile, pathlib
bundle = next(pathlib.Path(".").glob("avatar_case_1_1_*.zip"))
work = pathlib.Path("/content/avatar_case")
work.mkdir(exist_ok=True)
zipfile.ZipFile(bundle).extractall(work)
print(list((work / "inputs" / "crops").glob("*.jpg")))
```

Use Colab only for code you are comfortable running with private uploaded
photos. Avoid public hosted demos for private faces.

## AWS

Best path when credits are available:

1. Launch an EC2 GPU instance from an AWS Deep Learning AMI with PyTorch.
2. Use a small GPU first for setup smoke tests; move up only if LAM/GAGAvatar
   needs more VRAM.
3. Upload the bundle via `scp` or S3 private bucket.
4. Clone the selected external repo.
5. Run inference into `outputs/<method>/`.
6. Zip outputs and copy them back to `data/avatar_2026_work/`.
7. Stop or terminate the instance immediately.

AWS smoke test:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Method Order

1. **DECA/MICA/EMOCA** for geometry-first MRI comparison once FLAME/checkpoints
   are available.
2. **LAM/GAGAvatar** for perceptual Gaussian avatar quality on CUDA.
3. **MeshLAM** when a separate runnable code/weights release is available.

## Return Contract

Cloud outputs should return as:

```text
outputs/
  deca/
  mica/
  emoca/
  lam/
  gagavatar/
```

Copy them locally under:

```text
data/avatar_2026_work/photo_avatar_<method>/
```

Then rerun local evaluation scripts against MRI masks and surface metrics.

## References

- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- Google Colab FAQ: <https://research.google.com/colaboratory/faq.html>
- AWS Deep Learning AMIs: <https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html>
- LAM: <https://github.com/aigc3d/LAM>
- GAGAvatar: <https://github.com/xg-chu/GAGAvatar>
