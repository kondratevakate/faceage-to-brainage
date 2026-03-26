"""
FaceAge inference wrapper.

Supports two modes:
  - MTCNN mode  : standard FaceAge pipeline (face detection → crop → age)
  - Bypass mode : skips MTCNN, crops the image manually and feeds directly
                  into the Inception-ResNet-v1 age regressor.
                  Use this when MTCNN fails on MRI renders.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Expected input size for Inception-ResNet-v1 used in FaceAge
_FACE_SIZE = 160


def _load_model(faceage_root: str | Path) -> torch.nn.Module:
    """Load the pretrained Inception-ResNet-v1 age regressor from FaceAge."""
    import sys

    faceage_root = Path(faceage_root)
    if str(faceage_root) not in sys.path:
        sys.path.insert(0, str(faceage_root))

    from models.inception_resnet_v1 import InceptionResnetV1  # noqa: PLC0415

    model = InceptionResnetV1(pretrained=None, classify=False)
    weights_path = faceage_root / "models" / "FaceAge_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"FaceAge weights not found at {weights_path}.\n"
            "Download from the FaceAge Google Drive link in vendor/FaceAge/README.md"
        )
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _preprocess(img: Image.Image) -> torch.Tensor:
    """Resize to 160×160, convert to float tensor in [-1, 1]."""
    img = img.convert("RGB").resize((_FACE_SIZE, _FACE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def predict_age(
    img_path: str | Path,
    faceage_root: str | Path,
    bypass_mtcnn: bool = False,
    device: str = "cpu",
) -> dict:
    """
    Predict apparent age from a face image.

    Parameters
    ----------
    img_path     : path to the input image (PNG / JPEG)
    faceage_root : path to the cloned FaceAge repository
    bypass_mtcnn : if True, skip MTCNN and use the full image as face crop
    device       : 'cpu' or 'cuda'

    Returns
    -------
    dict with keys:
        'age'          : float predicted age (years)
        'embedding'    : np.ndarray shape (512,) face embedding
        'mtcnn_found'  : bool — whether MTCNN detected a face
        'bypass_used'  : bool — whether bypass mode was used
    """
    img_path = Path(img_path)
    model = _load_model(faceage_root)
    model = model.to(device)

    img = Image.open(str(img_path)).convert("RGB")
    mtcnn_found = False
    bypass_used = bypass_mtcnn

    if not bypass_mtcnn:
        try:
            from facenet_pytorch import MTCNN  # noqa: PLC0415

            mtcnn = MTCNN(image_size=_FACE_SIZE, device=device, keep_all=False)
            face_tensor = mtcnn(img)  # None if no face found
            if face_tensor is not None:
                mtcnn_found = True
                face_input = face_tensor.unsqueeze(0).to(device)
            else:
                bypass_used = True
        except Exception:
            bypass_used = True

    if bypass_used or not mtcnn_found:
        face_input = _preprocess(img).to(device)

    with torch.no_grad():
        embedding = model(face_input)  # (1, 512)

    embedding_np = embedding.squeeze(0).cpu().numpy()

    # Age regressor head: linear layer on top of embedding
    # FaceAge stores it separately; if available use it, else use L2 norm heuristic
    age_head_path = Path(faceage_root) / "models" / "age_regressor.pt"
    if age_head_path.exists():
        head_state = torch.load(str(age_head_path), map_location="cpu")
        # Expect a single linear layer: weight (1, 512) + bias (1,)
        w = head_state["weight"].to(device)
        b = head_state["bias"].to(device)
        age = (embedding @ w.T + b).item()
    else:
        # Fallback: cosine similarity with age anchor approach not available.
        # Return NaN so the caller knows the head is missing.
        age = float("nan")

    return {
        "age": age,
        "embedding": embedding_np,
        "mtcnn_found": mtcnn_found,
        "bypass_used": bypass_used,
    }


def predict_age_batch(
    img_paths: list,
    faceage_root: str | Path,
    bypass_mtcnn: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """Run predict_age for a list of image paths, sharing the loaded model."""
    faceage_root = Path(faceage_root)
    model = _load_model(faceage_root).to(device)

    age_head_path = faceage_root / "models" / "age_regressor.pt"
    w, b = None, None
    if age_head_path.exists():
        head_state = torch.load(str(age_head_path), map_location="cpu")
        w = head_state["weight"].to(device)
        b = head_state["bias"].to(device)

    if not bypass_mtcnn:
        from facenet_pytorch import MTCNN

        mtcnn = MTCNN(image_size=_FACE_SIZE, device=device, keep_all=False)
    else:
        mtcnn = None

    results = []
    for p in img_paths:
        img = Image.open(str(p)).convert("RGB")
        mtcnn_found = False
        bypass_used = bypass_mtcnn

        if mtcnn is not None:
            try:
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    mtcnn_found = True
                    face_input = face_tensor.unsqueeze(0).to(device)
                else:
                    bypass_used = True
            except Exception:
                bypass_used = True

        if bypass_used or not mtcnn_found:
            face_input = _preprocess(img).to(device)

        with torch.no_grad():
            embedding = model(face_input)

        emb_np = embedding.squeeze(0).cpu().numpy()
        age = (embedding @ w.T + b).item() if w is not None else float("nan")

        results.append(
            {
                "path": str(p),
                "age": age,
                "embedding": emb_np,
                "mtcnn_found": mtcnn_found,
                "bypass_used": bypass_used,
            }
        )
    return results
