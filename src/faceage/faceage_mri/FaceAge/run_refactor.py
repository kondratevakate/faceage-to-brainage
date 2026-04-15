import os
import gc
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from PIL import Image
import mtcnn
import keras

from typing import List
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# INPUT_BASE_PATH = Path("/workspace/faceage_mri/renders_filtered")
INPUT_BASE_PATH = Path("/workspace/faceage_mri/renders_filtered_150")
# INPUT_BASE_PATH = Path("/workspace/faceage_mri/renders_gemini")

OUTPUT_CROP_PATH = Path("/workspace/faceage_mri/renders_cropped")
# OUTPUT_CROP_PATH = Path("/workspace/faceage_mri/renders_cropped_gemini")
BASE_MODEL_PATH = Path("/workspace/faceage_mri/FaceAge/weights")

CSV_METADATA_PATH = Path("/workspace/data/mri_data/ixi/metadata/ixi_subject_resolution.csv")
RESULTS_OUTPUT_PATH = Path("/workspace/faceage_mri/age_predictions.csv")

N_SUBJECTS = 100
MODEL_INPUT_SIZE = (160, 160)
GC_EVERY_N_IMAGES = 5

from sklearn.linear_model import LinearRegression
import numpy as np

def fit_calibration(age_pred_dict):
    y_true = []
    y_pred = []

    for v in age_pred_dict.values():
        if v["faceage"] is None or v["true_age"] is None:
            continue

        pred = float(v["faceage"])
        true = float(v["true_age"])

        y_pred.append(pred)
        y_true.append(true)

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true)

    reg = LinearRegression().fit(y_pred, y_true)

    a = reg.coef_[0]
    b = reg.intercept_

    print(f"\nCalibration: true_age = {a:.3f} * pred + {b:.3f}")

    return a, b

def extract_ixi_id(subject_key: str) -> int:
    """
    Example:
        'IXI023-Guys-0699-T1_face' -> 23
    """
    return int(subject_key.split("-")[0].replace("IXI", ""))


def load_age_lookup(csv_path: Path) -> dict:
    """
    Load CSV and return a dictionary:
        {IXI_ID: resolved_age}
    """
    df = pd.read_csv(str(csv_path), sep=",")

    print(df.columns.tolist())

    # Make sure IDs are integers
    df["IXI_ID"] = df["IXI_ID"].astype(int)

    return dict(zip(df["IXI_ID"], df["resolved_age"]))


def add_true_age_to_predictions(age_pred_dict: dict, csv_path: Path) -> dict:
    """
    Add true age from CSV into the prediction dict using the subject key.
    """
    age_lookup = load_age_lookup(csv_path)

    for subject_key in age_pred_dict.keys():
        ixi_id = extract_ixi_id(subject_key)
        age_pred_dict[subject_key]["true_age"] = age_lookup.get(ixi_id, None)

    return age_pred_dict

def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image and strip alpha channel if present."""
    image = imread(str(image_path))

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return image


def crop_face(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop a face from an image using an MTCNN bounding box.

    MTCNN may return negative x/y values when the face is near the image edge,
    so coordinates are clipped to zero.
    """
    x, y, width, height = bbox
    x = max(0, x)
    y = max(0, y)

    x2 = x + width
    y2 = y + height

    return image[y:y2, x:x2]


def detect_first_face(image_path: Path, detector) -> dict:
    """Detect the first face in an image using MTCNN."""
    try:
        image = load_rgb_image(image_path)
        results = detector.detect_faces(image)
        return results[0] if results else {}
    except Exception as exc:
        print(f'ERROR: Processing error for file "{image_path}": {exc}')
        return {}


def save_face_crop(image_path: Path, bbox_dict: dict, output_dir: Path, subject_id: str) -> bool:
    """Crop the detected face and save it to disk."""
    if not bbox_dict:
        print("  -> WARNING: No face detected. Skipping crop.")
        return False

    image = load_rgb_image(image_path)
    face = crop_face(image, bbox_dict["box"])

    if face.size == 0:
        print("  -> ERROR: Invalid crop dimensions.")
        return False

    save_path = output_dir / f"{subject_id}_crop.png"
    imsave(str(save_path), face)
    print(f"  -> Success! Face cropped and saved to: {save_path}")
    return True


def predict_age(model, image_path: Path, bbox_dict: dict) -> np.ndarray:
    """Run age estimation on the detected face."""
    image = load_rgb_image(image_path)

    x, y, width, height = bbox_dict["box"]
    x = max(0, x)
    y = max(0, y)

    face = image[y:y + height, x:x + width]
    face = Image.fromarray(np.uint8(face)).convert("RGB")
    face = np.asarray(face.resize(MODEL_INPUT_SIZE))

    mean = face.mean()
    std = face.std()
    face = (face - mean) / std

    model_input = face.reshape(1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)
    prediction = model.predict(model_input, verbose=0)

    return np.squeeze(prediction)

def save_results_to_csv(age_pred_dict: dict, output_path: Path):
    """
    Save predictions and ground truth to a CSV file.
    """
    rows = []

    for subject_id, values in age_pred_dict.items():
        pred = values.get("faceage")
        true = values.get("true_age")

        # Convert numpy → float
        if isinstance(pred, np.ndarray):
            pred = float(pred)

        rows.append({
            "subject_id": subject_id,
            "predicted_age": pred,
            "true_age": true
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

def evaluate_predictions(age_pred_dict, use_calibrated=False):
    y_true, y_pred = [], []

    for v in age_pred_dict.values():
        true = v.get("true_age")

        if use_calibrated:
            pred = v.get("faceage_calibrated")
        else:
            pred = v.get("faceage")

        if pred is None or true is None:
            continue

        y_true.append(true)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)

    print(f"\nUsing {'CALIBRATED' if use_calibrated else 'RAW'} predictions")
    print(f"MAE: {mae:.2f}")
    print(f"Bias: {bias:.2f}")

if __name__ == "__main__":
    OUTPUT_CROP_PATH.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        f for f in INPUT_BASE_PATH.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if N_SUBJECTS > 0:
        input_files = input_files[:N_SUBJECTS]

    print(f"Found {len(input_files)} images to process.")
    print("Initializing MTCNN model...")
    detector = mtcnn.mtcnn.MTCNN()

    face_data = {}
    start_time = time.time()

    for idx, image_path in enumerate(input_files, start=1):
        subject_id = image_path.stem
        print(f'\n({idx}/{len(input_files)}) Running face localization for "{subject_id}"')

        bbox_dict = detect_first_face(image_path, detector)

        face_data[subject_id] = {
            "path_to_image": image_path,
            "mtcnn_output_dict": bbox_dict,
        }

        save_face_crop(image_path, bbox_dict, OUTPUT_CROP_PATH, subject_id)

        if idx % GC_EVERY_N_IMAGES == 0:
            tf.keras.backend.clear_session()
            gc.collect()

    print(f"\nPipeline finished in {time.time() - start_time:.2f} seconds.")
    print(f"Check the '{OUTPUT_CROP_PATH}' directory for your cropped images.")

    model_path = BASE_MODEL_PATH / "faceage_model.h5"
    model = keras.models.load_model(str(model_path))

    age_pred_dict = {}
    start_time = time.time()

    print()

    for idx, (subject_id, data) in enumerate(face_data.items(), start=1):
        print(f'({idx}/{len(face_data)}) Running the age estimation step for "{subject_id}"')

        bbox_dict = data["mtcnn_output_dict"]

        if not bbox_dict:
            age_pred_dict[subject_id] = {
                "faceage": None,
                "true_age": None
            }
            continue

        pred_age = predict_age(model, data["path_to_image"], bbox_dict)

        age_pred_dict[subject_id] = {
            "faceage": pred_age
        }

    # Add ground truth
    age_pred_dict = add_true_age_to_predictions(age_pred_dict, CSV_METADATA_PATH)


    # Evaluate
    evaluate_predictions(age_pred_dict)

    # Save results
    save_results_to_csv(age_pred_dict, RESULTS_OUTPUT_PATH)

    print(f"\nage_pred_dict: {age_pred_dict}")


    items = list(age_pred_dict.items())

    calib_items = items[:20]
    eval_items  = items[20:]

    calib_dict = dict(calib_items)
    eval_dict  = dict(eval_items)

    a, b = fit_calibration(calib_dict)

    for subject_id in eval_dict:
        pred = eval_dict[subject_id]["faceage"]

        if pred is not None:
            pred = float(pred)
            calibrated = a * pred + b

            eval_dict[subject_id]["faceage_calibrated"] = calibrated

    evaluate_predictions(eval_dict, use_calibrated=True)