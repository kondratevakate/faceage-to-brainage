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
from sklearn.linear_model import LinearRegression

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Paths
INPUT_BASE_PATH = Path("/workspace/faceage_mri/renders_multiview_complete")
OUTPUT_CROP_PATH = Path("/workspace/faceage_mri/renders_multiview_complete_cropped")

BASE_MODEL_PATH = Path("/workspace/faceage_mri/FaceAge/weights")
CSV_METADATA_PATH = Path("/workspace/data/mri_data/ixi/metadata/ixi_subject_resolution.csv")

RESULTS_OUTPUT_PATH = Path("/workspace/faceage_mri/age_predictions_multiview_complete.csv")

# Parameters
N_SUBJECTS = 1000
N_CALIB_IMAGES = 100
MODEL_INPUT_SIZE = (160, 160)
GC_EVERY_N_IMAGES = 15  # Increased slightly since we are processing batches of images per subject


def fit_calibration(age_pred_dict):
    y_true = []
    y_pred = []

    for v in age_pred_dict.values():
        if v.get("faceage") is None or v.get("true_age") is None:
            continue

        pred = float(v["faceage"])
        true = float(v["true_age"])

        y_pred.append(pred)
        y_true.append(true)

    if not y_pred:
        print("Not enough data for calibration.")
        return 1.0, 0.0

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
        'IXI023-Guys-0699-T1' -> 23
    """
    return int(subject_key.split("-")[0].replace("IXI", ""))

def load_age_lookup(csv_path: Path) -> dict:
    """
    Load CSV and return a dictionary: {IXI_ID: resolved_age}
    """
    df = pd.read_csv(str(csv_path), sep=",")
    df["IXI_ID"] = df["IXI_ID"].astype(int)
    return dict(zip(df["IXI_ID"], df["resolved_age"]))

def add_true_age_to_predictions(age_pred_dict: dict, csv_path: Path) -> dict:
    """
    Add true age from CSV into the prediction dict using the subject key.
    """
    age_lookup = load_age_lookup(csv_path)

    for subject_key in age_pred_dict.keys():
        try:
            ixi_id = extract_ixi_id(subject_key)
            age_pred_dict[subject_key]["true_age"] = age_lookup.get(ixi_id, None)
        except Exception:
            age_pred_dict[subject_key]["true_age"] = None

    return age_pred_dict

def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image and strip alpha channel if present."""
    image = imread(str(image_path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def crop_face(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Crop a face from an image using an MTCNN bounding box."""
    x, y, width, height = bbox
    x = max(0, x)
    y = max(0, y)
    return image[y:y+height, x:x+width]

def detect_first_face(image_path: Path, detector) -> dict:
    """Detect the first face in an image using MTCNN."""
    try:
        image = load_rgb_image(image_path)
        results = detector.detect_faces(image)
        return results[0] if results else {}
    except Exception as exc:
        print(f'  -> ERROR processing "{image_path.name}": {exc}')
        return {}

def save_face_crop(image_path: Path, bbox_dict: dict, output_dir: Path, file_stem: str) -> bool:
    """Crop the detected face and save it to disk."""
    if not bbox_dict:
        return False

    image = load_rgb_image(image_path)
    face = crop_face(image, bbox_dict["box"])

    if face.size == 0:
        return False

    save_path = output_dir / f"{file_stem}_crop.png"
    imsave(str(save_path), face)
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
    """Save predictions and ground truth to a CSV file."""
    rows = []
    for subject_id, values in age_pred_dict.items():
        pred = values.get("faceage")
        true = values.get("true_age")
        
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
        pred = v.get("faceage_calibrated") if use_calibrated else v.get("faceage")

        if pred is None or true is None or np.isnan(pred) or np.isnan(true):
            continue

        y_true.append(true)
        y_pred.append(pred)

    if not y_true:
        print("No valid pairs to evaluate.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)

    print(f"\nUsing {'CALIBRATED' if use_calibrated else 'RAW'} predictions")
    print(f"MAE: {mae:.2f}")
    print(f"Bias: {bias:.2f}")

if __name__ == "__main__":
    OUTPUT_CROP_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Discover all images recursively and group by subject folder
    all_image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        all_image_paths.extend(INPUT_BASE_PATH.rglob(ext))
    
    subject_to_images = {}
    for path in all_image_paths:
        subject_id = path.parent.name
        if subject_id not in subject_to_images:
            subject_to_images[subject_id] = []
        subject_to_images[subject_id].append(path)

    subject_ids = sorted(subject_to_images.keys())

    if N_SUBJECTS > 0:
        subject_ids = subject_ids[:N_SUBJECTS]

    print(f"Found {len(all_image_paths)} images across {len(subject_ids)} subjects.")
    print("Initializing MTCNN model...")
    detector = mtcnn.mtcnn.MTCNN()

    face_data = {}
    start_time = time.time()
    total_images_processed = 0

    # 2. MTCNN Localization Loop
    for idx, subject_id in enumerate(subject_ids, start=1):
        images_for_subject = subject_to_images[subject_id]
        print(f'\n({idx}/{len(subject_ids)}) Running localization for "{subject_id}" ({len(images_for_subject)} views)')
        
        # Ensure crop output subfolder exists
        subj_crop_dir = OUTPUT_CROP_PATH / subject_id
        subj_crop_dir.mkdir(parents=True, exist_ok=True)

        face_data[subject_id] = []
        faces_found = 0

        for img_path in images_for_subject:
            bbox_dict = detect_first_face(img_path, detector)
            
            if bbox_dict:
                save_face_crop(img_path, bbox_dict, subj_crop_dir, img_path.stem)
                face_data[subject_id].append({
                    "path_to_image": img_path,
                    "mtcnn_output_dict": bbox_dict,
                })
                faces_found += 1
            
            total_images_processed += 1
            
            # GC check
            if total_images_processed % GC_EVERY_N_IMAGES == 0:
                tf.keras.backend.clear_session()
                gc.collect()
        
        print(f"  -> Successfully detected faces in {faces_found}/{len(images_for_subject)} views.")

    print(f"\nPipeline finished in {time.time() - start_time:.2f} seconds.")

    # 3. Age Prediction Loop
    model_path = BASE_MODEL_PATH / "faceage_model.h5"
    model = keras.models.load_model(str(model_path))

    age_pred_dict = {}
    print("\nStarting age prediction step...")

    for idx, subject_id in enumerate(subject_ids, start=1):
        subject_faces = face_data.get(subject_id, [])
        
        if not subject_faces:
            print(f'({idx}/{len(subject_ids)}) No faces available to predict for "{subject_id}"')
            age_pred_dict[subject_id] = {"faceage": None}
            continue

        print(f'({idx}/{len(subject_ids)}) Predicting age for "{subject_id}"...')
        
        # Predict on ALL views and average the result
        preds = []
        for face_info in subject_faces:
            pred = predict_age(model, face_info["path_to_image"], face_info["mtcnn_output_dict"])
            preds.append(float(pred))
            
        avg_pred = np.mean(preds)
        age_pred_dict[subject_id] = {"faceage": avg_pred}

    # 4. Add Ground Truth & Evaluate
    age_pred_dict = add_true_age_to_predictions(age_pred_dict, CSV_METADATA_PATH)
    evaluate_predictions(age_pred_dict)


    save_results_to_csv(age_pred_dict, RESULTS_OUTPUT_PATH)

    # 5. Calibration Evaluation
    items = [(k, v) for k, v in age_pred_dict.items() if v.get("faceage") is not None and v.get("true_age") is not None]

    if len(items) > N_CALIB_IMAGES:
        calib_items = items[:N_CALIB_IMAGES]
        eval_items  = items[:N_CALIB_IMAGES]

        calib_dict = dict(calib_items)
        eval_dict  = dict(eval_items)

        a, b = fit_calibration(calib_dict)

        for subject_id, data in eval_dict.items():
            pred = data["faceage"]
            calibrated = a * pred + b
            eval_dict[subject_id]["faceage_calibrated"] = calibrated

        evaluate_predictions(eval_dict, use_calibrated=True)
    else:
        print("\nNot enough valid predictions to perform train/test calibration split.")