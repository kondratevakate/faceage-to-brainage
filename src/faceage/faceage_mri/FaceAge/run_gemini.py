import os
import gc
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
import keras
import pandas as pd
from sklearn.linear_model import LinearRegression

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Update these paths to match your environment ---
INPUT_BASE_PATH = Path("/workspace/data/mri_data/mri_realistic_backup") 
BASE_MODEL_PATH = Path("/workspace/faceage_mri/FaceAge/weights")
CSV_METADATA_PATH = Path("/workspace/data/mri_data/ixi/metadata/ixi_subject_resolution.csv")
RESULTS_OUTPUT_PATH = Path("/workspace/faceage_mri/age_predictions_gemini.csv")

# NEW: Path for the averaged results
AVERAGED_RESULTS_OUTPUT_PATH = Path("/workspace/faceage_mri/age_predictions_gemini_averaged.csv")

# Constants
N_SUBJECTS = 0 
MODEL_INPUT_SIZE = (160, 160)
GC_EVERY_N_IMAGES = 50 

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
        print("\nWARNING: Not enough valid data to fit calibration.")
        return 1.0, 0.0

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true)

    reg = LinearRegression().fit(y_pred, y_true)

    a = reg.coef_[0]
    b = reg.intercept_

    print(f"\nCalibration: true_age = {a:.3f} * pred + {b:.3f}")

    return a, b

def extract_ixi_id(subject_key: str) -> int:
    return int(subject_key.split("-")[0].replace("IXI", ""))

def load_age_lookup(csv_path: Path) -> dict:
    df = pd.read_csv(str(csv_path), sep=",")
    df["IXI_ID"] = df["IXI_ID"].astype(int)
    return dict(zip(df["IXI_ID"], df["resolved_age"]))

def add_true_age_to_predictions(age_pred_dict: dict, csv_path: Path) -> dict:
    age_lookup = load_age_lookup(csv_path)

    for subject_key in age_pred_dict.keys():
        try:
            ixi_id = extract_ixi_id(subject_key)
            age_pred_dict[subject_key]["true_age"] = age_lookup.get(ixi_id, None)
        except Exception as e:
            print(f"  -> WARNING: Could not extract ID for ground truth from {subject_key}. Error: {e}")
            age_pred_dict[subject_key]["true_age"] = None

    return age_pred_dict

def load_rgb_image(image_path: Path) -> np.ndarray:
    image = imread(str(image_path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def predict_age(model, image_path: Path) -> np.ndarray:
    try:
        image = load_rgb_image(image_path)

        face = Image.fromarray(np.uint8(image)).convert("RGB")
        face = np.asarray(face.resize(MODEL_INPUT_SIZE, Image.LANCZOS))

        mean = face.mean()
        std = face.std()
        
        if std == 0:
            std = 1e-6
            
        face = (face - mean) / std

        model_input = face.reshape(1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3)
        prediction = model.predict(model_input, verbose=0)

        return np.squeeze(prediction)
    except Exception as e:
        print(f"  -> ERROR predicting age for {image_path.name}: {e}")
        return None

def save_results_to_csv(age_pred_dict: dict, output_path: Path):
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
    print(f"\nRaw results saved to: {output_path}")

def save_averaged_results_to_csv(age_pred_dict: dict, output_path: Path) -> dict:
    """
    Groups predictions by the base subject ID, averages the predicted ages,
    saves them to a CSV, and returns a new dictionary for evaluation.
    """
    aggregated_data = {}
    
    for subject_id, values in age_pred_dict.items():
        pred = values.get("faceage")
        true = values.get("true_age")
        
        if pred is None:
            continue
            
        if isinstance(pred, np.ndarray):
            pred = float(pred)
            
        # Extract base ID (e.g., 'IXI023-Guys-0699-T1' from 'IXI023-Guys-0699-T1_az0_el0_crop')
        base_id = subject_id.split('_')[0]
        
        if base_id not in aggregated_data:
            aggregated_data[base_id] = {"preds": [], "true_age": true}
            
        aggregated_data[base_id]["preds"].append(pred)
        
    rows = []
    avg_pred_dict = {}
    
    for base_id, data in aggregated_data.items():
        avg_pred = np.mean(data["preds"])
        
        rows.append({
            "base_subject_id": base_id,
            "averaged_predicted_age": avg_pred,
            "true_age": data["true_age"],
            "views_averaged": len(data["preds"])
        })
        
        # Format like the original dictionary so we can pass it into evaluate_predictions
        avg_pred_dict[base_id] = {
            "faceage": avg_pred,
            "true_age": data["true_age"]
        }
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Averaged results saved to: {output_path}")
    
    return avg_pred_dict

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
        print("\nNo valid pairs found for evaluation.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)

    print(f"\nUsing {'CALIBRATED' if use_calibrated else 'RAW'} predictions")
    print(f"MAE:  {mae:.2f} years")
    print(f"Bias: {bias:.2f} years")


if __name__ == "__main__":
    
    input_files = sorted(
        f for f in INPUT_BASE_PATH.rglob("*")
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    if N_SUBJECTS > 0:
        input_files = input_files[:N_SUBJECTS]

    if not input_files:
        print(f"No images found in {INPUT_BASE_PATH}. Check your paths!")
        exit()

    print(f"Found {len(input_files)} Gemini-rendered images to process.")

    model_path = BASE_MODEL_PATH / "faceage_model.h5"
    print(f"Loading FaceAge model from {model_path}...")
    
    # Configure TensorFlow memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    model = keras.models.load_model(str(model_path))

    age_pred_dict = {}
    start_time = time.time()
    print("\nStarting age estimation...\n")

    for idx, image_path in enumerate(input_files, start=1):
        subject_id = image_path.stem  
        
        if idx % 10 == 0 or idx == 1 or idx == len(input_files):
            print(f"[{idx}/{len(input_files)}] Predicting age for: {subject_id}")

        pred_age = predict_age(model, image_path)

        age_pred_dict[subject_id] = {
            "faceage": pred_age
        }

        if idx % GC_EVERY_N_IMAGES == 0:
            tf.keras.backend.clear_session()
            gc.collect()

    print(f"\nPrediction pipeline finished in {time.time() - start_time:.2f} seconds.")

    print("\n--- EVALUATING INDIVIDUAL VIEWS ---")
    age_pred_dict = add_true_age_to_predictions(age_pred_dict, CSV_METADATA_PATH)
    evaluate_predictions(age_pred_dict)
    save_results_to_csv(age_pred_dict, RESULTS_OUTPUT_PATH)
    
    print("\n--- EVALUATING AVERAGED VIEWS ---")
    avg_pred_dict = save_averaged_results_to_csv(age_pred_dict, AVERAGED_RESULTS_OUTPUT_PATH)
    evaluate_predictions(avg_pred_dict)