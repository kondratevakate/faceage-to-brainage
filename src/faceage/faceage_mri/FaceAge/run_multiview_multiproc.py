import os
import gc
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List

# Set TF logging before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from skimage.io import imread, imsave
from PIL import Image
from sklearn.linear_model import LinearRegression

# Paths
INPUT_BASE_PATH = Path("/workspace/faceage_mri/renders_multiview_complete")
# INPUT_BASE_PATH = Path("/workspace/faceage_mri/simon_renders")

OUTPUT_CROP_PATH = Path("/workspace/faceage_mri/renders_multiview_complete_cropped")
# OUTPUT_CROP_PATH = Path("/workspace/faceage_mri/simon_renders_cropped")

BASE_MODEL_PATH = Path("/workspace/faceage_mri/FaceAge/weights")
CSV_METADATA_PATH = Path("/workspace/data/mri_data/ixi/metadata/ixi_subject_resolution.csv")

RESULTS_OUTPUT_PATH = Path("/workspace/faceage_mri/age_predictions.csv")

# Parameters
N_SUBJECTS = 1000
N_CALIB_IMAGES = 100
MODEL_INPUT_SIZE = (160, 160)
NUM_WORKERS = 4 # Adjust based on your CPU/GPU RAM


# ==============================================================================
# 1. HELPER FUNCTIONS (Stateless)
# ==============================================================================

def fit_calibration(age_pred_dict):
    y_true, y_pred = [], []
    for v in age_pred_dict.values():
        if v.get("faceage") is None or v.get("true_age") is None:
            continue
        y_pred.append(float(v["faceage"]))
        y_true.append(float(v["true_age"]))

    if not y_pred:
        print("Not enough data for calibration.")
        return 1.0, 0.0

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true)
    reg = LinearRegression().fit(y_pred, y_true)

    a, b = reg.coef_[0], reg.intercept_
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
        except Exception:
            age_pred_dict[subject_key]["true_age"] = None
    return age_pred_dict

def load_rgb_image(image_path: Path) -> np.ndarray:
    image = imread(str(image_path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def crop_face(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    x, y, width, height = [max(0, val) for val in bbox]
    return image[y:y+height, x:x+width]

def detect_first_face(image_path: Path, detector) -> dict:
    try:
        image = load_rgb_image(image_path)
        results = detector.detect_faces(image)
        return results[0] if results else {}
    except Exception as exc:
        print(f'  -> ERROR processing "{image_path.name}": {exc}')
        return {}

def save_face_crop(image_path: Path, bbox_dict: dict, output_dir: Path, file_stem: str) -> bool:
    if not bbox_dict: return False
    image = load_rgb_image(image_path)
    face = crop_face(image, bbox_dict["box"])
    if face.size == 0: return False
    
    save_path = output_dir / f"{file_stem}_crop.png"
    imsave(str(save_path), face)
    return True

def predict_age(model, image_path: Path, bbox_dict: dict, input_size: tuple) -> np.ndarray:
    image = load_rgb_image(image_path)
    x, y, width, height = [max(0, val) for val in bbox_dict["box"]]
    
    face = image[y:y + height, x:x + width]
    face = Image.fromarray(np.uint8(face)).convert("RGB")
    face = np.asarray(face.resize(input_size))

    face = (face - face.mean()) / face.std()
    model_input = face.reshape(1, input_size[0], input_size[1], 3)
    
    # Using training=False is faster and prevents memory leaks inside TF graphs
    prediction = model(model_input, training=False)
    return np.squeeze(prediction.numpy())

# ==============================================================================
# 2. MULTIPROCESSING WORKER FUNCTIONS
# ==============================================================================

def init_worker(model_base_path: str):
    """
    Initializes TensorFlow context, MTCNN, and Keras models per-process.
    This strictly prevents inter-process memory conflicts on the GPU.
    """
    global worker_detector, worker_age_model
    import tensorflow as tf
    import mtcnn
    import keras

    # Critical: Allow VRAM to grow dynamically so parallel processes don't OOM crash
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    worker_detector = mtcnn.mtcnn.MTCNN()
    model_path = Path(model_base_path) / "faceage_model.h5"
    worker_age_model = keras.models.load_model(str(model_path))

def process_subject(args):
    """
    Worker task: Localizes, crops, and predicts age for a single subject.
    """
    subject_id, image_paths, output_crop_base = args
    global worker_detector, worker_age_model

    subj_crop_dir = Path(output_crop_base) / subject_id
    subj_crop_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    faces_found = 0

    for img_path in image_paths:
        # 1. Localize
        bbox_dict = detect_first_face(img_path, worker_detector)
        
        if bbox_dict:
            # 2. Crop
            save_face_crop(img_path, bbox_dict, subj_crop_dir, img_path.stem)
            faces_found += 1
            
            # 3. Predict Age
            pred = predict_age(worker_age_model, img_path, bbox_dict, MODEL_INPUT_SIZE)
            preds.append(float(pred))

    avg_pred = np.mean(preds) if preds else None
    return subject_id, faces_found, len(image_paths), avg_pred

# ==============================================================================
# 3. EVALUATION & MAIN PIPELINE
# ==============================================================================

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

    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    bias = np.mean(np.array(y_pred) - np.array(y_true))

    print(f"\nUsing {'CALIBRATED' if use_calibrated else 'RAW'} predictions")
    print(f"MAE: {mae:.2f} | Bias: {bias:.2f}")

def save_results_to_csv(age_pred_dict: dict, output_path: Path):
    rows = [
        {
            "subject_id": subject_id, 
            "predicted_age": values.get("faceage"), 
            "true_age": values.get("true_age")
        }
        for subject_id, values in age_pred_dict.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    # Ensure standard multiprocessing behavior on all OS
    mp.set_start_method('spawn', force=True)
    OUTPUT_CROP_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Discover Images
    print("Discovering images...")
    all_image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        all_image_paths.extend(INPUT_BASE_PATH.rglob(ext))
    
    subject_to_images = {}
    for path in all_image_paths:
        subject_id = path.parent.name
        subject_to_images.setdefault(subject_id, []).append(path)

    subject_ids = sorted(subject_to_images.keys())
    if N_SUBJECTS > 0:
        subject_ids = subject_ids[:N_SUBJECTS]

    print(f"Found {len(all_image_paths)} images across {len(subject_ids)} subjects.")

    # 2. Multiprocessing Localization & Prediction
    start_time = time.time()
    age_pred_dict = {}

    # Pack arguments for the pool map
    task_args = [
        (subj, subject_to_images[subj], str(OUTPUT_CROP_PATH)) 
        for subj in subject_ids
    ]

    print(f"\nStarting Multiprocessing Pool with {NUM_WORKERS} workers...")
    
    # maxtasksperchild=10 forces a process reboot every 10 subjects, wiping out Keras memory leaks.
    with mp.Pool(processes=NUM_WORKERS, 
                 initializer=init_worker, 
                 initargs=(str(BASE_MODEL_PATH),), 
                 maxtasksperchild=10) as pool:
        
        # Using imap_unordered for a responsive progress stream
        for idx, result in enumerate(pool.imap_unordered(process_subject, task_args), 1):
            subj_id, faces_found, total_views, avg_pred = result
            
            print(f"[{idx}/{len(subject_ids)}] {subj_id} -> Found {faces_found}/{total_views} faces. Avg Age: {avg_pred:.2f}" if avg_pred else f"[{idx}/{len(subject_ids)}] {subj_id} -> No faces found.")
            age_pred_dict[subj_id] = {"faceage": avg_pred}

    print(f"\nPipeline finished in {time.time() - start_time:.2f} seconds.")

    # 3. Add Ground Truth & Evaluate
    age_pred_dict = add_true_age_to_predictions(age_pred_dict, CSV_METADATA_PATH)
    evaluate_predictions(age_pred_dict)
    save_results_to_csv(age_pred_dict, RESULTS_OUTPUT_PATH)

    # 4. Calibration Evaluation
    items = [(k, v) for k, v in age_pred_dict.items() if v.get("faceage") is not None and v.get("true_age") is not None]

    if len(items) > N_CALIB_IMAGES:
        calib_dict = dict(items[:N_CALIB_IMAGES])
        # Note: Changed from `items[:N_CALIB_IMAGES]` to `items[N_CALIB_IMAGES:]` to test on unseen data
        eval_dict  = dict(items[N_CALIB_IMAGES:]) 

        a, b = fit_calibration(calib_dict)

        for subject_id, data in eval_dict.items():
            eval_dict[subject_id]["faceage_calibrated"] = (a * data["faceage"]) + b

        evaluate_predictions(eval_dict, use_calibrated=True)
    else:
        print("\nNot enough valid predictions to perform train/test calibration split.")