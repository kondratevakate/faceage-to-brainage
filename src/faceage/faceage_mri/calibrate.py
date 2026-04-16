import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ==========================================
# Configuration
# ==========================================
RESULTS_CSV_PATH = Path("/workspace/faceage_mri/age_predictions_multiview_complete.csv")
N_CALIB_IMAGES = 100

# ==========================================
# Helper Functions
# ==========================================
def fit_calibration(calib_data):
    """Fits linear regression on the calibration subset."""
    y_pred = np.array([row["predicted_age"] for row in calib_data]).reshape(-1, 1)
    y_true = np.array([row["true_age"] for row in calib_data])

    if len(y_pred) == 0:
        return 1.0, 0.0

    reg = LinearRegression().fit(y_pred, y_true)
    a, b = reg.coef_[0], reg.intercept_
    
    print(f"Calibration Formula: true_age = {a:.3f} * pred + {b:.3f}\n")
    return a, b

def evaluate(y_true, y_pred, label=""):
    """Calculates and prints MAE and Bias."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    
    print(f"--- {label} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"Bias: {bias:.2f}\n")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    if not RESULTS_CSV_PATH.exists():
        print(f"Error: Could not find CSV at {RESULTS_CSV_PATH}")
        exit(1)

    # 1. Load data and drop any rows missing predictions or ground truth
    df = pd.read_csv(RESULTS_CSV_PATH)
    valid_df = df.dropna(subset=["predicted_age", "true_age"]).copy()
    
    # Convert to list of dictionaries for easy manipulation
    items = valid_df.to_dict(orient="records")
    
    if len(items) <= N_CALIB_IMAGES:
        print(f"Not enough valid data. Found {len(items)}, need > {N_CALIB_IMAGES}.")
        exit(1)

    # 2. Shuffle to prevent sequential/alphabetical bias
    # (Uncomment the seed if you want the exact same random split every time you run it)
    # random.seed(42) 
    random.shuffle(items)
    
    # 3. Split data into Calibration (Train) and Evaluation (Test)
    calib_data = items[:N_CALIB_IMAGES]
    eval_data = items[N_CALIB_IMAGES:]
    
    print(f"Total valid subjects: {len(items)}")
    print(f"Calibration set:      {len(calib_data)}")
    print(f"Evaluation set:       {len(eval_data)}\n")
    
    # 4. Fit the calibrator
    a, b = fit_calibration(calib_data)
    
    # 5. Prepare evaluation arrays
    eval_true = [row["true_age"] for row in eval_data]
    eval_pred_raw = [row["predicted_age"] for row in eval_data]
    
    # Apply calibration formula to the raw predictions
    eval_pred_calib = [(a * pred) + b for pred in eval_pred_raw]
    
    # 6. Print Results
    evaluate(eval_true, eval_pred_raw, "RAW PREDICTIONS (Evaluation Set Only)")
    evaluate(eval_true, eval_pred_calib, "CALIBRATED PREDICTIONS (Evaluation Set Only)")