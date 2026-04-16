import joblib
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
import scipy.io as sio

if __name__ == "__main__":
    #if len(sys.argv) != 6:
    #    print("Usage: python sliceEnhancement.py path/file.nii outputPath referenceNii")
    #    sys.exit(1)

    nii_file = sys.argv[1]
    output_path = sys.argv[2]
    model_path = sys.argv[3]


# Read model .joblib
linear_regression_model = joblib.load(model_path)

# Read NIfTI
nifti_img = nib.load(nii_file)
nifti_data = nifti_img.get_fdata()

# Extract max and mean intensity
max_intensity = np.max(nifti_data)
mean_intensity = np.mean(nifti_data)

# Create a dataframe
data = {'Int_Max': [max_intensity], 'Int_Media': [mean_intensity]}
df = pd.DataFrame(data)

# Treshold prediction
prediction = linear_regression_model.predict(df)

# Save prediction .mat
result_data = {'prediction': prediction}
sio.savemat(os.path.join(output_path,'threshold.mat'), result_data)