import nibabel as nib
import pyvista as pv
import numpy as np
import os
import glob
import scipy

# Start the virtual framebuffer automatically to prevent headless server crashes
pv.start_xvfb()

def render_mri_face(nifti_path, output_png_path):
    """Loads a 3D NIfTI volume, extracts the skin surface, and renders a 2D PNG."""
    try:
        # 1. Load the NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()

        # Apply a 3D Gaussian filter to smooth out static noise.
        # sigma=1.0 is a good starting point. If the face looks too melty/blurry, lower it to 0.5.
        data_smoothed = scipy.ndimage.gaussian_filter(data, sigma=1.0)
        
        # 2. Convert to PyVista ImageData
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = img.header.get_zooms()[:3]

        # Use only the smoothed data
        grid.point_data['values'] = data_smoothed.flatten(order='F')
        
        # 3. Extract the Isosurface (The Skin)
        # Threshold 50.0 is a starting point. Adjust if faces look too eroded or too noisy.
        skin_mesh = grid.contour([150.0]) 
        
        if skin_mesh.n_points == 0:
            print(f"  -> WARNING: No surface found for {os.path.basename(nifti_path)}. Skipping.")
            return False

        # Keep only the largest connected physical shape (the head)
        # and delete all floating noise/artifacts in the background.
        skin_mesh = skin_mesh.connectivity('largest')

        # 4. Set up the Virtual Camera
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(skin_mesh, color='bisque', smooth_shading=True, specular=0.1)
        
        # Position the camera. Change to 'xz' or 'xy' if the face is sideways
        plotter.camera_position = 'yz'
        plotter.camera.azimuth += 180
        plotter.camera.zoom(1.2) 
        
        plotter.camera.roll -= 90

        # 5. Take the "Photograph"
        plotter.screenshot(output_png_path)
        
        return True
        
    except Exception as e:
        print(f"  -> ERROR processing {os.path.basename(nifti_path)}: {e}")
        return False
        
    finally:
        # CRITICAL: Always close the plotter to prevent memory leaks and frozen terminals
        if 'plotter' in locals():
            plotter.close()

# ==========================================
# Batch Processing Execution
# ==========================================

dataset_path = "/workspace/data/mri_data/ixi/images/"
output_dir = "/workspace/faceage_mri/renders_complete"
N_SUBJECTS = 1000  # Set to 0 to process all files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all NIfTI files in the directory
nifti_files = sorted(glob.glob(os.path.join(dataset_path, "*.nii.gz")))

# Apply the N_SUBJECTS limit
if N_SUBJECTS > 0:
    nifti_files = nifti_files[:N_SUBJECTS]

if not nifti_files:
    print(f"No .nii.gz files found in {dataset_path}")
else:
    print(f"Found {len(nifti_files)} files to process. Starting batch render...\n")

    success_count = 0

    for idx, filepath in enumerate(nifti_files, 1):
        filename = os.path.basename(filepath)
        
        # Create the output filename (e.g., IXI408-Guys-0962-T1.nii.gz -> IXI408-Guys-0962-T1_face.png)
        base_name = filename.replace(".nii.gz", "")
        output_filepath = os.path.join(output_dir, f"{base_name}.png")
        
        print(f"[{idx}/{len(nifti_files)}] Rendering {filename}...")
        
        # Run the render function
        if render_mri_face(filepath, output_filepath):
            success_count += 1

    print(f"\nBatch processing complete! Successfully rendered {success_count} out of {len(nifti_files)} images.")
    print(f"Check the '{output_dir}' folder for your results.")


#NOTE to run:
# /usr/bin/python3 render_mri.py