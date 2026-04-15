import nibabel as nib
import pyvista as pv
import numpy as np
import os
import glob
import scipy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Start the virtual framebuffer automatically to prevent headless server crashes
pv.start_xvfb()

def render_mri_face(args):
    """Loads a 3D NIfTI volume, extracts the skin surface, and renders a 2D PNG."""
    # Unpack the arguments from the multiprocessing pool
    nifti_path, output_png_path = args
    
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
            print(f"\n  -> WARNING: No surface found for {os.path.basename(nifti_path)}. Skipping.")
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
        print(f"\n  -> ERROR processing {os.path.basename(nifti_path)}: {e}")
        return False
        
    finally:
        # CRITICAL: Always close the plotter to prevent memory leaks and frozen terminals
        if 'plotter' in locals():
            plotter.close()

# ==========================================
# Batch Processing Execution
# ==========================================

if __name__ == '__main__':
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
        # Determine how many CPU cores to use. Leaving a couple free keeps your OS stable.
        MAX_WORKERS = max(1, cpu_count() - 2)
        print(f"Found {len(nifti_files)} files to process. Starting batch render on {MAX_WORKERS} cores...\n")

        # Prepare arguments for multiprocessing
        job_args = []
        for filepath in nifti_files:
            filename = os.path.basename(filepath)
            base_name = filename.replace(".nii.gz", "")
            output_filepath = os.path.join(output_dir, f"{base_name}.png")
            job_args.append((filepath, output_filepath))

        success_count = 0

        # Execute jobs in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(render_mri_face, arg): arg for arg in job_args}
            
            for i, future in enumerate(as_completed(futures), 1):
                success = future.result()
                if success:
                    success_count += 1
                
                # Print progress over the same line to keep the console clean
                print(f"\rProgress: [{i}/{len(nifti_files)}] rendered.", end="", flush=True)

        print(f"\n\nBatch processing complete! Successfully rendered {success_count} out of {len(nifti_files)} images.")
        print(f"Check the '{output_dir}' folder for your results.")