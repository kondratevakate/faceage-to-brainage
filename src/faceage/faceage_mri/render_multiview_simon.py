import nibabel as nib
import pyvista as pv
import numpy as np
import os
import glob
import scipy.ndimage
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Start the virtual framebuffer automatically to prevent headless server crashes
pv.start_xvfb()

# ==========================================
# Batch Processing Execution
# ==========================================

def render_mri_face_randomized(args):
    """Loads a 3D NIfTI volume, extracts the skin surface, and renders multiple 2D PNGs with random lighting."""
    nifti_path, output_png_path = args
    
    try:
        # 1. Load the NIfTI file AND FORCE CANONICAL ORIENTATION
        img = nib.load(nifti_path)
        canonical_img = nib.as_closest_canonical(img) 
        data = canonical_img.get_fdata()

        # --- THE TRASH REMOVER ---
        # Zero out the back 25% and bottom 15% to destroy scanner beds and neck coils.
        # (Note: If this ever creates flat planes that bleed into the face, remove this
        # and use PyVista's `clip_box` on the mesh itself later on).
        y_cutoff = int(data.shape[1] * 0.25)
        data[:, :y_cutoff, :] = 0 
        
        z_cutoff = int(data.shape[2] * 0.15)
        data[:, :, :z_cutoff] = 0

        # Apply a 3D Gaussian filter to smooth out remaining static noise.
        data_smoothed = scipy.ndimage.gaussian_filter(data, sigma=1.0)
        
        # 2. Convert to PyVista ImageData
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = canonical_img.header.get_zooms()[:3] 
        grid.point_data['values'] = data_smoothed.flatten(order='F')
        
        # 3. Extract the Isosurface (FIX: ADAPTIVE THRESHOLDING)
        # Find the 98th percentile to ignore extreme bright outliers, then grab ~15% of that for skin.
        p98_intensity = np.percentile(data_smoothed, 98)
        adaptive_threshold = p98_intensity * 0.15 
        
        skin_mesh = grid.contour([adaptive_threshold])
        
        if skin_mesh.n_points == 0:
            print(f"\n  -> WARNING: No surface found for {os.path.basename(nifti_path)}. Skipping.")
            return False

        # Keep only the largest connected component (the head)
        skin_mesh = skin_mesh.connectivity('largest')

        # ==========================================
        # RANDOM MATERIAL & LIGHTING GENERATOR
        # ==========================================
        skin_colors = ['bisque', 'peachpuff', 'wheat', 'tan', 'navajowhite', 'moccasin', 'burlywood']
        
        mat_color = random.choice(skin_colors)
        mat_specular = round(random.uniform(0.1, 0.5), 2)
        mat_spec_power = random.randint(10, 40)
        mat_diffuse = round(random.uniform(0.6, 0.9), 2)
        mat_ambient = round(random.uniform(0.1, 0.35), 2)
        
        # 4. Define our 2D grid of angles
        azimuth_angles = [-10, 0, 10] 
        elevation_angles = [-10, 0, 10]
        
        base_path = output_png_path.replace(".png", "")
        
        # --- FIX: PRE-CALCULATE DYNAMIC CAMERA POSITION ---
        # bounds format: (xmin, xmax, ymin, ymax, zmin, zmax)
        bounds = skin_mesh.bounds
        cx, cy, cz = skin_mesh.center
        
        # In standard RAS, +Y is Anterior (the nose). bounds[3] is the absolute front of the face.
        nose_y = bounds[3]
        
        # Calculate the depth of the head so we scale the camera distance proportionally
        head_depth = bounds[3] - bounds[2] 
        
        # Position camera exactly 1.2x the head's depth away from the nose
        camera_y_position = nose_y + (head_depth * 1.2)

        # 5. Render every combination of Azimuth and Elevation
        for el in elevation_angles:
            for az in azimuth_angles:
                plotter = pv.Plotter(off_screen=True)
                
                plotter.add_mesh(
                    skin_mesh, 
                    color=mat_color, 
                    smooth_shading=True, 
                    specular=mat_specular,
                    specular_power=mat_spec_power,
                    diffuse=mat_diffuse,
                    ambient=mat_ambient
                )
                
                # --- CAMERA SETUP ---
                # 1. Focal point stays at the center
                plotter.camera.focal_point = (cx, cy, cz)
                
                # 2. Position dynamic camera
                plotter.camera.position = (cx, camera_y_position, cz)
                
                # 3. Tell the camera which way is "Up" (+Z is Superior)
                plotter.camera.up = (0, 0, 1)
                
                # 4. Frame it (Zoom of 1.0 is default, adjust if it's too tight/loose with the new math)
                plotter.camera.zoom(0.5)
                
                # 5. Apply your standard batch angles
                plotter.camera.Azimuth(az)
                plotter.camera.Elevation(el)
                
                angle_png_path = f"{base_path}_az{az}_el{el}.png"
                plotter.screenshot(angle_png_path)
                
                plotter.close()
        
        return True
        
    except Exception as e:
        print(f"\n  -> ERROR processing {os.path.basename(nifti_path)}: {e}")
        return False


if __name__ == '__main__':

    # Update these paths as needed
    dataset_path = "/workspace/data/mri_data/simon/images/"
    output_dir = "/workspace/faceage_mri/simon_renders"

    N_SUBJECTS = 1000
    skipped_scenes = [] # Add your skipped scenes list here

    # Ensure main output directory exists
    os.makedirs(output_dir, exist_ok=True)
    nifti_files = sorted(glob.glob(os.path.join(dataset_path, "*.nii.gz")))

    if N_SUBJECTS > 0:
        nifti_files = nifti_files[:N_SUBJECTS]

    if not nifti_files:
        print(f"No .nii.gz files found in {dataset_path}")
    else:
        MAX_WORKERS = max(1, cpu_count() // 2)
        print(f"Found {len(nifti_files)} files to process. Starting batch render on {MAX_WORKERS} cores...\n")

        # 1. Prepare jobs and generate folders BEFORE spinning up workers
        job_args = []
        for filepath in nifti_files:
            filename = os.path.basename(filepath)
            base_name = filename.replace(".nii.gz", "")

            if base_name in skipped_scenes:
                print(f"Skipping {base_name}")
                continue
            
            # Create a dedicated subfolder for this specific subject
            subject_dir = os.path.join(output_dir, base_name)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Route the output file base into the new subfolder
            output_filepath = os.path.join(subject_dir, f"{base_name}.png")
            job_args.append((filepath, output_filepath))

        success_count = 0

        # 2. Execute jobs in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(render_mri_face_randomized, arg): arg for arg in job_args}
            
            for i, future in enumerate(as_completed(futures), 1):
                success = future.result()
                if success:
                    success_count += 1
                
                # Print real-time progress update
                print(f"\rProgress: [{i}/{len(job_args)}] subject grids rendered.", end="", flush=True)

        print(f"\n\nBatch processing complete! Successfully rendered {success_count} subjects.")
        print(f"Check the '{output_dir}' folder for your perfectly organized results.")