import nibabel as nib
import pyvista as pv
import numpy as np
import os
import glob
import scipy
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# print(cpu_count())
# exit()

# Start the virtual framebuffer automatically to prevent headless server crashes
pv.start_xvfb()


# ==========================================
# Batch Processing Execution
# ==========================================



skipped_scenes = ["IXI012-HH-1211-T1",
                  "IXI019-Guys-0702-T1",
                  "IXI035-IOP-0873-T1",
                  "IXI039-HH-1261-T1",
                  "IXI069-Guys-0769-T1",
                  "IXI072-HH-2324-T1",
                  "IXI073-Guys-0755-T1",
                  "IXI083-HH-1357-T1",
                  "IXI087-Guys-0768-T1",
                  "IXI093-HH-1359-T1",
                  "IXI094-HH-1355-T1",
                  "IXI097-HH-1619-T1",


                  "IXI127-HH-1451-T1",
                  "IXI132-HH-1415-T1",
                  "IXI153-Guys-0782-T1",
                  "IXI156-Guys-0837-T1",
                  "IXI157-Guys-0816-T1",
                  "IXI161-HH-2533-T1",
                  "IXI145-Guys-0781-T1",
                  "IXI167-HH-1569-T1",
                  "IXI174-HH-1571-T1",
                  "IXI175-HH-1570-T1",
                  "IXI182-Guys-0792-T1",
                  "IXI188-Guys-0798-T1",
                  "IXI191-Guys-0801-T1",
                  "IXI198-Guys-0803-T1",
                  "IXI199-Guys-0802-T1",
                  "IXI206-HH-1650-T1",
                  "IXI209-Guys-0804-T1",
                  "IXI211-HH-1568-T1",
                  "IXI216-HH-1635-T1",
                  "IXI217-HH-1638-T1",
                  "IXI226-HH-1618-T1",
                  "IXI227-Guys-0813-T1",
                  "IXI230-IOP-0869-T1",
                  "IXI231-IOP-0866-T1",
                  "IXI232-IOP-0898-T1",
                  "IXI233-IOP-0875-T1",
                  "IXI234-IOP-0870-T1",
                  "IXI238-IOP-0883-T1",
                  "IXI239-HH-2296-T1",
                  "IXI252-HH-1693-T1",

                  "IXI256-HH-1723-T1",
                  "IXI259-HH-1804-T1",
                  "IXI260-HH-1805-T1",
                  "IXI266-Guys-0853-T1",
                  "IXI268-Guys-0858-T1",
                  "IXI270-Guys-0847-T1",
                  "IXI274-HH-2294-T1",
                  "IXI275-HH-1803-T1",
                  "IXI277-HH-1770-T1",
                  "IXI291-IOP-0882-T1",
                  "IXI292-IOP-0877-T1",
                  "IXI293-IOP-0876-T1",
                  "IXI294-IOP-0868-T1",
                  "IXI296-HH-1970-T1",
                  "IXI303-IOP-0968-T1",
                  "IXI305-IOP-0871-T1",
                  "IXI306-IOP-0867-T1",
                  "IXI307-IOP-0872-T1",
                  "IXI309-IOP-0897-T1",
                  "IXI310-IOP-0890-T1",
                  "IXI314-IOP-0889-T1",
                  "IXI315-IOP-0888-T1",
                  "IXI322-IOP-0891-T1",
                  "IXI331-IOP-0892-T1",
                  "IXI332-IOP-1134-T1",
                  "IXI354-HH-2024-T1",
                  "IXI357-HH-2076-T1",
                  "IXI369-Guys-0924-T1",
                  "IXI371-IOP-0970-T1",
                  "IXI372-IOP-0971-T1",
                  "IXI373-IOP-0967-T1",
                  "IXI378-IOP-0972-T1",
                  "IXI382-IOP-1135-T1",
                  "IXI383-HH-2099-T1",
                  "IXI384-HH-2100-T1",
                  "IXI385-HH-2078-T1",
                  "IXI387-HH-2101-T1",
                  "IXI388-IOP-0973-T1",
                  "IXI395-IOP-0969-T1",
                  "IXI404-Guys-0950-T1",
                  "IXI406-Guys-0963-T1",
                  "IXI417-Guys-0939-T1",
                  "IXI423-IOP-0974-T1",
                  "IXI424-IOP-0991-T1",
                  "IXI425-IOP-0988-T1",
                  "IXI426-IOP-1011-T1",
                  "IXI427-IOP-1012-T1",
                  "IXI430-IOP-0990-T1",
                  "IXI433-IOP-0989-T1",
                  "IXI434-IOP-1010-T1",
                  "IXI439-HH-2114-T1",
                  "IXI442-IOP-1041-T1",
                  "IXI443-HH-2215-T1",
                  "IXI453-HH-2214-T1",
                  "IXI460-Guys-0999-T1",
                  "IXI461-Guys-0998-T1",
                  "IXI462-IOP-1042-T1",
                  "IXI463-IOP-1043-T1",
                  "IXI464-IOP-1029-T1",
                  "IXI465-HH-2176-T1",
                  "IXI469-IOP-1136-T1",
                  "IXI470-IOP-1030-T1",
                  "IXI473-IOP-1137-T1",
                  "IXI474-IOP-1138-T1",
                  "IXI475-IOP-1139-T1",
                  "IXI476-IOP-1140-T1",
                  "IXI477-IOP-1141-T1",
                  "IXI478-IOP-1142-T1",
                  "IXI484-HH-2179-T1",
                  "IXI487-Guys-1037-T1",
                  "IXI500-Guys-1017-T1",
                  "IXI515-HH-2377-T1",
                  "IXI517-IOP-1144-T1",
                  "IXI526-HH-2392-T1",
                  "IXI527-HH-2376-T1",
                  "IXI532-IOP-1145-T1",
                  "IXI538-HH-2411-T1",
                  "IXI541-IOP-1146-T1",
                  "IXI542-IOP-1147-T1",
                  "IXI543-IOP-1148-T1",
                  "IXI544-HH-2395-T1",
                  "IXI547-IOP-1149-T1",
                  "IXI548-IOP-1150-T1",
                  "IXI553-IOP-1151-T1",
                  "IXI560-Guys-1070-T1",
                  "IXI561-IOP-1152-T1",
                  "IXI563-IOP-1153-T1",
                  "IXI565-HH-2534-T1",
                  "IXI568-HH-2607-T1",
                  "IXI571-IOP-1154-T1",
                  "IXI573-IOP-1155-T1",
                  "IXI574-IOP-1156-T1",
                  "IXI576-Guys-1077-T1",
                  "IXI577-HH-2661-T1",
                  "IXI582-Guys-1127-T1",
                  "IXI588-IOP-1158-T1",
                  "IXI592-Guys-1085-T1",
                  "IXI595-IOP-1159-T1",
                  "IXI596-IOP-1160-T1",
                  "IXI597-IOP-1161-T1",
                  "IXI598-HH-2606-T1",
                  "IXI599-HH-2659-T1",
                  "IXI603-HH-2701-T1",
                  "IXI605-HH-2598-T1",
                  "IXI616-Guys-1092-T1",
                  "IXI617-Guys-1090-T1",
                  "IXI627-Guys-1103-T1",
                  "IXI629-Guys-1095-T1",
                  "IXI630-Guys-1108-T1",
                  "IXI646-HH-2653-T1",
                  "IXI653-Guys-1122-T1",
                  "IXI662-Guys-1120-T1"
                  ]

def render_mri_face_randomized(args):
    """Loads a 3D NIfTI volume, extracts the skin surface, and renders multiple 2D PNGs with random lighting."""
    # Unpack the arguments from the multiprocessing pool
    nifti_path, output_png_path = args
    
    try:
        # 1. Load the NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()

        # Apply a 3D Gaussian filter to smooth out static noise.
        data_smoothed = scipy.ndimage.gaussian_filter(data, sigma=1.0)
        
        # 2. Convert to PyVista ImageData
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = img.header.get_zooms()[:3]

        grid.point_data['values'] = data_smoothed.flatten(order='F')
        
        # 3. Extract the Isosurface (The Skin)
        skin_mesh = grid.contour([150.0]) 
        
        if skin_mesh.n_points == 0:
            print(f"\n  -> WARNING: No surface found for {os.path.basename(nifti_path)}. Skipping.")
            return False

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
        
        # 5. Render every combination of Azimuth and Elevation
        for el in elevation_angles:
            for az in azimuth_angles:
                plotter = pv.Plotter(off_screen=True)
                
                # Apply the randomly generated material settings to the mesh
                plotter.add_mesh(
                    skin_mesh, 
                    color=mat_color, 
                    smooth_shading=True, 
                    specular=mat_specular,
                    specular_power=mat_spec_power,
                    diffuse=mat_diffuse,
                    ambient=mat_ambient
                )
                
                plotter.camera_position = 'yz'
                plotter.camera.azimuth += 180
                plotter.camera.roll -= 90
                plotter.camera.zoom(1.2) 
                
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
    dataset_path = "/workspace/data/mri_data/ixi/images/"
    output_dir = "/workspace/faceage_mri/renders_multiview_complete"
    N_SUBJECTS = 1000
    # skipped_scenes = [] # Add your skipped scenes list here

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