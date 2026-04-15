from pymeshlab import MeshSet, Percentage
import sys
import os


input_mesh = sys.argv[1]
threshold_z1 = sys.argv[2] 
threshold_y1 = sys.argv[3]
output_path = sys.argv[4]
smooth = sys.argv[5]

# Crear una instancia de la clase MeshSet
ms = MeshSet()

# Cargar una malla desde un archivo (reemplaza 'path/to/your/mesh.obj' con tu ruta)
ms.load_new_mesh(input_mesh)

# Operación 1: Selección condicional de caras
ms.apply_filter('compute_selection_by_condition_per_face', condselect="(z1 < "+threshold_z1+") || (y1 < "+threshold_y1+")")

# Operación 2: Eliminar caras y vértices seleccionados
ms.apply_filter('meshing_remove_selected_vertices_and_faces')

# Operación 3: Eliminar piezas aisladas con respecto al diámetro
ms.apply_filter('meshing_remove_connected_component_by_diameter', mincomponentdiag=Percentage(50), removeunref=True)

ms.save_current_mesh(os.path.join(os.path.dirname(output_path),'face_noSmooth.ply'), binary=False, save_vertex_color=False, save_vertex_quality=False)

if smooth=='True':
    # Operación 4: Taubin filter
    ms.apply_filter('apply_coord_taubin_smoothing',lambda_=0.8, mu=-0.53, stepsmoothnum=10, selected=False)
    
    # Operación 5: Reconstrucción de superficie: Screened Poisson
    ms.apply_filter(
        'generate_surface_reconstruction_screened_poisson',
        visiblelayer=False,
        depth=8,
        fulldepth=5,
        cgdepth=0,
        scale=1.1,
        samplespernode=1.5,
        pointweight=4,
        iters=8,
        confidence=False,
        preclean=True,
        threads=16
    )
    
    # Operación 6: Selección condicional de caras
    ms.apply_filter('compute_selection_by_condition_per_face', condselect="(z1 < "+threshold_z1+") || (y1 < "+threshold_y1+")")
    
    # Operación 7: Eliminar caras y vértices seleccionados
    ms.apply_filter('meshing_remove_selected_vertices_and_faces')
    
    # Operación 8: Eliminar piezas aisladas con respecto al diámetro
    ms.apply_filter('meshing_remove_connected_component_by_diameter', mincomponentdiag=Percentage(90), removeunref=True)


# Guardar la malla procesada (reemplaza 'path/to/your/processed_mesh.obj' con tu ruta)
ms.save_current_mesh(output_path, binary=False, save_vertex_color=False, save_vertex_quality=False)
