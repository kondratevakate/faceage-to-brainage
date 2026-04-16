import pymeshlab as ml
import sys
import os

def apply_transform_filter(input_mesh_path, output_mesh_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)

    # Remove artifacts
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=ml.PercentageValue(90), removeunref=True)

    # Guardar el resultado
    ms.save_current_mesh(output_mesh_path, binary=False)

if __name__ == "__main__":

    input_mesh_path = sys.argv[1]
    output_path = sys.argv[2]

    output_mesh_path = os.path.join(output_path,'head_cleaned.ply')
    apply_transform_filter(input_mesh_path, output_mesh_path)
