import pymeshlab as ml
import sys
import os

def apply_transform_filter(input_mesh_path, output_mesh_path, translation_vector):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)

    # Aplicar la transformación de traslación
    ms.apply_filter('compute_matrix_from_translation', traslmethod =0, axisx=translation_vector[0], axisy=translation_vector[1], axisz=translation_vector[2])

    # Guardar el resultado
    ms.save_current_mesh(output_mesh_path, binary=False)

if __name__ == "__main__":

    input_mesh_path = sys.argv[1]
    output_path = sys.argv[2]
    trans_x = sys.argv[3]
    trans_y = sys.argv[4]
    trans_z = sys.argv[5]

    # Translation vector
    translation_vector = [float(trans_x), float(trans_y), float(trans_z)]
    output_mesh_path = os.path.join(output_path,'head_centered.ply')
    apply_transform_filter(input_mesh_path, output_mesh_path, translation_vector)
