import pymeshlab as ml
import sys
import os

def apply_transform_filter(input_mesh_path, output_mesh_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)

    ms.compute_scalar_ambient_occlusion()

    ms.compute_selection_by_scalar_per_vertex(inclusive=True, minq=0, maxq=0.027)

    ms.meshing_remove_selected_faces()

    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=ml.PureValue(200), removeunref=True)
    ms.save_current_mesh(output_mesh_path, binary=False)

if __name__ == "__main__":

    input_mesh_path = sys.argv[1]
    output_path = sys.argv[2]

    # Translation vector
    output_mesh_path = os.path.join(output_path,'head_empty.ply')
    apply_transform_filter(input_mesh_path, output_mesh_path)
