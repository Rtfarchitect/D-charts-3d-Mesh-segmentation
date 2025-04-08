import trimesh

def load_mesh(filepath):
    return trimesh.load(filepath)

def save_segmentation(mesh, chart_labels, output_file):
    mesh.visual.vertex_colors = chart_labels
    mesh.export(output_file)