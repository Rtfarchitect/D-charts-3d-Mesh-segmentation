import numpy as np
import trimesh
from random import randint
def visualize_segmentation(mesh, charts, display=True):

    
    chart_colors = []
    for i in range(len(charts)):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        a = 255
        chart_colors.append([r, g, b, a])

    colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    colors[:, 3] = 255  # 
    

    for i, chart in enumerate(charts):

        color = np.array(chart_colors[i], dtype=np.uint8)
        chart_faces = list(chart)
        colors[chart_faces] = color
    
    viz_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_colors=colors,
        process=False
    )
    
    if display:

        scene = trimesh.Scene()
        scene.add_geometry(viz_mesh)
        scene.show(
            smooth=False,          
            background=[1,1,1,1],  
            flags={
                'wireframe': False,
                'cull': False,
                'shadows': False
            }
        )
    viz_mesh.export(file_type="obj", file_obj="output/mesh.obj") 
    return viz_mesh