
from mesh_io import load_mesh
import sys
sys.path.append('/home/ubuntu')  
from fixed_dcharts import DCharts
from visualization import visualize_segmentation
import numpy as np
import pandas as pd

mesh = load_mesh("lowres-bunny.obj")
mesh.face_normals  


vertices = mesh.vertices
bbox = np.ptp(vertices, axis=0)
scale = 1.0 / np.max(bbox)
mesh.vertices = vertices * scale


segmenter = DCharts(mesh, F_max=0.02, max_iter=50)
charts = segmenter.segment(num_charts=12)


min_size = len(mesh.faces) * 0.02
valid_charts = [chart for chart in charts if len(chart) >= min_size]

vch = pd.DataFrame(valid_charts).to_csv('output.csv', index=False, header=False)

print("\nDetailed chart information:")
for i, chart in enumerate(valid_charts):
    size = len(chart)
    percentage = (size / len(mesh.faces)) * 100
    print(f"Chart {i}: {size} faces ({percentage:.1f}%), indices: {list(chart)[:5]}...")

print(f"\nTotal faces in mesh: {len(mesh.faces)}")
print(f"Number of valid charts: {len(valid_charts)}")


mesh = visualize_segmentation(mesh, valid_charts)
mesh.export(file_type="obj", file_obj="output/mesh.obj") 