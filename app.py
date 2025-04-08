import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import trimesh
import sys

import pandas as pd
sys.path.append('/home/ubuntu')
from fixed_dcharts import DCharts
from visualization import visualize_segmentation
from mesh_io import load_mesh

print("Current working directory:", os.getcwd())
print("Templates directory exists:", os.path.exists("templates"))
print("Files in templates:", os.listdir("templates") if os.path.exists("templates") else "Directory not found")


template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

print("Template directory path:", template_dir)
print("Template directory exists:", os.path.exists(template_dir))

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'mesh_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['mesh_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({'filename': filename, 'message': 'File uploaded successfully'})

@app.route('/segment', methods=['POST'])
def segment_mesh():
    data = request.json
    filename = data.get('filename')
    num_charts = int(data.get('num_charts', 8))
    f_max = float(data.get('f_max', 0.1))
    max_iter = int(data.get('max_iter', 10))
    min_size_percent = float(data.get('min_size_percent', 2.0))
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:

        mesh = load_mesh(filepath)
        mesh.face_normals  
        

        vertices = mesh.vertices
        bbox = np.ptp(vertices, axis=0)
        scale = 1.0 / np.max(bbox)
        mesh.vertices = vertices * scale
        
 
        segmenter = DCharts(mesh, F_max=f_max, max_iter=max_iter)
        charts = segmenter.segment(num_charts=num_charts)
        

        min_size = len(mesh.faces) * (min_size_percent / 100.0)
        valid_charts = [chart for chart in charts if len(chart) >= min_size]
        
        vch = pd.DataFrame(valid_charts).to_csv('output.csv', index=False, header=False)
        chart_stats = []
        for i, chart in enumerate(valid_charts):
            size = len(chart)
            percentage = (size / len(mesh.faces)) * 100
            chart_stats.append({
                'id': i,
                'size': size,
                'percentage': round(percentage, 1)
            })
        

        output_filename = f"{os.path.splitext(filename)[0]}_segmented.obj"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Create visualization
        viz_mesh = visualize_segmentation(mesh, valid_charts, display=False)
        viz_mesh.export(file_type="obj", file_obj=output_path)
        
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'total_faces': len(mesh.faces),
            'num_valid_charts': len(valid_charts),
            'chart_stats': chart_stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/view/<filename>')
def view_result(filename):
    return render_template('view.html', filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
