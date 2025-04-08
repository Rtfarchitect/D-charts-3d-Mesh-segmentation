# 🧩 D-Charts: Developable Mesh Segmentation Tool

A Python + Flask-based application that segments 3D triangular meshes into nearly developable and connected charts using the **D-Charts algorithm**.

This tool supports real-time parameter tuning, result download, and 3D visualization directly in the browser.

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dcharts-segmentation.git
```

### 2. Create a Conda Environment

```bash
conda create -n dcharts-env python=3.9
conda activate dcharts-env
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

> 🔧 Make sure `trimesh`, `flask`, `numpy`, `networkx`, `scipy`, and `pandas` are included in `requirements.txt`.

---

## 🚀 Run the Application

Start the Flask app by running:

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## 🧠 How to Use

1. **Upload your mesh**  
   Click the **Upload Mesh** button and select a `.obj` file.

2. **Adjust Parameters**  
   - `F_max`: Maximum allowed fitting error (planarity).
   - `num_charts`: Number of charts (segments).
   - `max_iter`: Number of Lloyd iterations.
   - `min_size_percent`: Minimum chart size threshold.

3. **Run Segmentation**  
   Click the **Segment** button. The algorithm will run on the server and process your mesh.

4. **View Results**
   - Summary statistics are shown on screen.
   - You can **download** the segmented `.obj` mesh.
   - The mesh is **visualized in 3D** with color-coded charts.

---

## 📁 Output

- `output.csv`: Triangle indices for each chart.
- `output/mesh.obj`: Segmented mesh with chart-based face colors.

---

## 🎯 Features

- Robust D-Charts segmentation with multi-term cost function.
- Interactive web interface via Flask.
- 3D visualization using Trimesh.
- Outputs OBJ + CSV for use in modeling, fabrication, or UV mapping.

---

## 📄 License

MIT License or your preferred license.

---

## 🙌 Credits

Developed by [Your Name]  
Based on concepts from *Quasi-Developable Mesh Segmentation* (Eurographics 2005)

---

## 🦗 Rhino + Grasshopper Integration

At the end of the workflow, it's also possible to use the included **Grasshopper file** named **ToGh.gh** in the repository to:

- **Visualize the segmented mesh** inside Rhino
- **Unfold the parts** using the **Ivy plugin**

This allows for further design development and fabrication preparation within Rhino's parametric environment.
