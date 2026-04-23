# SpatioThematicVLM

**AI-Powered Thematic Map Intelligent Analysis Platform**  
Powered by Qwen2.5-VL-7B + LangGraph Multi-Agent Workflow + ArcPy Spatial Statistics, enabling automatic visual parsing, semantic understanding, spatial autocorrelation analysis, and professional report generation for thematic maps.

---

## Features

- Multimodal VLM automatically extracts map visual elements (title, legend, color scale, etc.)
- Intelligent semantic theme interpretation + geographic significance analysis
- Full spatial statistics: Global Moran's I + Local Moran's I (hot spot / cold spot / outlier detection)
- Automatic reverse geocoding of significant clusters (locate real place)
- Generates academic-grade Markdown analysis reports

---

## Project Structure

```
SpatioThematicVLM/
├── app/                      # Main backend service
│   ├── app.py                # FastAPI + LangGraph workflow
│   └── nodes.py              # All agent nodes + VLM calls
├── arcpy_gis_service.py      # ArcPy GIS preprocessing and spatial analysis service
├── streamlit_app.py          # Streamlit frontend interface
├── requirements.txt
└── README.md
```

---

## How to Install and Run

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- ArcGIS Pro (ArcPy license required and arcgis env is not included in the project)

### Installation Steps

1. Creat vitual Env
```bash
conda create -n thematicVLM
conda activate thematicVLM
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Locate at the project path
```bash
cd path/to/project
```

### Start Three Services (open three terminals)
**Terminal 1 — GIS Service (ArcPy)**
```bash
conda activate path/to/arcgis
python arcpy_gis_service.py
```

**Terminal 2 — Main Backend (FastAPI + VLM)**
```bash
conda activate thematicVLM
uvicorn app.app:app --reload
```

**Terminal 3 — Frontend (Streamlit)**
```bash
conda activate thematicVLM
streamlit run streamlit_app.py
```

---

## License

This project is for educational, research, and personal learning purposes only.

Commercial use requires an ArcGIS Pro license (ArcPy dependency).

The code itself is under MIT License.

---

*Built for geospatial intelligence.*

Transform thematic maps into actionable geographic insights.
