<div align="center">

# 🎯 MLX SAM3 Studio

**Interactive Image Segmentation Application — Built on MLX SAM3**

[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-MLX%20Community-yellow)](https://huggingface.co/mlx-community/sam3-image)

*An interactive web application for creating instance segmentation annotations using SAM3 on Apple Silicon*

> **Note**: This repository is specifically designed for **interactive annotation workflows**. If you're looking for the base MLX SAM3 implementation or just want to use SAM3 programmatically, check out the [original MLX SAM3 repository](https://github.com/Deekshith-Dade/mlx_sam3).

<br>

[![Read the Blog](https://img.shields.io/badge/📖_Read_the_Blog-Understanding_SAM3-FF6B6B?style=for-the-badge)](https://deekshith.me/blog/mlx-sam3)

</div>

---

## ✨ Features

- **🎨 Interactive Web Interface** — Beautiful, responsive UI for real-time segmentation
- **📦 Box Prompts** — Draw bounding boxes to include or exclude regions
- **📍 Point Prompts** — Click points for precise positive/negative guidance
- **🔧 Instance Refinement** — Select instances and refine masks with negative prompts
- **💾 Session Management** — Save and load annotation sessions with automatic image loading
- **📊 COCO Export** — Export annotations in standard COCO format with masks and visualizations
- **🚀 Native Apple Silicon** — Optimized for M1/M2/M3/M4/M5/M6 chips using MLX
- **⬇️ Auto Model Download** — Weights automatically fetched from HuggingFace

---

## 🖼️ Demo

<div align="center">
<table>
<tr>
<td style="width: 50%;"><img src="assets/images/pdish_raw.jpg" alt="Original petri dish image" style="width: 100%; max-width: 500px; height: auto;"></td>
<td style="width: 50%;"><img src="assets/images/pdish_labeled.jpg" alt="Labeled petri dish with 61 segmented instances" style="width: 100%; max-width: 500px; height: auto;"></td>
</tr>
<tr>
<td align="center"><em>Original image</em></td>
<td align="center"><em>61 segmented instances with colored overlays</em></td>
</tr>
</table>

*Instance segmentation example — Petri dish with fungal colonies*

> **Note**: The petri dish annotation above was created in less than 5 minutes using SAM3 Studio. Each of the 61 instances was segmented with just 1 prompt per mask (box or point), and each mask is saved individually as a separate PNG file alongside the COCO-format JSON annotations.

</div>

---

## 📋 Prerequisites

| Requirement       | Version | Notes                                                                                              |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------- |
| **macOS**   | 13.0+   | Apple Silicon required (M1/M2/M3/M4)                                                               |
| **Python**  | 3.13+   | Required for MLX compatibility                                                                     |
| **Node.js** | 18+     | For the web interface                                                                              |
| **uv**      | Latest  | *Optional but recommended* — [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |

> ⚠️ **Apple Silicon Only**: This project uses [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework optimized exclusively for Apple Silicon.

---

## 📦 Dependencies

All Python dependencies are managed in `pyproject.toml`:

- **Core dependencies**: Installed with `pip install -e .` or `uv sync`

  - MLX, NumPy, Pillow, and all SAM3 model dependencies
- **Backend dependencies** (optional): Installed with `pip install -e ".[backend]"` or `uv sync --extra backend`

  - FastAPI, Uvicorn, and other web server dependencies
  - Required only if running the SAM3 Studio web application

> **Note**: The `run.sh` script automatically installs backend dependencies when needed.

---

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed:

```bash
# Clone the repository
git clone https://github.com/RABL1N/mlx-sam3.git
cd mlx-sam3

# Install project dependencies (including backend extras)
uv sync --extra backend

# Launch the app (backend + frontend)
cd app && ./run.sh
```

The first run will automatically download MLX weights from [mlx-community/sam3-image](https://huggingface.co/mlx-community/sam3-image) (~3.5GB).

**Access the app:**

- 🌐 **Frontend**: http://localhost:3000
- 🔌 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop all servers.

---

### Option 2: Manual Setup (Standard pip)

<details>
<summary><strong>Click to expand manual setup instructions</strong></summary>

#### 1. Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/mlx-sam3.git
cd mlx-sam3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package with backend dependencies (required for web app)
pip install -e ".[backend]"
```

#### 2. Start the Backend

```bash
# Start the backend server
cd app/backend
python main.py
```

The backend will start on http://localhost:8000

#### 3. Start the Frontend (new terminal)

```bash
cd app/frontend
npm install
npm run dev
```

The frontend will start on http://localhost:3000

</details>

---

## 🐍 Python API

> **Note**: The Python API is available for advanced use cases, but the primary focus of this repository is the interactive web application. For programmatic SAM3 usage, consider the [original MLX SAM3 implementation](https://github.com/Deekshith-Dade/mlx_sam3).

You can also use SAM3 directly in your Python scripts:

```python
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model (auto-downloads MLX weights from mlx-community/sam3-image)
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.5)

# Load and process an image
image = Image.open("your_image.jpg")
state = processor.set_image(image)

# Segment with text prompt
state = processor.set_text_prompt("person", state)

# Access results
masks = state["masks"]       # Binary segmentation masks
boxes = state["boxes"]       # Bounding boxes [x0, y0, x1, y1]
scores = state["scores"]     # Confidence scores

print(f"Found {len(scores)} objects")
```

### Adding Box and Point Prompts

```python
# Add a box prompt (normalized coordinates: center_x, center_y, width, height)
# label=True for inclusion, label=False for exclusion
state = processor.add_geometric_prompt(
    box=[0.5, 0.5, 0.3, 0.3],  # Center of image, 30% width/height
    label=True,
    state=state
)

# Add a point prompt (normalized coordinates: [x, y] in [0, 1])
# label=True for positive, label=False for negative
state = processor.add_point_prompt(
    point=[0.5, 0.5],  # Center of image
    label=True,
    state=state
)

# Refine an existing instance with negative prompts
# First select an instance, then use negative prompts to exclude regions
state = processor.add_point_prompt(
    point=[0.6, 0.6],
    label=False,  # Negative prompt
    selected_instance_index=0,  # Refine instance 0
    state=state
)
```

### Reset and Try New Prompts

```python
# Clear all prompts while keeping the image
processor.reset_all_prompts(state)

# Try a different prompt
state = processor.set_text_prompt("car", state)
```

---

## 🏗️ Project Structure

```
mlx-sam3/
├── sam3/                    # Core MLX SAM3 implementation
│   ├── model/               # Model components
│   │   ├── sam3_image.py    # Main model architecture
│   │   ├── vitdet.py        # Vision Transformer backbone
│   │   ├── text_encoder_ve.py # Text encoder
│   │   └── ...
│   ├── model_builder.py     # Model construction utilities
│   ├── convert.py           # Weight conversion from PyTorch
│   └── utils.py             # Helper utilities
├── app/                     # Web application
│   ├── backend/             # FastAPI server
│   ├── frontend/            # Next.js React app
│   └── run.sh               # One-command launcher
├── assets/                  # Static assets & test images
├── annotations/             # Saved annotation sessions
│   └── session_{timestamp}/ # Session directories (COCO JSON, masks, visualizations)
├── examples/                # Jupyter notebook examples
└── pyproject.toml           # Project configuration (all dependencies)
```

---

## 🔌 API Reference

| Endpoint              | Method | Description                               |
| --------------------- | ------ | ----------------------------------------- |
| `/health`           | GET    | Check if the model is loaded and ready    |
| `/upload`           | POST   | Upload an image and create a session      |
| `/segment/text`     | POST   | Segment using a text prompt               |
| `/segment/box`      | POST   | Add a box prompt (include/exclude)        |
| `/segment/point`    | POST   | Add a point prompt (include/exclude)      |
| `/reset`            | POST   | Clear all prompts and masks for a session |
| `/save-annotations` | POST   | Save annotations in COCO format           |
| `/load-session`     | POST   | Load a previously saved session           |
| `/list-sessions`    | GET    | List all available annotation sessions    |
| `/update-category`  | POST   | Update category label for an instance     |
| `/remove-instance`  | POST   | Remove a specific instance                |
| `/confidence`       | POST   | Update confidence threshold               |
| `/session/{id}`     | DELETE | Delete a session and free memory          |

### Example API Call

```bash
# Upload an image
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_image.jpg"

# Response: {"session_id": "abc-123", "width": 1920, "height": 1080, ...}

# Segment with point prompt
curl -X POST "http://localhost:8000/segment/point" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "point": [0.5, 0.5], "label": true}'

# Save annotations
curl -X POST "http://localhost:8000/save-annotations" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "image_id": 1}'
```

---

## 🎯 Using SAM3 Studio

SAM3 Studio is designed for creating instance segmentation annotations through an interactive web interface:

1. **Start the application:**

   ```bash
   cd app && ./run.sh
   ```
2. **Open in browser:** Navigate to `http://localhost:3000`
3. **Create annotations:**

   - Upload an image or load a previous session
   - Use **box prompts** (draw boxes) or **point prompts** (click points) to segment instances
   - Select an instance to refine it with negative prompts
   - Click "Save Session" to export annotations in COCO format
4. **Output:** Annotations are saved to `annotations/session_{timestamp}/` with:

   - COCO JSON file with all instances
   - Individual mask PNGs for each instance
   - Visualization images for each instance
   - Original image

### Annotation Format

Annotations follow the COCO format with additional fields:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": null,
  "segmentation": {
    "counts": [804394, 10, 1007, 20, ...],
    "size": [982, 1023]
  },
  "bbox": [270.86, 785.04, 354.05, 870.36],
  "area": 5430,
  "iscrowd": 0,
  "score": 0.782,
  "instance_id": 0
}
```

---

## 📓 Examples

Jupyter notebooks are available in the `examples/` directory:

- **`sam3_image_predictor_example.ipynb`** — Basic image segmentation
- **`sam3_image_interactive.ipynb`** — Interactive prompting workflows

Run them with:

```bash
cd examples
jupyter notebook
```

---

## 🛠️ Tech Stack

| Component              | Technology                                               |
| ---------------------- | -------------------------------------------------------- |
| **ML Framework** | [MLX](https://github.com/ml-explore/mlx)                    |
| **Backend**      | FastAPI, Uvicorn                                         |
| **Frontend**     | Next.js 16, React 19, Tailwind CSS 4                     |
| **Model**        | [SAM3 MLX](https://huggingface.co/mlx-community/sam3-image) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for the original SAM3 model
- [MLX Team](https://github.com/ml-explore/mlx) at Apple for the incredible ML framework
- [Deekshith Dade](https://github.com/Deekshith-Dade) for the original MLX SAM3 implementation
- The open-source community for continuous inspiration

**Original MLX SAM3 Implementation:**

- Hugging Face: https://huggingface.co/mlx-community/sam3-image
- GitHub: https://github.com/Deekshith-Dade/mlx_sam3
- Blog: https://deekshith.me/blog/mlx-sam3

---

<div align="center">

**Built with ❤️ for Apple Silicon**

[Report Bug](https://github.com/your-username/mlx-sam3/issues) · [Request Feature](https://github.com/your-username/mlx-sam3/issues)

</div>
