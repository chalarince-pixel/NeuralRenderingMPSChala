# 📘 Client Delivery & Deployment Guide (`instructionForUser.md`)

This guide explains how to transfer, install, and run the self-contained 3D Neural SDF Mesh Generator on any client machine.

---

## 1. What Is In This Client Bundle?

Everything needed to run inference is inside this single folder (**Total size: ~9.3 MB**):

```
inference/
├── ficus_model_300k.pth       # (9.2 MB) The trained neural model weights
├── generate_mesh.py           # Self-contained CLI generator script
├── server.py                  # Standalone FastAPI REST microservice
├── requirements.txt           # Minimal client Python dependencies
├── instructionForUser.md      # This deployment manual
├── README.md                  # Quick reference documentation
├── sample_ficus.ply           # Reference output mesh (128^3 grid, 20.5k faces)
└── ficus_256.ply              # Ultra-high-res output mesh (256^3 grid, 94.8k faces)
```

> [!NOTE]
> **Zero Training Dependencies:** This bundle does **not** need the 400 training photos, camera matrices, ray marchers, or OpenCV. It only evaluates the trained 9.2 MB weight file.

---

## 2. Hardware & OS Compatibility

The engine automatically selects the best available compute backend:

| Client Machine | Backend Used | Latency (128³ Grid) | Setup Required |
| :--- | :--- | :---: | :--- |
| **Apple Silicon Mac (M1/M2/M3/M4)** | **Metal (`mps`)** | **~1.1 s** | None (Built into macOS PyTorch) |
| **Linux / Windows PC with NVIDIA** | **CUDA** | **< 0.5 s** | Standard NVIDIA driver |
| **Standard Laptop / CPU Server** | **CPU** | **~3.5 s** | None (Runs everywhere) |

---

## 3. Client Installation (1 Minute)

On the client machine:

```bash
# 1. Open terminal inside the inference folder:
cd inference

# 2. (Recommended) Create a fresh Python virtual environment:
python3 -m venv venv
source venv/bin/activate

# 3. Install the 4 lightweight dependencies:
pip install -r requirements.txt
```

---

## 4. How to Run: Three Orchestrator Modes

### Mode A: Command-Line Interface (CLI)

Use `generate_mesh.py` for batch automation or local scripts:

```bash
# Standard 3D mesh (128^3 grid, ~1.1s, 20k triangles):
python3 generate_mesh.py --resolution 128 --output ficus_mesh.ply

# Ultra-fine micro-leaf detail (256^3 grid, ~9.0s, 95k triangles):
python3 generate_mesh.py --resolution 256 --output ficus_ultra.ply

# Fast low-poly preview (64^3 grid, ~0.2s, 3.6k triangles):
python3 generate_mesh.py --resolution 64 --output ficus_lowpoly.ply
```

#### CLI Parameters:
- `--checkpoint`: Path to the `.pth` weights (default: `ficus_model_300k.pth`).
- `--resolution`: 3D sampling density. Choose `64` (fast), `128` (standard), or `256` (ultra).
- `--output`: Output filepath ending in `.ply`.

---

### Mode B: REST API Microservice (Web & App Integration)

To let remote clients, web browsers, or game engines query 3D meshes over HTTP:

1. **Launch the server:**
   ```bash
   python3 server.py
   ```
   *The server starts on `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.*

2. **Download a 3D mesh via cURL:**
   ```bash
   # Request standard 128-resolution mesh:
   curl -o my_tree.ply "http://localhost:8000/generate-mesh?resolution=128"

   # Request ultra-detail 256-resolution mesh:
   curl -o my_ultra_tree.ply "http://localhost:8000/generate-mesh?resolution=256"
   ```

---

### Mode C: Embedded Python Library

Call the generator directly inside any Python application:

```python
import torch
from generate_mesh import load_sdf_model, extract_mesh_on_the_fly

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = load_sdf_model('ficus_model_300k.pth', device)

# Generate trimesh object in memory in 1.1s
mesh = extract_mesh_on_the_fly(model, device, resolution=128)

print(f"Generated {len(mesh.faces)} faces!")
mesh.export('output.ply')
```

---

## 5. Native Apple Graphics & Metal 120 FPS Rendering

We provide a compiled native Apple Silicon Metal application in this folder:

### Option 1: Native Apple Metal 3D Viewer (`apple_metal_viewer`)
Runs directly on the Apple Metal rasterizer with 3-point studio lighting, PBR materials, and real-time performance statistics at 120 FPS:

```bash
# View the standard mesh:
./apple_metal_viewer sample_ficus.obj

# View the 1.6-Million face ultra-mesh:
./apple_metal_viewer ficus_1024.obj
```
- **Orbit Camera:** Click and drag with mouse or trackpad.
- **Pan Camera:** Two-finger drag.
- **Zoom:** Pinch or scroll wheel.

### Option 2: macOS Native Quick Look (Zero Tools Needed)
Select `sample_ficus.obj` or `ficus_1024.obj` in macOS Finder and press the **Spacebar**. macOS will instantly launch its native Metal-accelerated 3D Quick Look inspector.

### Option 3: Professional 3D & Game Engines
- **Blender / Maya:** `File > Import > Wavefront (.obj)` or `Stanford (.ply)`.
- **Unreal Engine 5 / Unity:** Drag and drop the `.obj` into the Content Browser for Nanite/Lumen lighting.

---

## 6. Packaging for Client Delivery (Creating a Zip)

To send the bundle to a client, zip the directory:

```bash
cd ..
zip -r ficus_client_package.zip inference/ -x "*.git*" "*__pycache__*"
```

The resulting `ficus_client_package.zip` is **~10 MB** and ready to deploy on any machine.
