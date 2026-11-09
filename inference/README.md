# 3D Neural SDF Inference & Mesh Generation

This directory contains the self-contained inference pipeline for extracting 3D polygon meshes (`.ply`) on the fly from trained neural Signed Distance Field (SDF) tensors on Apple Silicon (`mps`).

---

## 📦 Directory Manifest

| File | Size | Description |
| :--- | :--- | :--- |
| [`ficus_model_300k.pth`](file:///Users/chalaserver/git/NeuralRenderingMPSChala/inference/ficus_model_300k.pth) | **9.2 MB** | Trained neural network checkpoint (300,000 steps). Encodes the complete 3D geometry of the Ficus tree. |
| [`generate_mesh.py`](file:///Users/chalaserver/git/NeuralRenderingMPSChala/inference/generate_mesh.py) | **5.1 KB** | Standalone inference script. Evaluates the neural field and runs Marching Cubes. |
| [`sample_ficus.ply`](file:///Users/chalaserver/git/NeuralRenderingMPSChala/inference/sample_ficus.ply) | **386 KB** | Extracted 3D polygon mesh sample (10,748 vertices, 20,516 faces). |
| [`sample_ficus_preview.png`](file:///Users/chalaserver/git/NeuralRenderingMPSChala/inference/sample_ficus_preview.png) | **140 KB** | Shaded visual preview of the extracted 3D mesh. |

---

## ⚡ Quick Start

Run the standalone inference script to generate a 3D polygon mesh from the trained weights:

```bash
```bash
# If running from project root (NeuralRenderingMPSChala/):
python3 inference/generate_mesh.py --resolution 128 --output inference/sample_ficus.ply

# If already inside the inference/ folder:
python3 generate_mesh.py --resolution 128 --output sample_ficus.ply
```

### Options & Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `inference/ficus_model_300k.pth` | Path to the trained `.pth` weight file. |
| `--resolution` | `128` | 3D grid resolution for Marching Cubes ($N \times N \times N$). |
| `--output` | `inference/sample_ficus.ply` | Destination path for the generated `.ply` mesh. |

### Resolution Presets

**From inside the `inference/` folder:**
```bash
# Fast low-poly preview (64^3 grid, ~0.2s, 3.6k faces)
python3 generate_mesh.py --resolution 64 --output ficus_64.ply

# Standard high-detail (128^3 grid, ~1.1s, 20.5k faces)
python3 generate_mesh.py --resolution 128 --output ficus_128.ply

# Ultra-fine micro-leaf mesh (256^3 grid, ~6.5s, ~80k faces)
python3 generate_mesh.py --resolution 256 --output ficus_256.ply
```

**From the project root (`NeuralRenderingMPSChala/`):**
```bash
python3 inference/generate_mesh.py --resolution 256 --output inference/ficus_256.ply
```

---

## 🧠 How Inference Works (The Math)

Unlike traditional 3D graphics that store explicit triangle vertices, the neural network stores an implicit continuous mathematical field:

$$f_\theta: (x, y, z) \longrightarrow \text{Signed Distance } S(\mathbf{x})$$

- Points **inside** the solid tree/leaf: $S(\mathbf{x}) < 0$
- Points in **empty air**: $S(\mathbf{x}) > 0$
- The physical **surface boundary**: The zero-level isosurface $S(\mathbf{x}) = 0$

```
[ficus_model_300k.pth] (9.2 MB)
         │
         ▼ Loads weights into 8-layer MLP
   [SDFNetwork]
         │
         ▼ Evaluates 3D grid points on GPU (MPS)
   [Grid of S(x)]
         │
         ▼ Finds zero-crossings S(x) = 0
 [Marching Cubes]
         │
         ▼ Generates triangle vertices & faces
  [sample_ficus.ply] (Instant 3D Mesh in 1.17s)
```

During inference, **no images, camera matrices, or ray marching are required**. The script generates a 3D coordinate bounding box $[-1.01, 1.01]^3$, runs a single vectorized forward pass through the trained MLP, and uses `PyMCubes` to extract the zero-crossing polygon surface.

---

## 🚀 Performance Benchmarks (Apple Silicon M4 Pro)

| Resolution ($N^3$) | Total Query Points | GPU Forward Pass | Marching Cubes | Total Latency | Triangles / Faces |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$64^3$** | 262,144 | 0.18 s | 0.01 s | **0.19 s** | 3,684 |
| **$128^3$** | 2,097,152 | 1.15 s | 0.02 s | **1.17 s** | 20,516 |
| **$256^3$** | 16,777,216 | 6.20 s | 0.15 s | **6.35 s** | ~82,000 |

---

## 🎨 Viewing & Exporting

The generated `.ply` polygon meshes are standard 3D files compatible with:
- **macOS Quick Look:** Select `sample_ficus.ply` in Finder and press **Spacebar** for native 3D interactive viewing.
- **Blender / Maya:** `File > Import > Stanford (.ply)`
- **MeshLab / CloudCompare:** Full point-cloud and triangle surface inspection.
- **Unreal Engine 5 / Unity:** Direct asset import for physics colliders, lighting, and real-time rendering.
