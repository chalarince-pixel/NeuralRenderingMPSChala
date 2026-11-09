# Iteration 1: Neural Rendering & Diffusion MPS Baseline

A high-performance PyTorch baseline connecting **Neural Radiance Fields (Neural Rendering)** and **Diffusion Models / LoRA** on Apple Silicon (`mps`). Built from scratch for the Apple M4 Pro (24 GB Unified Memory).

---

## ⚡ Instant 3D Geometry Extraction (Fast Inference)

Extract colored 3D point cloud surface geometry on the fly from `.safetensors` checkpoints in **~0.5s** on Apple Silicon (`mps`) or CPU:

```bash
# Extract 3D geometry from base model safetensors:
python3 extract_mesh.py --checkpoint checkpoints/base_model.safetensors --output exports/scene_geometry.ply

# Extract 3D geometry with LoRA fine-tuning adapter applied:
python3 extract_mesh.py --checkpoint checkpoints/base_model.safetensors --lora checkpoints/lora_adapter.safetensors --output exports/lora_geometry.ply
```

---

## ⚡ Architecture Highlights

- **From-Scratch LoRA Engine (`models/lora.py`)**: Implements low-rank decomposition $W_0 + \frac{\alpha}{r}BA$ with zero-copy `.safetensors` serialization.
- **Unified Backbone (`models/unified_backbone.py`)**: Positional frequency encoding, FiLM conditioning, and residual MLP heads.
- **Differentiable Volume Raymarcher (`engine/volume_renderer.py`)**: Optical depth numerical quadrature with full autograd backpropagation on MPS.
- **Graph & Kernel Debugger (`engine/graph_debugger.py`)**: MPS memory tracking, kernel timing benchmarks, and autograd validation.
- **Turntable Video & 3D Exporter (`render_video.py`)**: Renders full 360° `.mp4` video flythroughs and extracts `.ply` 3D point clouds.

---

## 🚀 Training & Execution

### 1. Procedural Baseline & LoRA Fine-Tuning
Demonstrates LoRA adaptation (Summer foliage $\to$ Autumn foliage) on Apple Silicon MPS:

```bash
python3 train_baseline.py
```

### 2. Multi-View Photo NeRF Training
Direct multi-view photo training with standard 8-layer MLP NeRF:

```bash
python3 train_from_images.py
```

### 3. Orbit Video Rendering
Generate turntable 360° flythroughs from trained checkpoints:

```bash
python3 render_video.py
```

---

## 📦 Directory Structure

```
.
├── checkpoints/              # Base model and LoRA adapter weights (.safetensors)
├── data/nerf_synthetic/      # Synthetic test scene datasets (lego, ficus)
├── datasets/
│   ├── synthetic_generator.py # Procedural 3D scene synthesizer
│   └── nerf_image_dataset.py  # Multi-view NeRF camera & ray loader
├── engine/
│   ├── volume_renderer.py    # Differentiable ray marching & quadrature
│   └── graph_debugger.py     # Apple Silicon MPS memory & timing profiler
├── models/
│   ├── unified_backbone.py   # Harmonic positional encoding + MLP
│   └── lora.py               # Rank decomposition adapter engine
├── renders/                  # Turntable MP4s, GIFs, and depth passes
├── render_video.py           # Turntable exporter
├── train_baseline.py         # Main training script
└── train_from_images.py      # Real image training script
```
