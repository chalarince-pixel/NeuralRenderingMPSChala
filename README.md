# Iteration 2: CMU Volumetric SDF & Accelerated Hash-SDF Pipeline

Replication of Carnegie Mellon University's CVPR formulation (*"Objects as volumes: A stochastic geometry view of opaque solids"*) and accelerated multi-resolution Hash-SDF with a native Apple Silicon Metal viewer.

---

## ⚡ Instant 3D Mesh Generation (Fast Inference)

Extract the complete 3D polygon mesh from the trained 300,000-step checkpoint (`ficus_model_300k.pth`) in **~1.1s** on Apple Silicon (`mps`) or CPU without downloading raw datasets or running raymarching:

```bash
# Standard 3D polygon mesh extraction (128^3 resolution, ~1.1s):
python3 inference/generate_mesh.py --resolution 128 --output ficus_mesh.ply

# Ultra-fine micro-leaf mesh (256^3 resolution, ~6.3s):
python3 inference/generate_mesh.py --resolution 256 --output ficus_256.ply

# Real-time 120 FPS Native Apple Metal Viewer (Swift):
swift inference/apple_metal_viewer.swift ficus_mesh.ply
```

---

## ⚡ Key Highlights

### 1. CMU / NeuS Surface-Finding SDF Pipeline
Formulation for opaque solids:
- Ingests **all 100 camera views** at high resolution ($16,000,000$ high-res rays).
- **Surface-Finding Point Sampler (`engine/cmu_sampler.py`):** Dynamically locates the zero-crossing surface ($S(\mathbf{x}) = 0$) and concentrates dense samples on thin boundary details.
- **Binary Silhouette Mask Loss:** Strict foreground/background constraint using ground-truth alpha masks.
- **Eikonal Surface Regularization:** Enforces $||\nabla S|| = 1$ to eliminate empty-space fog.

```bash
# Train on all 100 views with surface-finding point sampler:
python3 train_cmu_pipeline.py --scene ficus --resolution 400 --views 100 --steps 3000 --batch_size 2048
```

### 2. State-of-the-Art Hash-Encoded SDF (Instant-NGP)
Fast 50-second convergence with multi-resolution hash grid:

```bash
python3 train_hash_sdf.py --scene ficus --size 120 --steps 1200
```

### 3. Standalone Mesh Generation & Apple Metal 3D Viewer
Extract polygon meshes on the fly from neural field weights using vectorized Marching Cubes on MPS:

```bash
# Generate high-resolution 3D mesh:
python3 inference/generate_mesh.py --resolution 128 --output inference/sample_ficus.ply

# Launch native Apple Metal viewer (Swift):
swift inference/apple_metal_viewer.swift
```

---

## 📦 Directory Structure

```
.
├── checkpoints/              # Checkpoint safetensors (cmu_ficus_model, ficus_hash_sdf)
├── cmu_reference/            # Original CMU paper reference codebase
│   ├── confs/                # PyHocon experiment configurations
│   ├── exp_runner.py         # Paper experiment runner
│   └── models/               # Attenuation coefficients, samplers, and renderers
├── datasets/
│   ├── cmu_dataset.py        # 100-view high-res ray sampler
│   └── download_ficus.py     # Automated dataset fetcher
├── engine/
│   └── cmu_sampler.py        # Surface-finding 0-crossing sampler
├── exports/                  # Extracted .ply 3D meshes
├── inference/
│   ├── apple_metal_viewer.swift # Native macOS Metal 3D viewer
│   ├── ficus_model_300k.pth     # 300k-step trained neural SDF weights
│   ├── generate_mesh.py         # Standalone Marching Cubes extractor
│   └── server.py                # Web Three.js mesh inspector
├── models/
│   ├── hash_encoder.py       # Multi-resolution hash table
│   └── sdf_backbone.py       # Implicit surface neural field
├── renders/                  # Orbit animations, mesh evolutions, normal maps
├── render_mesh_timelapse.py  # Surface evolution video generator
├── train_cmu_pipeline.py     # 100-view CMU opaque solid trainer
└── train_hash_sdf.py         # Accelerated Instant-NGP trainer
```
