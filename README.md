# ⚡ NeuralRenderingMPSChala
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20Linux%20%7C%20Windows%20%7C%20Raspberry%20Pi-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20(MPS%2FCUDA%2FCPU)-red.svg)](#)
[![Inference Latency](https://img.shields.io/badge/3D%20Mesh%20Extraction-~1.1s-brightgreen.svg)](#)
[![Branches](https://img.shields.io/badge/Architecture-3%20Independent%20Branches-orange.svg)](#)
> **An exploration of Neural Radiance Fields (NeRF), Diffusion LoRA engines, and Stochastic Geometry Signed Distance Fields (SDF) built natively for Apple Silicon Unified Memory and cross-platform edge inference.**
---

---

## 🧭 Repository Architecture & Branches

To maintain clean separation between research iterations, **the `main` branch contains no code**. The codebase is organized into **three independent standalone branches**, each representing an evolutionary phase of the research:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Branch: v1-baseline-nerf-mps                                                │
│ Initial from-scratch PyTorch NeRF baseline + LoRA adaptation on MPS         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Branch: v2-cmu-volumetric-sdf                                               │
│ CMU CVPR "Objects as Volumes" reproduction, Instant-NGP Hash SDF,          │
│ Marching Cubes mesh extraction & Native Apple Metal 3D Viewer               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Branch: v3-human-multiface                                                  │
│ Meta Multiface 38-camera dome dataset pipeline & high-fidelity 3D human     │
│ head/body surface reconstruction                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Universal Cross-Platform 3D Mesh Extraction (Cheapest Inference)

Every branch includes standalone inference scripts allowing anyone who forks the repo to extract full 3D polygon meshes directly from trained weights on **macOS (Apple Silicon / Intel), Linux (CUDA / CPU), Windows (CUDA / CPU), and Raspberry Pi OS (ARM64)** in **~1 second** without needing camera datasets, raw images, or training libraries:

```bash
# 🌿 Branch 1: Extract 3D Geometry from NeRF/LoRA safetensors
git checkout v1-baseline-nerf-mps
python3 extract_mesh.py --checkpoint checkpoints/base_model.safetensors --output exports/scene.ply

# 🌳 Branch 2: Extract 3D Ficus Tree Mesh from 300k SDF weights
git checkout v2-cmu-volumetric-sdf
python3 inference/generate_mesh.py --resolution 128 --output ficus_mesh.ply

# 👤 Branch 3: Extract 3D Human Head Mesh from 300k Meta Dome weights
git checkout v3-human-multiface
python3 generate_mesh.py --resolution 128 --output human_head.ply

# 🍓 Running on Raspberry Pi OS / Low-Memory Systems (<40MB RAM):
python3 generate_mesh.py --resolution 64 --chunk_size 32 --device cpu --output mesh.ply
```

### 📦 Minimal Cross-Platform Dependencies (Inference Only)
No C++ compiler or heavy dependencies needed:
```bash
pip install torch scikit-image safetensors
# Optional (for OBJ format export):
pip install trimesh
```

---

## 🚀 Branch Quick Access

### 1. [`v1-baseline-nerf-mps`](https://github.com/chalarince-pixel/NeuralRenderingMPSChala/tree/v1-baseline-nerf-mps)
* **Scope:** Custom from-scratch PyTorch implementation connecting Neural Radiance Fields and Low-Rank Adaptation (LoRA).
* **Key Features:**
  * From-scratch low-rank decomposition engine ($W_0 + \frac{\alpha}{r}BA$) with zero-copy `.safetensors`.
  * Unified backbone with frequency positional encoding and FiLM conditioning.
  * Differentiable volume raymarcher with optical depth numerical quadrature.
  * Native MPS memory and kernel timing debugger.
* **Checkout:**
  ```bash
  git checkout v1-baseline-nerf-mps
  ```

---

### 2. [`v2-cmu-volumetric-sdf`](https://github.com/chalarince-pixel/NeuralRenderingMPSChala/tree/v2-cmu-volumetric-sdf)
* **Scope:** Reproduction of Carnegie Mellon University's CVPR paper *"Objects as volumes: A stochastic geometry view of opaque solids"* on synthetic opaque solids (`ficus`), accelerated with Instant-NGP Hash SDF.
* **Key Features:**
  * Full 100-view high-resolution surface-finding point sampler locating $S(\mathbf{x}) = 0$.
  * Eikonal surface regularization $||\nabla S|| = 1$ and binary silhouette mask supervision.
  * Multi-resolution hash grid encoding converging in under 50 seconds on MPS.
  * Standalone Marching Cubes mesh extraction and native Swift Apple Metal 3D viewer.
* **Checkout:**
  ```bash
  git checkout v2-cmu-volumetric-sdf
  ```

---

### 3. [`v3-human-multiface`](https://github.com/chalarince-pixel/NeuralRenderingMPSChala/tree/v3-human-multiface)
* **Scope:** Extending stochastic geometry volume rendering to calibrated multi-camera human dome capture (Meta Multiface dataset).
* **Key Features:**
  * Automated dome camera calibration parser converting 38 KRT camera angles into normalized unit-sphere projection matrices.
  * Human face & body Signed Distance Field training pipeline (Gaussian implicit distribution + linear mixture normal distribution with spatial anisotropy network).
  * Reconstructed 300,000-step 3D human head polygon meshes, surface normals, and depth maps.
* **Checkout:**
  ```bash
  git checkout v3-human-multiface
  ```

---

## 💻 Hardware Requirements & Prerequisites

* **Hardware:** Apple Silicon Mac (M1/M2/M3/M4, Pro/Max/Ultra recommended with $\ge 16\text{ GB}$ Unified Memory).
* **Environment:** macOS 14.0+, Python 3.10+, PyTorch 2.1+ with native `mps` backend support.
* **Dependencies:** `pip install torch torchvision safetensors trimesh pymcubes opencv-python tqdm pyhocon`
