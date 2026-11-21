# Iteration 3: 3D Human Reconstruction from Calibrated Multi-Camera Dome

Extension of stochastic geometry neural volume rendering to calibrated multi-view human head & body captures using the Meta Multiface dome dataset.

---

## ⚡ Instant 3D Mesh Generation (Cross-Platform Inference)

Extract the complete 3D polygon mesh from the trained neural SDF checkpoint (`checkpoints/human_head_model.safetensors` or `ckpt_300000.pth`) on **macOS, Linux, Windows, or Raspberry Pi OS** without downloading raw dome datasets or running raymarching:

```bash
# Standard extraction using zero-copy .safetensors (default):
python3 generate_mesh.py --resolution 128 --output human_head_mesh.ply

# Explicitly pass .safetensors checkpoint:
python3 generate_mesh.py --checkpoint checkpoints/human_head_model.safetensors --resolution 128 --output human_head.ply

# macOS (Apple Silicon MPS - ~0.4s forward pass):
python3 generate_mesh.py --resolution 128 --output human_head_mesh.ply

# Linux / Windows (NVIDIA CUDA GPU - ~0.3s):
python3 generate_mesh.py --resolution 128 --output human_head_mesh.ply

# Raspberry Pi OS / Low-Memory CPU (using chunking, <40MB RAM):
python3 generate_mesh.py --resolution 64 --chunk_size 32 --device cpu --output human_head_mesh.ply
```

### 📦 Minimal Cross-Platform Dependencies
Only 3 packages required (zero C++ compilation needed):
```bash
pip install torch safetensors scikit-image
# (Optional for OBJ format export):
pip install trimesh
```

---

## ⚡ Key Highlights

### 1. Multi-Camera Dome Calibration & Normalization (`prepare_multiface.py`)
- Automatically ingests calibrated camera intrinsic and extrinsic parameters ($K, [R|t]$) from 38 dome angles.
- Computes multi-camera focal intersection center and scales head boundaries to the unit sphere $[-1, 1]^3$.
- Synchronizes viewpoint images and generates foreground alpha masks for neural rendering supervision.

```bash
python3 cmu_reference/prepare_multiface.py
```

### 2. Neural Signed Distance Field Training on Human Capture
- **Implicit Field:** Gaussian implicit distribution for sharp surface boundaries.
- **Normal Distribution:** Linear mixture normal distribution with learned spatial anisotropy.
- **Silhouette Supervision:** Intersecting ray-cast interval sampling with Eikonal regularization ($||\nabla S|| = 1$).

```bash
# Run training on human face multiface dataset:
python3 cmu_reference/exp_runner.py --conf cmu_reference/confs/human_face_multiface.conf --mode train
```

### 3. Reconstructed 3D Geometry
- **Checkpoints & Meshes:** High-fidelity 3D polygon mesh reconstructed at 300,000 training steps (`cmu_reference/exp/human_face_multiface/gaussian_linearmixture_spatial_black/meshes/00300000.ply`).
- Validated with fine-detail surface normal maps and depth sweeps across all 38 calibrated viewpoints.

---

## 📦 Directory Structure

```
.
├── checkpoints/
│   └── human_head_model.safetensors # 3.1 MB (Fast zero-copy neural SDF weights)
├── generate_mesh.py                 # Universal cross-platform mesh generator
├── requirements.txt                 # Minimal inference dependencies
├── cmu_reference/
│   ├── confs/
│   │   └── human_face_multiface.conf # PyHocon config for 38-view dome training
│   ├── exp/
│   │   └── human_face_multiface/     # Trained 300k human head mesh & validations
│   │       └── gaussian_linearmixture_spatial_black/
│   │           ├── checkpoints/
│   │           │   └── ckpt_300000.pth # 9.2 MB (Original PyTorch checkpoint)
│           ├── depth/             # Viewpoint depth maps
│           ├── meshes/            # 300k step reconstructed 3D polygon mesh (.ply)
│           ├── normals/           # Viewpoint surface normal maps
│           └── validations_fine/  # High-fidelity RGB renders
├── exp_runner.py                  # Differentiable training and validation runner
├── models/
│   ├── attenuation_coefficient.py # Gaussian & linear mixture formulations
│   ├── dataset.py                 # Multi-view dome loader
│   ├── fields.py                  # 8-layer SDF network + anisotropy network
│   ├── renderer.py                # Volumetric raymarcher with optical depth
│   └── sampler.py                 # Foreground/Surface/Background ray sampler
├── multiface_tools/               # Multi-camera dome dataset extraction tools
├── prepare_multiface.py           # KRT parsing & coordinate sphere projection
└── public_data/
    └── human_face_multiface/
        ├── cameras_sphere.npz     # Normalized 38-view projection matrices
        └── mask/                  # 38-view foreground segmentation masks
```
