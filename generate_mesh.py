#!/usr/bin/env python3
"""
Standalone 3D Human Head Mesh Generation Inference Script.
Extracts zero-crossing polygon meshes (.ply / .obj) in ~1 second on Apple Silicon (mps) or CPU
from the trained 300,000-step neural Signed Distance Field weights.
No camera datasets, raymarching, or training dependencies required.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

try:
    import mcubes
except ImportError:
    mcubes = None

try:
    from skimage.measure import marching_cubes as skimage_mcubes
except ImportError:
    skimage_mcubes = None

try:
    import trimesh
except ImportError:
    trimesh = None


# --- Self-Contained Positional Frequency Encoding & SDF Network ---
class Embedder:
    def __init__(self, input_dims=3, num_freqs=6):
        self.embed_fns = [lambda x: x]
        self.out_dim = input_dims
        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += input_dims

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class SDFNetwork(nn.Module):
    """
    8-layer MLP with geometric initialization and skip connections at layer 4,
    mapping 3D coordinate (x, y, z) -> Signed Distance S(x).
    Surface boundary is the zero-level isosurface: S(x) = 0.
    """
    def __init__(self, d_in=3, d_out=257, d_hidden=256, n_layers=8, skip_in=(4,), multires=6):
        super(SDFNetwork, self).__init__()
        self.embedder = Embedder(input_dims=d_in, num_freqs=multires)
        dims = [self.embedder.out_dim] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1] - dims[0] if l + 1 in self.skip_in else dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            lin = nn.utils.weight_norm(lin)
            setattr(self, f"lin{l}", lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        x = self.embedder.embed(inputs)
        h = x
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            if l in self.skip_in:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = lin(h)
            if l < self.num_layers - 2:
                h = self.activation(h)
        return torch.cat([h[:, :1], h[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(
    SCRIPT_DIR,
    "cmu_reference", "exp", "human_face_multiface",
    "gaussian_linearmixture_spatial_black", "checkpoints", "ckpt_300000.pth"
)


def load_sdf_model(checkpoint_path, device):
    """Instantiate SDF Network and load trained tensors."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    sdf_net = SDFNetwork().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'sdf_network_fine' in ckpt:
        sdf_net.load_state_dict(ckpt['sdf_network_fine'])
    else:
        sdf_net.load_state_dict(ckpt)
    sdf_net.eval()
    print("SDF Network loaded successfully.")
    return sdf_net


def extract_mesh(sdf_net, device, resolution=128, threshold=0.0):
    """Queries neural SDF across a 3D coordinate bounding box and extracts polygon surface."""
    bound_min = np.array([-1.01, -1.01, -1.01], dtype=np.float32)
    bound_max = np.array([ 1.01,  1.01,  1.01], dtype=np.float32)

    print(f"Evaluating neural field on {resolution}x{resolution}x{resolution} grid ({resolution**3:,} points) on {device}...")
    t0 = time.time()

    chunk_size = 64
    xs = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(chunk_size)
    ys = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(chunk_size)
    zs = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(chunk_size)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    with torch.no_grad():
        for xi, x_chunk in enumerate(xs):
            for yi, y_chunk in enumerate(ys):
                for zi, z_chunk in enumerate(zs):
                    xx, yy, zz = torch.meshgrid(x_chunk, y_chunk, z_chunk, indexing='ij')
                    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)

                    # Query signed distance
                    sdf_vals = -sdf_net.sdf(pts)
                    chunk_vals = sdf_vals.reshape(len(x_chunk), len(y_chunk), len(z_chunk)).cpu().numpy()
                    u[xi*chunk_size : xi*chunk_size + len(x_chunk),
                      yi*chunk_size : yi*chunk_size + len(y_chunk),
                      zi*chunk_size : zi*chunk_size + len(z_chunk)] = chunk_vals

    t_eval = time.time() - t0
    print(f"GPU forward pass completed in {t_eval:.2f}s.")

    # Marching Cubes
    t_mc = time.time()
    if mcubes is not None:
        vertices, triangles = mcubes.marching_cubes(u, threshold)
    elif skimage_mcubes is not None:
        vertices, triangles, _, _ = skimage_mcubes(u, level=threshold)
    else:
        raise ImportError(
            "Marching Cubes library not found. Please install either:\n"
            "  pip install PyMCubes\n"
            "or (pre-compiled, no C++ toolchain needed):\n"
            "  pip install scikit-image"
        )

    # Rescale vertices back to world coordinates
    b_max_min = bound_max - bound_min
    vertices = vertices / (resolution - 1.0) * b_max_min + bound_min
    print(f"Marching Cubes extracted {len(vertices):,} vertices and {len(triangles):,} faces in {time.time() - t_mc:.2f}s.")

    if trimesh is not None:
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        return mesh
    return vertices, triangles


def main():
    parser = argparse.ArgumentParser(description="Extract 3D Human Head Mesh from Neural SDF Checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to ckpt_300000.pth")
    parser.add_argument("--resolution", type=int, default=128, help="3D grid resolution (64=fast preview, 128=standard, 256=highres)")
    parser.add_argument("--output", type=str, default="human_head_mesh.ply", help="Output path (.ply or .obj)")
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running on device: {device}")

    sdf_net = load_sdf_model(args.checkpoint, device)
    result = extract_mesh(sdf_net, device, resolution=args.resolution)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if hasattr(result, "export"):
        result.export(args.output)
        print(f"Exported 3D mesh successfully to: {args.output}")
    else:
        vertices, triangles = result
        # Write simple PLY
        with open(args.output, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for t in triangles:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        print(f"Exported PLY mesh successfully to: {args.output}")


if __name__ == "__main__":
    main()
