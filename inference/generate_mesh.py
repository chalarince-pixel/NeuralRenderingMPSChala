#!/usr/bin/env python3
"""
Self-contained 3D Mesh Generation Inference Script.
Loads trained neural SDF tensors and extracts on-the-fly .ply polygon meshes.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import mcubes
import trimesh

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import mcubes
import trimesh

# --- Self-Contained Positional Encoding & SDF Network ---
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

def load_sdf_model(checkpoint_path, device):
    """Instantiate SDF Network and load trained tensors."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    sdf_net = SDFNetwork().to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sdf_net.load_state_dict(ckpt['sdf_network_fine'])
    sdf_net.eval()
    print("SDF Network loaded successfully into memory.")
    return sdf_net

def extract_mesh_on_the_fly(sdf_net, device, resolution=128, threshold=0.0):
    """Queries neural SDF across a 3D coordinate grid and runs Marching Cubes."""
    bound_min = np.array([-1.01, -1.01, -1.01], dtype=np.float32)
    bound_max = np.array([ 1.01,  1.01,  1.01], dtype=np.float32)
    
    print(f"Generating 3D grid at resolution {resolution}x{resolution}x{resolution} ({resolution**3:,} query points)...")
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
                    
                    # Forward pass through trained geometry tensors
                    sdf_vals = -sdf_net.sdf(pts)
                    
                    chunk_vals = sdf_vals.reshape(len(x_chunk), len(y_chunk), len(z_chunk)).cpu().numpy()
                    u[xi*chunk_size : xi*chunk_size + len(x_chunk),
                      yi*chunk_size : yi*chunk_size + len(y_chunk),
                      zi*chunk_size : zi*chunk_size + len(z_chunk)] = chunk_vals

    query_time = time.time() - t0
    print(f"Grid inference completed in {query_time:.2f} seconds.")
    
    print("Running Marching Cubes zero-level isosurface extraction...")
    t1 = time.time()
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    
    # Scale vertices from grid indices back to metric 3D world space
    vertices = vertices / (resolution - 1.0) * (bound_max - bound_min)[None, :] + bound_min[None, :]
    mc_time = time.time() - t1
    print(f"Marching Cubes completed in {mc_time:.2f} seconds.")
    
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Polygon Mesh On-the-Fly from Trained Tensors")
    parser.add_argument('--checkpoint', type=str, 
                        default=os.path.join(SCRIPT_DIR, 'ficus_model_300k.pth'),
                        help="Path to trained .pth checkpoint")
    parser.add_argument('--resolution', type=int, default=128, 
                        help="3D grid resolution for Marching Cubes (e.g., 64, 128, 256)")
    parser.add_argument('--output', type=str, 
                        default=os.path.join(SCRIPT_DIR, 'sample_ficus.ply'),
                        help="Path to output .ply mesh")
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Inference Device: {device}")

    # 1. Load model tensors
    sdf_net = load_sdf_model(args.checkpoint, device)

    # 2. Extract polygon mesh
    t_start = time.time()
    mesh = extract_mesh_on_the_fly(sdf_net, device, resolution=args.resolution)
    total_time = time.time() - t_start

    # 3. Export to disk
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mesh.export(args.output)
    
    file_size_kb = os.path.getsize(args.output) / 1024
    print("\n" + "="*50)
    print("✅ INFERENCE COMPLETE")
    print(f"Output File:     {args.output}")
    print(f"File Size:       {file_size_kb:.1f} KB")
    print(f"Vertices:        {len(mesh.vertices):,}")
    print(f"Faces/Triangles: {len(mesh.faces):,}")
    print(f"Total Time:      {total_time:.2f} seconds")
    print("="*50)

if __name__ == '__main__':
    main()
