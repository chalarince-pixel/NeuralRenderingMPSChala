#!/usr/bin/env python3
"""
Standalone 3D Geometry & Point Cloud Extraction Inference Script (V1 Baseline).
Loads trained .safetensors weights (base model + optional LoRA adapter)
and extracts colored 3D geometry (.ply) on Apple Silicon (mps) or CPU in ~0.5 seconds.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from models.unified_backbone import UnifiedBackbone
from models.lora import LoRALinear


def load_model(base_path: str, lora_path: str = None, device: torch.device = torch.device("cpu")):
    """Instantiate UnifiedBackbone and load .safetensors tensors."""
    print(f"Loading base model from: {base_path}")
    model = UnifiedBackbone(pos_freqs=10, dir_freqs=4, hidden_dim=256, num_layers=8)
    
    # Apply LoRA layers if adapter provided
    if lora_path and os.path.exists(lora_path):
        print(f"Injecting LoRA adapter layers from: {lora_path}")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "pts_linears" in name:
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank=16, alpha=32.0))

    state_dict = load_file(base_path)
    model.load_state_dict(state_dict, strict=False)

    if lora_path and os.path.exists(lora_path):
        lora_dict = load_file(lora_path)
        model.load_state_dict(lora_dict, strict=False)
        print("LoRA adapter weights merged into active model.")

    model.to(device)
    model.eval()
    return model


def extract_point_cloud(
    model: nn.Module,
    filepath: str = "exports/scene_geometry.ply",
    grid_size: int = 64,
    bound: float = 1.0,
    density_threshold: float = 5.0,
    device: torch.device = torch.device("cpu")
):
    """Evaluates 3D neural density field across a coordinate grid and exports colored .ply point cloud."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    print(f"Evaluating 3D neural field at {grid_size}^3 ({grid_size**3:,} points) on {device}...")
    t0 = time.time()

    coords = torch.linspace(-bound, bound, grid_size, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([x, y, z], dim=-1).view(-1, 3)
    dirs = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(pts)

    chunk_size = 32768
    densities = []
    colors = []

    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk_size):
            pts_chunk = pts[i : i + chunk_size].unsqueeze(0)
            dirs_chunk = dirs[i : i + chunk_size].unsqueeze(0)
            d_chunk, c_chunk = model(pts_chunk, dirs_chunk)
            densities.append(d_chunk.squeeze(0).squeeze(-1).cpu())
            colors.append(c_chunk.squeeze(0).cpu())

    density = torch.cat(densities, dim=0)
    color = torch.cat(colors, dim=0)

    # Filter points by density threshold
    mask = density > density_threshold
    active_pts = pts[mask].cpu().numpy()
    active_colors = (color[mask].numpy() * 255.0).clip(0, 255).astype(np.uint8)

    t_eval = time.time() - t0
    print(f"Evaluation finished in {t_eval:.2f}s. Found {len(active_pts):,} surface points.")

    # Write binary/ASCII PLY
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(active_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(active_pts, active_colors):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")

    print(f"Saved extracted 3D point cloud mesh to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Extract 3D Geometry from NeRF/LoRA safetensors")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base_model.safetensors", help="Path to base model .safetensors")
    parser.add_argument("--lora", type=str, default=None, help="Optional path to lora_adapter.safetensors")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution (e.g. 64 for 262k points, 100 for 1M points)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Density threshold for surface extraction")
    parser.add_argument("--output", type=str, default="exports/scene_geometry.ply", help="Output path (.ply)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running inference on: {device}")

    model = load_model(args.checkpoint, args.lora, device=device)
    extract_point_cloud(
        model=model,
        filepath=args.output,
        grid_size=args.resolution,
        density_threshold=args.threshold,
        device=device
    )


if __name__ == "__main__":
    main()
