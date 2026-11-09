"""
360-Degree Video Renderer & 3D Geometry Exporter
Renders a smooth cinematic orbiting camera trajectory into an .mp4 video
and exports the 3D scene point cloud into a .ply file readable by Blender/MeshLab.
"""

import os
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn

from datasets.synthetic_generator import generate_orbit_poses, get_camera_rays
from engine.volume_renderer import VolumeRenderer


def render_orbit_video(
    network: nn.Module,
    renderer: VolumeRenderer,
    output_dir: str = "renders",
    video_name: str = "orbit_360.mp4",
    num_frames: int = 60,
    H: int = 128,
    W: int = 128,
    focal: float = 128.0,
    radius: float = 3.0,
    elevation: float = 20.0,
    fps: int = 24,
    cond: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("mps"),
) -> str:
    """
    Renders a continuous 360-degree turntable video of the 3D neural scene.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    network.eval()
    poses = generate_orbit_poses(num_frames, radius=radius, elevation=elevation, device=device)

    video_path = os.path.join(output_dir, video_name)
    depth_video_path = os.path.join(output_dir, "depth_pass.mp4")

    # OpenCV VideoWriter for .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    depth_writer = cv2.VideoWriter(depth_video_path, fourcc, fps, (W, H))

    print(f"[Renderer] Generating {num_frames} frames for 360° video ({W}x{H}) on {device}...")

    background = torch.tensor([1.0, 1.0, 1.0], device=device)  # White background
    rgb_pil_frames = []
    depth_pil_frames = []

    with torch.no_grad():
        for f in range(num_frames):
            c2w = poses[f]
            rays_o, rays_d = get_camera_rays(H, W, focal, c2w)
            
            # Flatten rays
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)

            # Render in chunks to keep memory light
            chunk_size = 4096
            rgb_chunks = []
            depth_chunks = []

            for i in range(0, rays_o_flat.shape[0], chunk_size):
                o_chunk = rays_o_flat[i : i + chunk_size]
                d_chunk = rays_d_flat[i : i + chunk_size]

                out = renderer.render_rays(
                    network=network,
                    rays_o=o_chunk,
                    rays_d=d_chunk,
                    cond=cond,
                    background_color=background,
                )
                rgb_chunks.append(out["rgb"])
                depth_chunks.append(out["depth"])

            frame_rgb = torch.cat(rgb_chunks, dim=0).view(H, W, 3)
            frame_depth = torch.cat(depth_chunks, dim=0).view(H, W)

            # Convert RGB tensor [0, 1] -> uint8 BGR and RGB
            rgb_np = (frame_rgb.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

            # Depth map normalization for visualization
            depth_np = frame_depth.cpu().numpy()
            d_min, d_max = depth_np.min(), depth_np.max()
            depth_norm = ((depth_np - d_min) / (d_max - d_min + 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

            from PIL import Image
            rgb_pil_frames.append(Image.fromarray(rgb_np))
            depth_pil_frames.append(Image.fromarray(depth_rgb))

            # Save individual frame
            frame_file = os.path.join(frames_dir, f"frame_{f:04d}.png")
            cv2.imwrite(frame_file, bgr_np)

            # Write to video streams
            rgb_writer.write(bgr_np)
            depth_writer.write(depth_colormap)

    rgb_writer.release()
    depth_writer.release()

    # Also save animated GIF for instant preview without video player dependencies
    gif_path = os.path.join(output_dir, video_name.replace(".mp4", ".gif"))
    depth_gif_path = os.path.join(output_dir, "depth_pass.gif")
    if rgb_pil_frames:
        rgb_pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=rgb_pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        depth_pil_frames[0].save(
            depth_gif_path,
            save_all=True,
            append_images=depth_pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    print(f"[Renderer] 3D Orbit Video saved successfully -> {video_path} and {gif_path}")
    print(f"[Renderer] Depth Pass Video saved successfully -> {depth_video_path} and {depth_gif_path}")
    return video_path


def export_point_cloud_ply(
    network: nn.Module,
    filepath: str = "exports/scene_geometry.ply",
    grid_size: int = 40,
    bound: float = 1.0,
    density_threshold: float = 5.0,
    device: torch.device = torch.device("mps"),
):
    """
    Evaluates the neural field on a 3D bounding box grid and exports high-density voxels
    as a colored 3D point cloud in .ply format.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    network.eval()

    # Generate 3D grid
    coords = torch.linspace(-bound, bound, grid_size, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([x, y, z], dim=-1).view(-1, 3)  # (N, 3)
    dirs = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(pts)

    with torch.no_grad():
        density, color = network(pts.unsqueeze(0), dirs.unsqueeze(0))
        density = density.squeeze(0).squeeze(-1)  # (N,)
        color = color.squeeze(0)  # (N, 3)

    # Filter points by density
    mask = density > density_threshold
    active_pts = pts[mask].cpu().numpy()
    active_colors = (color[mask].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    num_vertices = len(active_pts)
    if num_vertices == 0:
        print("[Export PLY] No points exceeded density threshold.")
        return

    # Write ASCII PLY
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(active_pts, active_colors):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")

    print(f"[Export PLY] Saved {num_vertices} 3D vertices to -> {filepath}")
