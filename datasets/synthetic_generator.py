"""
Synthetic & Procedural 3D Multi-View Data Generator
Constructs camera intrinsics, camera extrinsics (turntable orbit), and procedural 3D ground truth
scenes (volumetric tree geometry) directly in PyTorch tensors on Apple Silicon GPU.
Zero external downloads required.
"""

import math
from typing import Dict, List, Tuple
import torch


def get_camera_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes camera ray origins and unit directions for all pixels in an HxW sensor.
    Args:
        H, W: Image dimensions
        focal: Focal length in pixels
        c2w: Camera-to-world transformation matrix (4, 4)
    Returns:
        rays_o: (H, W, 3) Ray origins
        rays_d: (H, W, 3) Ray direction unit vectors
    """
    device = c2w.device
    # Pixel grid
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
        indexing="xy",
    )
    # Direction in camera coordinates (pinhole camera model)
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    # Rotate directions from camera to world space: R * d
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins: camera position in world space
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def generate_orbit_poses(
    num_views: int,
    radius: float = 3.0,
    elevation: float = 20.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generates turntable 360 camera-to-world (c2w) matrices orbiting around origin (0, 0, 0).
    """
    poses = []
    elev_rad = math.radians(elevation)

    for i in range(num_views):
        azimuth_rad = 2.0 * math.pi * (i / num_views)

        # Camera position
        cam_x = radius * math.cos(elev_rad) * math.sin(azimuth_rad)
        cam_y = radius * math.sin(elev_rad)
        cam_z = radius * math.cos(elev_rad) * math.cos(azimuth_rad)
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=device, dtype=torch.float32)

        # Look-at matrix towards origin
        forward = -cam_pos / torch.norm(cam_pos)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        right = torch.cross(forward, up, dim=-1)
        right = right / torch.norm(right)
        up = torch.cross(right, forward, dim=-1)

        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        poses.append(c2w)

    return torch.stack(poses, dim=0)


def evaluate_procedural_tree(
    pts: torch.Tensor,
    season_style: str = "summer",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytical ground-truth 3D density and color for a volumetric tree:
    - Vertical cylindrical trunk (x^2 + z^2 < r_trunk^2, y in [-1, 0.2])
    - Multi-lobed spherical foliage canopy (y in [0.0, 1.2])
    """
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    # 1. Trunk (brown cylinder)
    dist_trunk_sq = x**2 + z**2
    trunk_mask = (dist_trunk_sq < 0.04) & (y >= -1.0) & (y <= 0.2)
    trunk_density = torch.where(trunk_mask, torch.tensor(20.0, device=pts.device), torch.tensor(0.0, device=pts.device))
    trunk_color = torch.tensor([0.45, 0.28, 0.15], device=pts.device).expand_as(pts)

    # 2. Foliage canopy (upper spherical cluster)
    canopy_center = torch.tensor([0.0, 0.5, 0.0], device=pts.device)
    dist_canopy = torch.norm(pts - canopy_center, dim=-1)
    canopy_mask = dist_canopy < 0.65
    canopy_density = torch.where(canopy_mask, 15.0 * (1.0 - dist_canopy / 0.65), torch.tensor(0.0, device=pts.device))

    # Seasonal color variation (Summer Green vs Autumn Amber)
    if season_style == "autumn":
        foliage_color = torch.tensor([0.85, 0.35, 0.10], device=pts.device).expand_as(pts)
    else:
        foliage_color = torch.tensor([0.18, 0.65, 0.22], device=pts.device).expand_as(pts)

    # Blend densities and colors
    total_density = (trunk_density + canopy_density).unsqueeze(-1)
    
    weight_trunk = trunk_density / (trunk_density + canopy_density + 1e-6)
    total_color = weight_trunk.unsqueeze(-1) * trunk_color + (1.0 - weight_trunk.unsqueeze(-1)) * foliage_color

    return total_density, total_color


def generate_ground_truth_views(
    poses: torch.Tensor,
    H: int = 64,
    W: int = 64,
    focal: float = 64.0,
    season_style: str = "summer",
    num_samples: int = 64,
    near: float = 1.0,
    far: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Renders analytical ground-truth RGB images for all given camera poses.
    Returns:
        images: (num_views, H, W, 3)
        all_rays_o: (num_views, H, W, 3)
        all_rays_d: (num_views, H, W, 3)
    """
    num_views = poses.shape[0]
    device = poses.device

    images = []
    all_rays_o = []
    all_rays_d = []

    for v in range(num_views):
        rays_o, rays_d = get_camera_rays(H, W, focal, poses[v])
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

        # Raymarch through analytical scene
        rays_o_flat = rays_o.view(-1, 3)
        rays_d_flat = rays_d.view(-1, 3)

        t_vals = torch.linspace(near, far, num_samples, device=device)
        z_vals = t_vals.unsqueeze(0).expand(rays_o_flat.shape[0], -1)

        pts = rays_o_flat.unsqueeze(1) + rays_d_flat.unsqueeze(1) * z_vals.unsqueeze(-1)
        density, color = evaluate_procedural_tree(pts, season_style=season_style)

        # Quadrature
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        last_dist = dists[:, -1:]
        dists = torch.cat([dists, last_dist], dim=-1)

        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)
        cum_alpha = torch.cat(
            [torch.ones((rays_o_flat.shape[0], 1), device=device), 1.0 - alpha + 1e-10],
            dim=-1,
        )
        transmittance = torch.cumprod(cum_alpha, dim=-1)[:, :-1]
        weights = alpha * transmittance

        # Composite against white sky background
        rgb_flat = torch.sum(weights.unsqueeze(-1) * color, dim=1)
        acc = torch.sum(weights, dim=-1, keepdim=True)
        rgb_flat = rgb_flat + (1.0 - acc) * 1.0  # White background

        images.append(rgb_flat.view(H, W, 3))

    return (
        torch.stack(images, dim=0),
        torch.stack(all_rays_o, dim=0),
        torch.stack(all_rays_d, dim=0),
    )
