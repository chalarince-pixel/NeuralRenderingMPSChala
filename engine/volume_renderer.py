"""
Differentiable Volumetric Raymarching Engine
Executes numerical quadrature over continuous density and radiance fields:
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
Fully differentiable; gradients backpropagate smoothly to network weights and LoRA adapters.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class VolumeRenderer(nn.Module):
    def __init__(
        self,
        near: float = 1.0,
        far: float = 5.0,
        num_samples: int = 64,
        perturb: bool = True,
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.perturb = perturb

    def sample_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples 3D points along rays between near and far planes.
        Args:
            rays_o: (N_rays, 3)
            rays_d: (N_rays, 3)
        Returns:
            pts: (N_rays, num_samples, 3)
            z_vals: (N_rays, num_samples)
            dirs: (N_rays, num_samples, 3)
        """
        N_rays = rays_o.shape[0]
        device = rays_o.device

        # Linear partition along ray
        t_vals = torch.linspace(0.0, 1.0, self.num_samples, device=device)
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals
        z_vals = z_vals.expand(N_rays, self.num_samples)

        # Stratified sampling during training
        if self.perturb and self.training:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # 3D sample coordinates: o + t * d
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        dirs = rays_d.unsqueeze(1).expand(-1, self.num_samples, -1).contiguous()

        return pts, z_vals, dirs

    def render_rays(
        self,
        network: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        background_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs full volume rendering pass for a batch of camera rays.
        """
        pts, z_vals, dirs = self.sample_along_rays(rays_o, rays_d)

        # Evaluate neural network: (N_rays, num_samples, 1) and (N_rays, num_samples, 3)
        density, color = network(pts, dirs, cond)

        # Compute delta distances between consecutive samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        last_dist = dists[:, -1:]
        dists = torch.cat([dists, last_dist], dim=-1)  # (N_rays, num_samples)

        # Numerical quadrature (Alpha compositing)
        # alpha = 1 - exp(-sigma * delta)
        sigma = density.squeeze(-1)
        alpha = 1.0 - torch.exp(-sigma * dists)

        # Transmittance T_i = prod_{j<i} (1 - alpha_j)
        # Compute cumprod in log space for numerical stability:
        # T_i = exp(cumsum(-sigma * delta))
        cum_alpha = torch.cat(
            [torch.ones((rays_o.shape[0], 1), device=rays_o.device), 1.0 - alpha + 1e-10],
            dim=-1,
        )
        transmittance = torch.cumprod(cum_alpha, dim=-1)[:, :-1]

        # Weight of each sample point: w_i = T_i * alpha_i
        weights = alpha * transmittance  # (N_rays, num_samples)

        # Rendered outputs
        # 1. RGB Beauty Pass
        rgb_map = torch.sum(weights.unsqueeze(-1) * color, dim=1)

        # 2. Depth Map
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # 3. Accumulated Opacity
        acc_map = torch.sum(weights, dim=-1)

        # Background blending
        if background_color is not None:
            rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1)) * background_color

        return {
            "rgb": rgb_map,
            "depth": depth_map,
            "opacity": acc_map,
            "weights": weights,
        }
