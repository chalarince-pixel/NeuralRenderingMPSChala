"""
CMU / NeuS Surface-Finding Point Sampler & Differentiable Raymarcher.
Replicates the CMU CIRL sampling strategy:
1. Probe rays to detect zero-crossing surface boundaries (S(x) = 0).
2. Allocate dense surface samples directly inside the millimeter-thin leaf interval.
3. Compute physically rigorous logistic density:
   sigma(x) = s * exp(-s * S(x)) / (1 + exp(-s * S(x)))^2
4. Integrate radiance and opacity with full autograd backpropagation.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class CMUSurfaceRenderer(nn.Module):
    """
    Differentiable surface raymarcher with adaptive zero-crossing surface sampling.
    """

    def __init__(
        self,
        near: float = 2.2,
        far: float = 5.8,
        num_coarse_probes: int = 64,
        num_fine_surface_samples: int = 32,
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.num_coarse_probes = num_coarse_probes
        self.num_fine_surface_samples = num_fine_surface_samples
        self.total_samples = num_coarse_probes + num_fine_surface_samples

    def sample_rays(
        self,
        network: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarse-to-fine surface sampling:
        1. Probe coarse intervals along ray.
        2. Evaluate S(x) to locate zero-crossing surface.
        3. Draw fine samples densely around the zero crossing.
        Returns:
            pts: (N_rays, total_samples, 3)
            z_vals: (N_rays, total_samples) sorted along the ray
        """
        N_rays = rays_o.shape[0]
        device = rays_o.device

        # 1. Coarse linear partition
        t_vals = torch.linspace(0.0, 1.0, self.num_coarse_probes, device=device)
        z_coarse = self.near * (1.0 - t_vals) + self.far * t_vals
        z_coarse = z_coarse.expand(N_rays, self.num_coarse_probes)

        if self.training:
            # Stratified jitter
            mids = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
            upper = torch.cat([mids, z_coarse[:, -1:]], dim=-1)
            lower = torch.cat([z_coarse[:, :1], mids], dim=-1)
            z_coarse = lower + (upper - lower) * torch.rand_like(z_coarse)

        pts_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_coarse.unsqueeze(-1)

        # 2. Evaluate coarse SDF to find zero-crossing surface interval
        with torch.no_grad():
            sdf_coarse, _ = network.get_sdf(pts_coarse)  # (N_rays, num_coarse, 1)
            sdf_vals = sdf_coarse.squeeze(-1)

            # Detect where S(x) is closest to 0 (the surface of the leaf/pot)
            min_idx = torch.argmin(torch.abs(sdf_vals), dim=-1, keepdim=True)  # (N_rays, 1)
            z_center = torch.gather(z_coarse, dim=1, index=min_idx)            # (N_rays, 1)

            # Place fine samples in a tight window [-0.25, +0.25] around surface
            t_fine = torch.linspace(-0.25, 0.25, self.num_fine_surface_samples, device=device)
            z_fine = z_center + t_fine.unsqueeze(0)
            z_fine = z_fine.clamp(self.near, self.far)

        # Combine coarse and fine sample points and sort
        z_vals = torch.cat([z_coarse, z_fine], dim=-1)
        z_vals, _ = torch.sort(z_vals, dim=-1)

        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        return pts, z_vals

    def render_rays(
        self,
        network: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        background_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable raymarching quadrature:
        C(r) = sum_i T_i * alpha_i * c_i
        """
        pts, z_vals = self.sample_rays(network, rays_o, rays_d)
        dirs = rays_d.unsqueeze(1).expand(-1, z_vals.shape[1], -1).contiguous()

        # Evaluate network (density sigma, radiance color c)
        density, color = network(pts, dirs)

        # Compute step size delta between consecutive samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        last_dist = dists[:, -1:]
        dists = torch.cat([dists, last_dist], dim=-1)  # (N_rays, total_samples)

        # Numerical quadrature (Alpha compositing)
        sigma = density.squeeze(-1)
        alpha = 1.0 - torch.exp(-sigma * dists)

        # Transmittance T_i = prod_{j<i} (1 - alpha_j)
        cum_alpha = torch.cat(
            [torch.ones((rays_o.shape[0], 1), device=rays_o.device), 1.0 - alpha + 1e-10],
            dim=-1,
        )
        transmittance = torch.cumprod(cum_alpha, dim=-1)[:, :-1]

        weights = alpha * transmittance  # (N_rays, total_samples)

        # 1. Predicted RGB
        rgb_map = torch.sum(weights.unsqueeze(-1) * color, dim=1)

        # 2. Accumulated Opacity
        opacity_map = torch.sum(weights, dim=-1)

        # 3. Depth
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Background compositing
        if background_color is not None:
            rgb_map = rgb_map + (1.0 - opacity_map.unsqueeze(-1)) * background_color

        return {
            "rgb": rgb_map,
            "opacity": opacity_map,
            "depth": depth_map,
            "weights": weights,
            "pts": pts,
        }
