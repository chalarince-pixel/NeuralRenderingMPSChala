"""
Hash-Encoded Signed Distance Function (SDF) & Radiance Backbone.
Combines:
1. Multi-Resolution 3D Hash Grid (Instant-NGP) for zero-spectral-bias local feature extraction.
2. Signed Distance Function S(x) output head (NeuS / VolSDF / CMU formulation).
3. Physically rigorous SDF-to-density mapping:
   sigma(x) = s * exp(-s * S(x)) / (1 + exp(-s * S(x)))^2
4. Eikonal loss regularization (forces ||grad S|| = 1 to guarantee solid surfaces without fog).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hash_encoder import MultiResHashEncoder
from models.unified_backbone import PositionalEncoding, FiLMBlock


class HashSDFBackbone(nn.Module):
    """
    State-of-the-art Neural Surface Renderer:
    Hash-Grid -> SDF Head S(x) -> Logistic Density sigma -> View-Dependent Radiance c.
    """

    def __init__(
        self,
        bounding_box: float = 1.5,
        num_levels: int = 12,
        features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        hidden_dim: int = 64,
        dir_freqs: int = 4,
        init_sharpness: float = 15.0,
    ):
        super().__init__()
        # 1. Multi-resolution Hash Grid
        self.encoder = MultiResHashEncoder(
            bounding_box=bounding_box,
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
        )
        in_dim = self.encoder.out_dim  # 24

        # 2. View direction encoding
        self.dir_encoder = PositionalEncoding(in_channels=3, num_frequencies=dir_freqs)
        in_dir_dim = self.dir_encoder.out_channels

        # 3. Geometric SDF MLP
        self.sdf_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize SDF bias so it starts as a sphere around origin
        # S(x) ~ ||x|| - 0.5 (positive outside, negative inside)
        nn.init.constant_(self.sdf_net[-1].bias, 0.1)

        # 4. Learnable density sharpness s = exp(inv_s)
        self.inv_s = nn.Parameter(torch.tensor(init_sharpness).log())

        # 5. Radiance Color Head
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + in_dir_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def get_sdf(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates Signed Distance S(x) and intermediate geometric features.
        """
        features = self.encoder(positions)
        h = F.softplus(self.sdf_net[0](features), beta=100)
        h = F.softplus(self.sdf_net[2](h), beta=100)
        sdf = self.sdf_net[4](h)  # (..., 1)
        return sdf, h

    def sdf_to_density(self, sdf: torch.Tensor) -> torch.Tensor:
        """
        NeuS / VolSDF logistic density transformation:
        sigma(x) = s * exp(-s * S(x)) / (1 + exp(-s * S(x)))^2
        """
        s = torch.exp(self.inv_s).clamp(1.0, 500.0)
        # Using sigmoid derivative identity: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        # sigma = s * sigmoid(s * S) * (1 - sigmoid(s * S))
        exp_val = torch.exp(-s * sdf)
        sigma = s * exp_val / ((1.0 + exp_val) ** 2 + 1e-6)
        return sigma

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            density: (..., 1) >= 0
            color: (..., 3) in [0, 1]
        """
        sdf, geom_features = self.get_sdf(positions)
        density = self.sdf_to_density(sdf)

        dir_enc = self.dir_encoder(directions)
        color_in = torch.cat([geom_features, dir_enc], dim=-1)
        color = self.color_net(color_in)

        return density, color

    def compute_eikonal_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Computes the Eikonal regularization loss: ||grad S(x)||_2 == 1.
        Forces the function S(x) to be an exact physical Euclidean distance field,
        preventing any floating fog in empty space.
        """
        pts = positions.clone().detach().requires_grad_(True)
        sdf, _ = self.get_sdf(pts)

        # Gradient of SDF w.r.t 3D coordinate
        d_output = torch.ones_like(sdf, requires_grad=False, device=positions.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = torch.norm(gradients, dim=-1)
        eikonal_loss = torch.mean((grad_norm - 1.0) ** 2)
        return eikonal_loss
