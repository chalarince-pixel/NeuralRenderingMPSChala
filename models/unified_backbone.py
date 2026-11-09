"""
Unified Neural Rendering & Diffusion Backbone
Shared architectural principles:
1. Positional Frequency Encoding (Fourier embeddings for coordinates & timesteps)
2. FiLM (Feature-wise Linear Modulation) conditioning
3. Residual MLP blocks with density & color prediction heads
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Fourier frequency positional encoding:
    gamma(p) = [sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^(L-1) pi p), cos(2^(L-1) pi p)]
    Maps low-dimensional coordinates into high-dimensional frequency space.
    """

    def __init__(self, in_channels: int, num_frequencies: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.out_channels = in_channels * (2 * num_frequencies)

        # 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_channels)
        # Output: (..., in_channels * 2 * num_frequencies)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_channels)
        
        # (N, in_channels, 1) * (1, 1, num_frequencies) -> (N, in_channels, num_frequencies)
        scaled = x_flat.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
        scaled = scaled.reshape(x_flat.shape[0], -1)

        sin_features = torch.sin(scaled)
        cos_features = torch.cos(scaled)

        encoded = torch.cat([sin_features, cos_features], dim=-1)
        return encoded.reshape(*orig_shape[:-1], self.out_channels)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM):
    h = gamma(cond) * h + beta(cond)
    Identical conditioning mechanism used across Diffusion (AdaLN) and Neural Rendering.
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.modulation = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is None:
            return x
        # cond: (..., cond_dim) -> (..., feature_dim * 2)
        mod = self.modulation(cond)
        gamma, beta = mod.chunk(2, dim=-1)
        # 1 + gamma allows identity initialization
        return (1.0 + gamma) * x + beta


class UnifiedRadianceBackbone(nn.Module):
    """
    Standard Production NeRF & Diffusion Backbone:
    - 8 Linear layers of hidden_dim (default: 256)
    - Identity skip connection at layer 4 (concatenates original pos_enc back into features)
    - Density head (softplus)
    - View-direction conditioned Radiance color head (sigmoid)
    - Optional FiLM conditioning for styles / species / diffusion timesteps
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
        cond_dim: int = 16,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(in_channels=3, num_frequencies=pos_freqs)
        self.dir_encoder = PositionalEncoding(in_channels=3, num_frequencies=dir_freqs)

        in_pos_dim = self.pos_encoder.out_channels
        in_dir_dim = self.dir_encoder.out_channels
        self.skip_layer = skip_layer
        self.num_layers = num_layers

        # Pre-skip layers: layers 0 to skip_layer-1
        self.layers_pre = nn.ModuleList()
        prev_dim = in_pos_dim
        for i in range(skip_layer):
            self.layers_pre.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Post-skip layers: layer 4 receives (hidden_dim + in_pos_dim)
        self.layers_post = nn.ModuleList()
        prev_dim = hidden_dim + in_pos_dim
        for i in range(skip_layer, num_layers):
            self.layers_post.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Density head: predicts volume density sigma
        self.density_head = nn.Linear(hidden_dim, 1)
        # Initialize bias slightly negative to discourage initial fog
        nn.init.constant_(self.density_head.bias, -0.5)

        # Radiance color branch (conditioned on view direction)
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.color_pre = nn.Linear(hidden_dim + in_dir_dim, hidden_dim // 2)
        self.color_head = nn.Linear(hidden_dim // 2, 3)

        # Optional FiLM conditioning
        self.film = FiLMBlock(hidden_dim, cond_dim)

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(directions)

        # First half of MLP
        h = pos_enc
        for layer in self.layers_pre:
            h = F.relu(layer(h))

        # Skip connection: inject input coordinates back at layer 4
        h = torch.cat([h, pos_enc], dim=-1)

        # Second half of MLP
        for layer in self.layers_post:
            h = F.relu(layer(h))

        # Optional FiLM modulation
        if cond is not None:
            if cond.dim() == 2:
                cond_expanded = cond.unsqueeze(1).expand(-1, positions.shape[1], -1)
            else:
                cond_expanded = cond
            h = self.film(h, cond_expanded)

        # 1. Volume Density: softplus ensures sigma >= 0
        density = F.softplus(self.density_head(h))

        # 2. Radiance Color: conditioned on 3D view direction
        features = self.feature_linear(h)
        color_in = torch.cat([features, dir_enc], dim=-1)
        h_color = F.relu(self.color_pre(color_in))
        color = torch.sigmoid(self.color_head(h_color))

        return density, color
