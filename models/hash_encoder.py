"""
Multi-Resolution 3D Hash Grid Encoder (Instant-NGP Architecture) in Pure PyTorch.
Decouples 3D space into localized multi-scale hash tables.
Eliminates the global spectral bias of plain MLPs:
- Local parameter updates: gradients only affect local grid corners.
- Extreme convergence speed: converges in seconds without global interference.
- Fully compatible with Apple Silicon Metal Performance Shaders (MPS).
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn


class MultiResHashEncoder(nn.Module):
    """
    Multi-Resolution 3D Hash Table Encoder.
    Maps 3D coordinates (x, y, z) into a multi-scale concatenated feature vector.
    """

    def __init__(
        self,
        bounding_box: float = 1.5,
        num_levels: int = 12,
        features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.bounding_box = bounding_box
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 1 << log2_hashmap_size  # 2^15 = 32,768 entries per level
        self.out_dim = num_levels * features_per_level

        # Compute resolution growth factor b = exp( (ln N_max - ln N_min) / (L - 1) )
        self.growth_factor = math.exp(
            (math.log(finest_resolution) - math.log(base_resolution)) / (num_levels - 1)
        )

        self.resolutions = [
            math.floor(base_resolution * (self.growth_factor ** l))
            for l in range(num_levels)
        ]

        # Learnable hash tables: one embedding per level
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.empty(self.hashmap_size, features_per_level))
            for _ in range(num_levels)
        ])

        # Large prime numbers for spatial hashing
        self.register_buffer("primes", torch.tensor([1, 2654435761, 805459861], dtype=torch.int64))

        # Initialize embeddings with small uniform noise
        for emb in self.embeddings:
            nn.init.uniform_(emb, -1e-4, 1e-4)

    def hash_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Fast spatial hash function: h(x,y,z) = (x*p1 ^ y*p2 ^ z*p3) % hashmap_size
        Args:
            coords: (N, 8, 3) integer grid vertex coordinates
        Returns:
            indices: (N, 8) hash table entry indices
        """
        x = coords[..., 0].to(torch.int64) * self.primes[0]
        y = coords[..., 1].to(torch.int64) * self.primes[1]
        z = coords[..., 2].to(torch.int64) * self.primes[2]
        hashed = (x ^ y ^ z) & (self.hashmap_size - 1)  # Bitwise modulo for power of 2
        return hashed

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (..., 3) continuous 3D world coordinates
        Returns:
            features: (..., num_levels * features_per_level)
        """
        orig_shape = positions.shape
        x_flat = positions.reshape(-1, 3)

        # Normalize 3D coordinates from [-bound, bound] to [0, 1]
        x_norm = (x_flat + self.bounding_box) / (2.0 * self.bounding_box)
        x_norm = torch.clamp(x_norm, 0.0, 1.0 - 1e-6)

        level_features = []

        for l in range(self.num_levels):
            res = self.resolutions[l]
            # Grid coordinates scaled by resolution: [0, res]
            scaled = x_norm * res
            c0 = torch.floor(scaled).to(torch.int64)  # Lower corner (N, 3)
            c1 = c0 + 1                               # Upper corner (N, 3)
            weights = scaled - c0.float()             # Offset weights in [0, 1] (N, 3)

            w_x = weights[:, 0:1]
            w_y = weights[:, 1:2]
            w_z = weights[:, 2:3]

            # 8 corners of the 3D voxel
            corners = torch.stack([
                torch.stack([c0[:, 0], c0[:, 1], c0[:, 2]], dim=-1),  # 000
                torch.stack([c1[:, 0], c0[:, 1], c0[:, 2]], dim=-1),  # 100
                torch.stack([c0[:, 0], c1[:, 1], c0[:, 2]], dim=-1),  # 010
                torch.stack([c1[:, 0], c1[:, 1], c0[:, 2]], dim=-1),  # 110
                torch.stack([c0[:, 0], c0[:, 1], c1[:, 2]], dim=-1),  # 001
                torch.stack([c1[:, 0], c0[:, 1], c1[:, 2]], dim=-1),  # 101
                torch.stack([c0[:, 0], c1[:, 1], c1[:, 2]], dim=-1),  # 011
                torch.stack([c1[:, 0], c1[:, 1], c1[:, 2]], dim=-1),  # 111
            ], dim=1)  # (N, 8, 3)

            indices = self.hash_indices(corners)  # (N, 8)

            # Retrieve corner features: (N, 8, F)
            corner_feats = self.embeddings[l][indices]

            # Vectorized trilinear interpolation
            c000 = corner_feats[:, 0]
            c100 = corner_feats[:, 1]
            c010 = corner_feats[:, 2]
            c110 = corner_feats[:, 3]
            c001 = corner_feats[:, 4]
            c101 = corner_feats[:, 5]
            c011 = corner_feats[:, 6]
            c111 = corner_feats[:, 7]

            c00 = c000 * (1.0 - w_x) + c100 * w_x
            c01 = c001 * (1.0 - w_x) + c101 * w_x
            c10 = c010 * (1.0 - w_x) + c110 * w_x
            c11 = c011 * (1.0 - w_x) + c111 * w_x

            c0 = c00 * (1.0 - w_y) + c10 * w_y
            c1 = c01 * (1.0 - w_y) + c11 * w_y

            interp = c0 * (1.0 - w_z) + c1 * w_z  # (N, F)
            level_features.append(interp)

        # Concatenate across all resolution levels: (N, L * F)
        out_feats = torch.cat(level_features, dim=-1)
        return out_feats.reshape(*orig_shape[:-1], self.out_dim)
