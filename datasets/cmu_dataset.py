"""
High-Resolution Multi-View Dataset Loader for CMU / NeuS Pipeline.
Ingests all 100 views at full 800x800 native resolution.
Extracts:
1. Native RGB pixel targets
2. Exact binary silhouette masks (from alpha channel) for CMU silhouette loss
3. Camera ray origins and directions (o, d)
"""

import os
import json
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch

from datasets.synthetic_generator import get_camera_rays


class CMUNeRFDataset:
    """
    Ingests all 100 camera views at native high resolution (up to 800x800).
    Provides balanced ray sampling and exact binary silhouette masks.
    """

    def __init__(
        self,
        data_dir: str = "data/nerf_synthetic/ficus",
        split: str = "train",
        resolution: int = 800,
        max_views: int = 100,
        device: torch.device = torch.device("mps"),
    ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.device = device

        transforms_file = os.path.join(data_dir, f"transforms_{split}.json")
        if not os.path.exists(transforms_file):
            raise FileNotFoundError(f"Transforms file not found: {transforms_file}")

        with open(transforms_file, "r") as f:
            meta = json.load(f)

        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * resolution / np.tan(0.5 * camera_angle_x)

        frames = meta.get("frames", [])[:max_views]
        print(f"[CMU Dataset] Loading all {len(frames)} views at {resolution}x{resolution} native resolution...")

        self.c2w_matrices = []
        self.images = []
        self.masks = []

        for idx, frame in enumerate(frames):
            rel_path = frame["file_path"].lstrip("./")
            if not rel_path.endswith(".png"):
                rel_path = f"{rel_path}.png"
            img_path = os.path.join(data_dir, rel_path)

            img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                continue

            if img_raw.shape[0] != resolution or img_raw.shape[1] != resolution:
                img_raw = cv2.resize(img_raw, (resolution, resolution), interpolation=cv2.INTER_AREA)

            alpha = (img_raw[:, :, 3:4].astype(np.float32) / 255.0)
            bgr = img_raw[:, :, :3].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Composite on pure white background
            rgb_comp = rgb * alpha + (1.0 - alpha) * 1.0
            binary_mask = (alpha > 0.1).astype(np.float32).squeeze(-1)

            self.images.append(torch.tensor(rgb_comp, dtype=torch.float32))
            self.masks.append(torch.tensor(binary_mask, dtype=torch.float32))
            self.c2w_matrices.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

        self.num_views = len(self.images)
        self.c2w_matrices = torch.stack(self.c2w_matrices, dim=0).to(device)

        # Precompute camera rays on device
        print(f"[CMU Dataset] Precomputing camera rays for all {self.num_views} views...")
        all_rays_o = []
        all_rays_d = []
        for v in range(self.num_views):
            ro, rd = get_camera_rays(resolution, resolution, self.focal, self.c2w_matrices[v])
            all_rays_o.append(ro)
            all_rays_d.append(rd)

        self.rays_o = torch.stack(all_rays_o, dim=0).to(device)  # (N, H, W, 3)
        self.rays_d = torch.stack(all_rays_d, dim=0).to(device)  # (N, H, W, 3)
        self.images = torch.stack(self.images, dim=0).to(device)  # (N, H, W, 3)
        self.masks = torch.stack(self.masks, dim=0).to(device)    # (N, H, W)

        # Flattened dataset indexing for ray sampling
        self.rays_o_flat = self.rays_o.view(-1, 3)
        self.rays_d_flat = self.rays_d.view(-1, 3)
        self.images_flat = self.images.view(-1, 3)
        self.masks_flat = self.masks.view(-1)

        self.fg_indices = torch.where(self.masks_flat > 0.5)[0]
        self.bg_indices = torch.where(self.masks_flat <= 0.5)[0]

        print(f"[CMU Dataset] Successfully initialized {self.num_views} views ({resolution}x{resolution}).")
        print(f"              Total High-Res Rays: {self.rays_o_flat.shape[0]:,} (Tree: {len(self.fg_indices):,}, Sky: {len(self.bg_indices):,})")

    def sample_batch(self, batch_size: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a balanced ray batch:
        50% foreground rays (tree leaves, trunk, pot)
        50% background rays (sky)
        Returns:
            rays_o: (B, 3)
            rays_d: (B, 3)
            target_rgb: (B, 3)
            target_mask: (B,)
        """
        half = batch_size // 2
        idx_fg = self.fg_indices[torch.randint(0, len(self.fg_indices), (half,), device=self.device)]
        idx_bg = self.bg_indices[torch.randint(0, len(self.bg_indices), (half,), device=self.device)]
        idx = torch.cat([idx_fg, idx_bg])

        return (
            self.rays_o_flat[idx],
            self.rays_d_flat[idx],
            self.images_flat[idx],
            self.masks_flat[idx],
        )
