"""
NeRF Multi-View Image Dataset Loader.
Loads real image files (r_0.png, r_1.png, ...) and camera matrices from transforms_train.json.
Converts real photos into PyTorch camera ray tensors on Apple Silicon GPU.
"""

import os
import json
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch

from datasets.synthetic_generator import get_camera_rays


class NeRFImageDataset:
    """
    Loads real multi-view images and camera poses from NeRF synthetic format.
    """

    def __init__(
        self,
        data_dir: str = "data/nerf_synthetic/ficus",
        split: str = "train",
        target_size: int = 100,
        max_views: int = 25,
        device: torch.device = torch.device("mps"),
    ):
        self.data_dir = data_dir
        self.target_size = target_size
        self.device = device

        transforms_file = os.path.join(data_dir, f"transforms_{split}.json")
        if not os.path.exists(transforms_file):
            raise FileNotFoundError(f"Transforms file not found: {transforms_file}")

        with open(transforms_file, "r") as f:
            meta = json.load(f)

        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * target_size / np.tan(0.5 * camera_angle_x)

        frames = meta.get("frames", [])[:max_views]
        print(f"[DataLoader] Ingesting {len(frames)} real photos from {data_dir}...")

        images = []
        c2w_matrices = []
        fg_masks = []

        for frame in frames:
            rel_path = frame["file_path"]
            if not rel_path.endswith(".png"):
                rel_path = f"{rel_path}.png"
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]

            img_path = os.path.join(data_dir, rel_path)
            # Read RGBA
            img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                continue

            # Resize to target training size (e.g., 100x100)
            img_resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # Convert BGR(A) -> RGB(A) float [0, 1]
            if img_resized.shape[-1] == 4:
                alpha = img_resized[:, :, 3:4].astype(np.float32) / 255.0
                rgb = cv2.cvtColor(img_resized[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Composite on white background
                rgb = rgb * alpha + (1.0 - alpha) * 1.0
                fg_mask = (alpha > 0.1).squeeze(-1)
            else:
                rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                fg_mask = (rgb < 0.95).any(axis=-1)

            images.append(torch.tensor(rgb, dtype=torch.float32, device=device))
            fg_masks.append(torch.tensor(fg_mask, dtype=torch.bool, device=device))

            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
            c2w_matrices.append(c2w)

        self.images = torch.stack(images, dim=0)          # (N, H, W, 3)
        self.c2w_matrices = torch.stack(c2w_matrices, dim=0) # (N, 4, 4)
        self.fg_masks = torch.stack(fg_masks, dim=0)      # (N, H, W)
        self.H = target_size
        self.W = target_size
        self.num_views = len(images)

        # Generate camera rays for all views
        all_rays_o = []
        all_rays_d = []
        for i in range(self.num_views):
            ro, rd = get_camera_rays(self.H, self.W, self.focal, self.c2w_matrices[i])
            all_rays_o.append(ro)
            all_rays_d.append(rd)

        self.rays_o = torch.stack(all_rays_o, dim=0)
        self.rays_d = torch.stack(all_rays_d, dim=0)

        # Flattened dataset tensors
        self.all_rays_o_flat = self.rays_o.view(-1, 3)
        self.all_rays_d_flat = self.rays_d.view(-1, 3)
        self.all_target_rgb_flat = self.images.view(-1, 3)
        self.all_fg_mask_flat = self.fg_masks.view(-1)

        self.fg_indices = torch.where(self.all_fg_mask_flat)[0]
        self.bg_indices = torch.where(~self.all_fg_mask_flat)[0]

        print(f"[DataLoader] Successfully loaded {self.num_views} real images ({self.W}x{self.H}).")
        print(f"             Total Rays: {self.all_rays_o_flat.shape[0]:,} (Tree: {len(self.fg_indices):,}, Sky: {len(self.bg_indices):,})")

    def sample_batch(self, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a balanced batch: 50% rays on the tree, 50% rays on the background.
        """
        half = batch_size // 2
        idx_fg = self.fg_indices[torch.randint(0, len(self.fg_indices), (half,), device=self.device)]
        idx_bg = self.bg_indices[torch.randint(0, len(self.bg_indices), (half,), device=self.device)]
        idx = torch.cat([idx_fg, idx_bg])

        return (
            self.all_rays_o_flat[idx],
            self.all_rays_d_flat[idx],
            self.all_target_rgb_flat[idx],
        )
