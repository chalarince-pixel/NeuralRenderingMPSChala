"""
Production-Grade Neural Radiance Field Training on Real Multi-View Ficus Photos.
Upgrades:
1. Full 8-layer NeRF backbone with 256 hidden channels and skip connection at layer 4
2. Tight volumetric bounds (near=2.3, far=5.7) with 80 stratified samples per ray
3. 3,000 optimization steps with CosineAnnealingLR schedule
4. Fog-elimination loss (penalizes opacity along empty background rays)
5. Side-by-side photo comparison (renders/ficus_gt_vs_pred.png)
6. 360-degree video (renders/ficus_orbit_360.mp4 / .gif)
7. SafeTensors checkpoint export
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

from datasets.nerf_image_dataset import NeRFImageDataset
from models.unified_backbone import UnifiedRadianceBackbone
from engine.volume_renderer import VolumeRenderer
from engine.graph_debugger import MPSGraphDebugger
from render_video import render_orbit_video, export_point_cloud_ply


def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Device] Using Apple Silicon GPU Acceleration: {device}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU.")
    return device


def main():
    print("=" * 75)
    print("PRODUCTION NeRF TRAINING: REAL FICUS TREE (APPLE M4 PRO)")
    print("=" * 75)

    device = setup_device()
    debugger = MPSGraphDebugger(device=device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # 1. Load Real Images
    data_dir = "data/nerf_synthetic/ficus"
    dataset = NeRFImageDataset(
        data_dir=data_dir,
        split="train",
        target_size=100,  # 100x100 resolution
        max_views=25,     # 25 multi-view real photos
        device=device,
    )

    real_v0_np = (dataset.images[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite("renders/ficus_real_photo_0.png", cv2.cvtColor(real_v0_np, cv2.COLOR_RGB2BGR))

    # 2. 8-Layer NeRF Backbone (256 channels + Layer-4 Skip Connection)
    print("\n[Architecture] Initializing 8-Layer NeRF Backbone (256 channels, skip at layer 4)...")
    model = UnifiedRadianceBackbone(
        pos_freqs=10,
        dir_freqs=4,
        hidden_dim=256,
        num_layers=8,
        skip_layer=4,
        cond_dim=16,
    ).to(device)

    # Tight bounding sphere (2.3 - 5.7) with 80 stratified samples
    renderer = VolumeRenderer(near=2.3, far=5.7, num_samples=80, perturb=True)

    # 3. Setup AdamW Optimizer & Cosine Annealing Schedule
    num_steps = 3000
    batch_size = 1024

    print("[Optimizer] Setting up AdamW with Cosine Annealing Learning Rate...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"            Total Trainable Parameters: {total_params:,}")

    # 4. Training Loop
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)
    print(f"\n[Training] Optimizing for {num_steps} steps on Apple M4 Pro GPU...")
    model.train()

    start_time = time.perf_counter()

    for step in range(1, num_steps + 1):
        # Balanced batch: 50% tree rays, 50% sky rays
        batch_o, batch_d, batch_target = dataset.sample_batch(batch_size=batch_size)

        # FORWARD PASS
        out = renderer.render_rays(
            network=model,
            rays_o=batch_o,
            rays_d=batch_d,
            background_color=white_bg,
        )
        predicted_rgb = out["rgb"]
        opacity = out["opacity"]

        # LOSS: RGB reconstruction loss + Fog-elimination penalty
        loss_rgb = criterion(predicted_rgb, batch_target)

        bg_mask = (batch_target > 0.98).all(dim=-1)
        if bg_mask.any():
            loss_bg = torch.mean(opacity[bg_mask] ** 2)
        else:
            loss_bg = torch.tensor(0.0, device=device)

        total_loss = loss_rgb + 0.05 * loss_bg

        # BACKWARD PASS & ADAMW UPDATE
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == num_steps:
            elapsed = time.perf_counter() - start_time
            current_lr = scheduler.get_last_lr()[0]
            mem = debugger.get_memory_info()
            print(f"  Step [{step:04d}/{num_steps:04d}] | Loss: {loss_rgb.item():.5f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s | MPS Mem: {mem['allocated_mb']:.1f} MB")

    # 5. Visual Validation (Real Photo vs Neural Render)
    print("\n[Validation] Rendering View #0 vs Real Camera Photo...")
    model.eval()
    with torch.no_grad():
        v0_rays_o = dataset.rays_o[0].view(-1, 3)
        v0_rays_d = dataset.rays_d[0].view(-1, 3)

        chunk = 2048
        preds = []
        for i in range(0, v0_rays_o.shape[0], chunk):
            p_out = renderer.render_rays(
                network=model,
                rays_o=v0_rays_o[i : i + chunk],
                rays_d=v0_rays_d[i : i + chunk],
                background_color=white_bg,
            )
            preds.append(p_out["rgb"])

        pred_v0 = torch.cat(preds, dim=0).view(dataset.H, dataset.W, 3)
        pred_v0_np = (pred_v0.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Side-by-side comparison: [ Real Camera Photo | Neural Reconstructed View ]
        comparison = np.concatenate([real_v0_np, pred_v0_np], axis=1)
        comp_path = "renders/ficus_gt_vs_pred.png"
        cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"             Saved side-by-side comparison image -> {comp_path}")

    # 6. Save Model to SafeTensors
    checkpoint_path = "checkpoints/ficus_base_model.safetensors"
    model_state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(model_state, checkpoint_path, metadata={"dataset": "nerf_ficus", "architecture": "NeRF_8Layer_256D"})
    print(f"\n[SafeTensors] Saved base model checkpoint -> {checkpoint_path}")

    # 7. Render 360-Degree Video
    print("\n[Rendering] Rendering 360-degree orbit cinematic video...")
    render_orbit_video(
        network=model,
        renderer=renderer,
        output_dir="renders",
        video_name="ficus_orbit_360.mp4",
        num_frames=48,
        H=100,
        W=100,
        focal=dataset.focal,
        radius=4.03,
        elevation=25.0,
        fps=24,
        device=device,
    )

    # 8. Export 3D Point Cloud PLY
    ply_path = "exports/ficus_geometry.ply"
    export_point_cloud_ply(
        network=model,
        filepath=ply_path,
        grid_size=48,
        bound=1.2,
        density_threshold=5.0,
        device=device,
    )

    print("\n" + "=" * 75)
    print("PRODUCTION TRAINING COMPLETE!")
    print(f"1. Comparison Image: renders/ficus_gt_vs_pred.png")
    print(f"2. 360° Video:        renders/ficus_orbit_360.mp4 (and .gif)")
    print(f"3. 3D PLY Asset:      exports/ficus_geometry.ply")
    print(f"4. Model SafeTensors: checkpoints/ficus_base_model.safetensors")
    print("=" * 75)


if __name__ == "__main__":
    main()
