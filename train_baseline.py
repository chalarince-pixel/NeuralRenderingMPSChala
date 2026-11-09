"""
End-to-End Master Training & Verification Script
Executes Step 1 of the Neural Rendering Baseline on Apple Silicon M4 Pro:
1. Procedural 3D multi-view data generation
2. Base model training & .safetensors export
3. Pure PyTorch LoRA injection and adaptation
4. Forward/Backward autograd gradient verification
5. MPS kernel profiling & memory measurement
6. Full 360-degree .mp4 video rendering and .ply point cloud export
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

from models.unified_backbone import UnifiedRadianceBackbone
from models.lora import inject_lora, save_lora_safetensors, load_lora_safetensors
from engine.volume_renderer import VolumeRenderer
from engine.graph_debugger import MPSGraphDebugger
from datasets.synthetic_generator import generate_orbit_poses, generate_ground_truth_views
from render_video import render_orbit_video, export_point_cloud_ply


def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Hardware] Using Apple Silicon MPS GPU Acceleration: {device}")
    else:
        device = torch.device("cpu")
        print("[Hardware] MPS not available, falling back to CPU.")
    return device


def main():
    print("=" * 70)
    print("NEURAL RENDERING & DIFFUSION MPS BASELINE - STEP 1 VERIFICATION")
    print("=" * 70)

    # 1. Device and Environment Setup
    device = setup_device()
    debugger = MPSGraphDebugger(device=device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # 2. Generate Synthetic 3D Multi-View Dataset
    H, W = 64, 64
    focal = 64.0
    num_train_views = 8

    print(f"\n[1/6] Generating {num_train_views} procedural 3D multi-view training angles ({W}x{H})...")
    train_poses = generate_orbit_poses(num_train_views, radius=3.0, elevation=20.0, device=device)
    gt_images, gt_rays_o, gt_rays_d = generate_ground_truth_views(
        train_poses, H=H, W=W, focal=focal, season_style="summer", num_samples=32
    )

    # Flatten ray batches
    rays_o_flat = gt_rays_o.view(-1, 3)
    rays_d_flat = gt_rays_d.view(-1, 3)
    target_rgb_flat = gt_images.view(-1, 3)

    # Balanced Foreground / Background ray partition
    fg_mask = (target_rgb_flat < 0.95).any(dim=-1)
    fg_idx = torch.where(fg_mask)[0]
    bg_idx = torch.where(~fg_mask)[0]

    print(f"      Total Training Rays: {rays_o_flat.shape[0]:,} on {device}")
    print(f"      Foreground (Tree) Rays: {len(fg_idx):,} | Background (Sky) Rays: {len(bg_idx):,}")

    # Save Ground Truth reference image
    import cv2
    import numpy as np
    gt_v0_np = (gt_images[0].cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite("renders/ground_truth_view_0.png", cv2.cvtColor(gt_v0_np, cv2.COLOR_RGB2BGR))

    # 3. Instantiate Neural Models
    print("\n[2/6] Initializing Unified Radiance Backbone & Volume Raymarcher...")
    model = UnifiedRadianceBackbone(
        pos_freqs=8,
        dir_freqs=4,
        hidden_dim=128,
        num_layers=4,
        cond_dim=16,
    ).to(device)

    renderer = VolumeRenderer(near=1.5, far=4.5, num_samples=32, perturb=True)

    # Base training optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.MSELoss()

    # 4. Train Base Model (Geometry & Summer Foliage)
    print("\n[3/6] Pre-training Base Neural Radiance Field (600 steps)...")
    batch_size = 1024
    num_steps = 600
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    model.train()
    for step in range(1, num_steps + 1):
        # 50% foreground rays, 50% background rays to prevent background domination
        half_b = batch_size // 2
        idx_fg = fg_idx[torch.randint(0, len(fg_idx), (half_b,), device=device)]
        idx_bg = bg_idx[torch.randint(0, len(bg_idx), (half_b,), device=device)]
        idx = torch.cat([idx_fg, idx_bg])

        batch_o = rays_o_flat[idx]
        batch_d = rays_d_flat[idx]
        batch_target = target_rgb_flat[idx]

        out = renderer.render_rays(
            network=model,
            rays_o=batch_o,
            rays_d=batch_d,
            background_color=white_bg,
        )
        loss = criterion(out["rgb"], batch_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == num_steps:
            print(f"      Step [{step:03d}/{num_steps:03d}] - Reconstruction MSE Loss: {loss.item():.5f}")

    # Validate Base Model view 0 vs Ground Truth
    model.eval()
    with torch.no_grad():
        v0_o = gt_rays_o[0].view(-1, 3)
        v0_d = gt_rays_d[0].view(-1, 3)
        v0_out = renderer.render_rays(network=model, rays_o=v0_o, rays_d=v0_d, background_color=white_bg)
        pred_v0_np = (v0_out["rgb"].view(H, W, 3).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Side-by-side comparison image: GT | Predicted
        comparison = np.concatenate([gt_v0_np, pred_v0_np], axis=1)
        cv2.imwrite("renders/gt_vs_predicted.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print("      Saved side-by-side comparison image -> renders/gt_vs_predicted.png")

    # Save Base Model SafeTensors
    base_safetensors_path = "checkpoints/base_model.safetensors"
    base_state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(base_state, base_safetensors_path)
    print(f"      Saved Base Model -> {base_safetensors_path}")

    # 5. Inject LoRA for Style/Species Adaptation (Autumn Foliage Adaptation)
    print("\n[4/6] Injecting LoRA Modules for Fine-Tuning & Verifying Autograd Graph...")
    injected = inject_lora(model, target_names=["trunk_layers", "color_pre"], rank=8, alpha=1.0)
    model.to(device)
    print(f"      Injected LoRA layers: {list(injected.keys())}")

    # Verify trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Model Parameters:      {total_params:,}")
    print(f"      Trainable (LoRA) Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")

    # Generate Autumn Ground Truth for LoRA fine-tuning
    gt_autumn, _, _ = generate_ground_truth_views(
        train_poses, H=H, W=W, focal=focal, season_style="autumn", num_samples=32
    )
    autumn_rgb_flat = gt_autumn.view(-1, 3)
    fg_autumn_mask = (autumn_rgb_flat < 0.95).any(dim=-1)
    fg_autumn_idx = torch.where(fg_autumn_mask)[0]

    # Only optimize LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lora_optimizer = optim.AdamW(lora_params, lr=5e-3)

    model.train()
    for step in range(1, 201):
        half_b = batch_size // 2
        idx_fg = fg_autumn_idx[torch.randint(0, len(fg_autumn_idx), (half_b,), device=device)]
        idx_bg = bg_idx[torch.randint(0, len(bg_idx), (half_b,), device=device)]
        idx = torch.cat([idx_fg, idx_bg])

        batch_o = rays_o_flat[idx]
        batch_d = rays_d_flat[idx]
        batch_target = autumn_rgb_flat[idx]

        out = renderer.render_rays(
            network=model,
            rays_o=batch_o,
            rays_d=batch_d,
            background_color=white_bg,
        )
        lora_loss = criterion(out["rgb"], batch_target)

        lora_optimizer.zero_grad()
        lora_loss.backward()

        if step == 1:
            debugger.inspect_lora_gradients(model)

        lora_optimizer.step()

        if step % 50 == 0 or step == 200:
            print(f"      LoRA Step [{step:03d}/200] - Autumn Adaptation MSE Loss: {lora_loss.item():.5f}")

    # Benchmark MPS Kernel execution
    print("\n[5/6] Profiling MPS Kernels & Apple Silicon Memory...")
    def bench_forward():
        return renderer.render_rays(network=model, rays_o=batch_o, rays_d=batch_d, background_color=white_bg)

    def bench_backward(out):
        l = criterion(out["rgb"], batch_target)
        lora_optimizer.zero_grad()
        l.backward()

    debugger.benchmark_step(forward_fn=bench_forward, backward_fn=bench_backward, num_warmup=2, num_iters=10)

    # Save LoRA Adapter SafeTensors
    lora_safetensors_path = "checkpoints/lora_adapter.safetensors"
    save_lora_safetensors(model, lora_safetensors_path)

    # 6. Render Full 360-Degree Video & 3D Point Cloud
    print("\n[6/6] Rendering 360-Degree Cinematic .mp4 Video & Exporting 3D Asset...")
    render_orbit_video(
        network=model,
        renderer=renderer,
        output_dir="renders",
        video_name="orbit_360.mp4",
        num_frames=48,
        H=96,
        W=96,
        focal=96.0,
        radius=3.0,
        elevation=20.0,
        fps=24,
        device=device,
    )

    export_point_cloud_ply(
        network=model,
        filepath="exports/scene_geometry.ply",
        grid_size=36,
        bound=0.9,
        density_threshold=1.5,
        device=device,
    )

    print("\n" + "=" * 70)
    print("STEP 1 VERIFICATION COMPLETE!")
    print(f"1. Base Model Checkpoint: {base_safetensors_path}")
    print(f"2. LoRA Adapter:          {lora_safetensors_path}")
    print(f"3. 360° Cinematic Video:  renders/orbit_360.mp4")
    print(f"4. 3D Depth Pass Video:   renders/depth_pass.mp4")
    print(f"5. 3D Geometry PLY:       exports/scene_geometry.ply")
    print("=" * 70)


if __name__ == "__main__":
    main()
