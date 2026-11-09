"""
State-of-the-Art Neural Rendering: Hash-Encoded SDF (Neuralangelo / Instant-NSR).
Unifies:
1. Multi-Resolution 3D Hash Grid (Instant-NGP) for sub-millimeter leaf details & zero spectral bias
2. Signed Distance Function S(x) (NeuS / VolSDF / CMU formulation)
3. Eikonal regularization loss ||grad S|| = 1 (eradicates floating fog & enforces solid boundaries)
4. AdamW optimization on Apple Silicon M4 Pro GPU
"""

import os
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

from datasets.nerf_image_dataset import NeRFImageDataset
from models.sdf_backbone import HashSDFBackbone
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="ficus", choices=["ficus", "lego"])
    parser.add_argument("--size", type=int, default=400, help="High-resolution image size (e.g. 400x400)")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Rays per batch")
    args = parser.parse_args()

    print("=" * 80)
    print(f"HIGH-RESOLUTION HASH-ENCODED SDF ({args.size}x{args.size}): SCENE='{args.scene.upper()}'")
    print("=" * 80)

    device = setup_device()
    debugger = MPSGraphDebugger(device=device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # 1. Load Dataset at High Resolution (400x400)
    data_dir = f"data/nerf_synthetic/{args.scene}"
    dataset = NeRFImageDataset(
        data_dir=data_dir,
        split="train",
        target_size=args.size,
        max_views=25,
        device=device,
    )

    real_v0_np = (dataset.images[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f"renders/{args.scene}_real_photo_0.png", cv2.cvtColor(real_v0_np, cv2.COLOR_RGB2BGR))

    # 2. Hash-SDF Backbone
    print("\n[Architecture] Initializing Multi-Resolution Hash-Grid + SDF Backbone...")
    model = HashSDFBackbone(
        bounding_box=1.5,
        num_levels=12,
        features_per_level=2,
        log2_hashmap_size=15,
        hidden_dim=64,
        init_sharpness=20.0,
    ).to(device)

    renderer = VolumeRenderer(near=2.2, far=5.8, num_samples=64, perturb=True)

    # 3. Setup Separate Parameter Groups for AdamW
    # Hash table parameters train faster; MLPs train smoothly
    hash_params = list(model.encoder.parameters())
    mlp_params = (
        list(model.sdf_net.parameters())
        + list(model.color_net.parameters())
        + [model.inv_s]
    )

    optimizer = optim.AdamW([
        {"params": hash_params, "lr": 5e-3, "weight_decay": 0.0},
        {"params": mlp_params, "lr": 1e-3, "weight_decay": 1e-4},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)
    criterion = nn.L1Loss()  # L1 loss is sharper on leaf edges than MSE

    total_params = sum(p.numel() for p in model.parameters())
    print(f"            Total Parameters: {total_params:,} (Hash Grid: {sum(p.numel() for p in hash_params):,})")

    # 4. Training Loop with Eikonal & Opacity Regularization
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)
    print(f"\n[Training] Optimizing {args.steps} steps with Eikonal Surface Loss on M4 Pro...")
    model.train()

    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        batch_o, batch_d, batch_target = dataset.sample_batch(batch_size=args.batch_size)

        # FORWARD PASS: Volume Raymarching
        out = renderer.render_rays(
            network=model,
            rays_o=batch_o,
            rays_d=batch_d,
            background_color=white_bg,
        )
        predicted_rgb = out["rgb"]
        opacity = out["opacity"]

        # 1. Primary RGB Reconstruction Loss (L1)
        loss_rgb = criterion(predicted_rgb, batch_target)

        # 2. Background Fog Penalty (Empty sky rays must have zero opacity)
        bg_mask = (batch_target > 0.98).all(dim=-1)
        loss_bg = torch.mean(opacity[bg_mask] ** 2) if bg_mask.any() else torch.tensor(0.0, device=device)

        # 3. Eikonal Loss: Every 4 steps, regularize sample points to enforce true solid distance field ||grad S|| = 1
        if step % 4 == 0:
            sample_pts = batch_o.unsqueeze(1) + batch_d.unsqueeze(1) * 3.5  # near center
            loss_eikonal = model.compute_eikonal_loss(sample_pts.view(-1, 3)[:256])
        else:
            loss_eikonal = torch.tensor(0.0, device=device)

        total_loss = loss_rgb + 0.05 * loss_bg + 0.05 * loss_eikonal

        # BACKWARD PASS & ADAMW STEP
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 250 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start_time
            sharpness = torch.exp(model.inv_s).item()
            mem = debugger.get_memory_info()
            print(f"  Step [{step:04d}/{args.steps:04d}] | L1 Loss: {loss_rgb.item():.5f} | Sharpness s: {sharpness:.1f} | Elapsed: {elapsed:.1f}s | MPS Mem: {mem['allocated_mb']:.1f} MB")

    # 5. Visual Validation View 0
    print(f"\n[Validation] Rendering View #0 Comparison for '{args.scene}'...")
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

        # Side-by-side comparison: [ Real Camera Photo | Hash-SDF Render ]
        comparison = np.concatenate([real_v0_np, pred_v0_np], axis=1)
        comp_path = f"renders/{args.scene}_hash_sdf_comp.png"
        cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"             Saved side-by-side comparison image -> {comp_path}")

    # 6. Save Model Checkpoint
    checkpoint_path = f"checkpoints/{args.scene}_hash_sdf.safetensors"
    model_state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(model_state, checkpoint_path, metadata={"scene": args.scene, "architecture": "HashSDFBackbone"})
    print(f"\n[SafeTensors] Saved Hash-SDF model checkpoint -> {checkpoint_path}")

    # 7. Render 360-Degree Video
    print("\n[Rendering] Rendering 360-degree orbit cinematic video...")
    render_orbit_video(
        network=model,
        renderer=renderer,
        output_dir="renders",
        video_name=f"{args.scene}_hash_sdf_orbit.mp4",
        num_frames=48,
        H=args.size,
        W=args.size,
        focal=dataset.focal,
        radius=4.03,
        elevation=25.0,
        fps=24,
        device=device,
    )

    # 8. Export 3D Point Cloud PLY
    ply_path = f"exports/{args.scene}_hash_sdf.ply"
    export_point_cloud_ply(
        network=model,
        filepath=ply_path,
        grid_size=48,
        bound=1.2,
        density_threshold=5.0,
        device=device,
    )

    print("\n" + "=" * 80)
    print("HASH-ENCODED SDF RECONSTRUCTION COMPLETE!")
    print(f"1. Comparison Image: renders/{args.scene}_hash_sdf_comp.png")
    print(f"2. 360° Video:        renders/{args.scene}_hash_sdf_orbit.mp4 (and .gif)")
    print(f"3. 3D Mesh PLY:       exports/{args.scene}_hash_sdf.ply")
    print(f"4. Model SafeTensors: checkpoints/{args.scene}_hash_sdf.safetensors")
    print("=" * 80)


if __name__ == "__main__":
    main()
