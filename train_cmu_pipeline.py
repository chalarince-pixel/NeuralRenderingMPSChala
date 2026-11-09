"""
Full CMU / NeuS Neural Surface Rendering Pipeline on Apple Silicon M4 Pro.
Implements the exact methodology of "Objects as volumes: A stochastic geometry view of opaque solids":
1. Ingests all 100 camera views
2. Multi-Resolution Hash Grid (Instant-NGP) + Signed Distance Function S(x)
3. Zero-crossing surface-finding ray sampler (dense samples directly on leaf surfaces)
4. Binary Silhouette Mask Loss (BCE against ground-truth alpha masks)
5. Physical Eikonal Regularization Loss ||grad S|| = 1
6. High-definition rendering & 360-degree video export
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

from datasets.cmu_dataset import CMUNeRFDataset
from models.sdf_backbone import HashSDFBackbone
from engine.cmu_sampler import CMUSurfaceRenderer
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
    parser.add_argument("--scene", type=str, default="ficus")
    parser.add_argument("--resolution", type=int, default=400, help="Native image resolution")
    parser.add_argument("--views", type=int, default=100, help="Number of camera views")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Ray batch size")
    args = parser.parse_args()

    print("=" * 80)
    print(f"CMU / NeuS RECONSTRUCTION PIPELINE: SCENE='{args.scene.upper()}' (APPLE M4 PRO)")
    print(f"Configuration: {args.views} Views | {args.resolution}x{args.resolution} Resolution | {args.steps} Steps")
    print("=" * 80)

    device = setup_device()
    debugger = MPSGraphDebugger(device=device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # 1. Ingest All 100 Views with Exact Binary Silhouette Masks
    data_dir = f"data/nerf_synthetic/{args.scene}"
    dataset = CMUNeRFDataset(
        data_dir=data_dir,
        split="train",
        resolution=args.resolution,
        max_views=args.views,
        device=device,
    )

    real_v0_np = (dataset.images[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f"renders/cmu_{args.scene}_real_photo_0.png", cv2.cvtColor(real_v0_np, cv2.COLOR_RGB2BGR))

    # 2. Hash-SDF Backbone (Multi-resolution hash table + SDF MLP)
    print("\n[Architecture] Initializing Multi-Resolution Hash Grid + Signed Distance Function Backbone...")
    model = HashSDFBackbone(
        bounding_box=1.5,
        num_levels=12,
        features_per_level=2,
        log2_hashmap_size=15,
        hidden_dim=64,
        init_sharpness=20.0,
    ).to(device)

    # 3. Surface-Finding Point Sampler (CMU / NeuS strategy)
    renderer = CMUSurfaceRenderer(
        near=2.2,
        far=5.8,
        num_coarse_probes=64,
        num_fine_surface_samples=32,
    ).to(device)

    # 4. Optimizer: AdamW with Cosine Annealing Schedule
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

    # Loss Functions: L1 RGB loss + Binary Cross Entropy Mask loss
    criterion_rgb = nn.L1Loss()
    criterion_mask = nn.BCELoss()

    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    # 5. Training Loop
    print(f"\n[Training] Optimizing {args.steps} steps with CMU Silhouette Mask & Eikonal Loss on M4 Pro...")
    model.train()
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        # Sample balanced batch (50% tree rays, 50% sky rays)
        batch_o, batch_d, batch_target_rgb, batch_target_mask = dataset.sample_batch(batch_size=args.batch_size)

        # FORWARD PASS: Surface-finding raymarching
        out = renderer.render_rays(
            network=model,
            rays_o=batch_o,
            rays_d=batch_d,
            background_color=white_bg,
        )
        pred_rgb = out["rgb"]
        pred_opacity = out["opacity"]

        # 1. RGB Reconstruction Loss
        loss_rgb = criterion_rgb(pred_rgb, batch_target_rgb)

        # 2. Binary Silhouette Mask Loss (CMU / NeuS key)
        loss_mask = criterion_mask(pred_opacity.clamp(1e-5, 1.0 - 1e-5), batch_target_mask)

        # 3. Physical Eikonal Surface Loss ||grad S|| = 1 (every 4 steps)
        if step % 4 == 0:
            sample_pts = batch_o.unsqueeze(1) + batch_d.unsqueeze(1) * 3.5
            loss_eikonal = model.compute_eikonal_loss(sample_pts.view(-1, 3)[:256])
        else:
            loss_eikonal = torch.tensor(0.0, device=device)

        # Total Objective
        total_loss = loss_rgb + 0.1 * loss_mask + 0.05 * loss_eikonal

        # BACKWARD PASS & ADAMW UPDATE
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start_time
            sharpness = torch.exp(model.inv_s).item()
            mem = debugger.get_memory_info()
            print(f"  Step [{step:04d}/{args.steps:04d}] | RGB Loss: {loss_rgb.item():.5f} | Mask Loss: {loss_mask.item():.5f} | Sharpness s: {sharpness:.1f} | Elapsed: {elapsed:.1f}s | MPS Mem: {mem['allocated_mb']:.1f} MB")

    # 6. Visual Evaluation View #0
    print(f"\n[Validation] Rendering View #0 at {args.resolution}x{args.resolution}...")
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

        pred_v0 = torch.cat(preds, dim=0).view(args.resolution, args.resolution, 3)
        pred_v0_np = (pred_v0.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # High-resolution side-by-side comparison: [ Real Camera Photo | CMU-SDF Reconstruction ]
        comparison = np.concatenate([real_v0_np, pred_v0_np], axis=1)
        comp_path = f"renders/cmu_{args.scene}_comp_{args.resolution}.png"
        cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"             Saved side-by-side comparison ({args.resolution}x{args.resolution * 2}) -> {comp_path}")

    # 7. Save Model Checkpoint
    checkpoint_path = f"checkpoints/cmu_{args.scene}_model.safetensors"
    model_state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(model_state, checkpoint_path, metadata={"scene": args.scene, "method": "CMU_NeuS_SDF"})
    print(f"\n[SafeTensors] Saved CMU-SDF model checkpoint -> {checkpoint_path}")

    # 8. Render 360-Degree Video
    print("\n[Rendering] Rendering 360-degree orbit cinematic video...")
    render_orbit_video(
        network=model,
        renderer=renderer,
        output_dir="renders",
        video_name=f"cmu_{args.scene}_orbit.mp4",
        num_frames=48,
        H=args.resolution,
        W=args.resolution,
        focal=dataset.focal,
        radius=4.03,
        elevation=25.0,
        fps=24,
        device=device,
    )

    # 9. Export 3D Mesh PLY
    ply_path = f"exports/cmu_{args.scene}_geometry.ply"
    export_point_cloud_ply(
        network=model,
        filepath=ply_path,
        grid_size=48,
        bound=1.2,
        density_threshold=5.0,
        device=device,
    )

    print("\n" + "=" * 80)
    print("CMU / NeuS RECONSTRUCTION COMPLETE!")
    print(f"1. Comparison Image: renders/cmu_{args.scene}_comp_{args.resolution}.png")
    print(f"2. 360° Video:        renders/cmu_{args.scene}_orbit.mp4 (and .gif)")
    print(f"3. 3D Surface PLY:    exports/cmu_{args.scene}_geometry.ply")
    print(f"4. SafeTensors Model: checkpoints/cmu_{args.scene}_model.safetensors")
    print("=" * 80)


if __name__ == "__main__":
    main()
