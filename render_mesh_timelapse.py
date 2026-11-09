"""
Renders a 3D polygon mesh evolution timelapse from 60 .ply checkpoints.
Outputs MP4 and animated GIF with step annotations.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import cv2 as cv
from PIL import Image

def get_mesh_files(mesh_dir):
    files = sorted(glob.glob(os.path.join(mesh_dir, '*.ply')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    return files

def compute_global_bounds(mesh_files):
    # Sample last few meshes to determine bounding box
    mins, maxs = [], []
    for f in mesh_files[-5:]:
        m = trimesh.load(f)
        if len(m.vertices) > 0:
            mins.append(m.vertices.min(axis=0))
            maxs.append(m.vertices.max(axis=0))
    min_b = np.min(mins, axis=0)
    max_b = np.max(maxs, axis=0)
    # Expand slightly
    margin = (max_b - min_b) * 0.1
    return min_b - margin, max_b + margin

def render_mesh_frame(mesh_path, step, global_min, global_max, azim=45, elev=20, size=(800, 800)):
    mesh = trimesh.load(mesh_path)
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    fig.patch.set_facecolor('#0d1117') # Sleek dark background
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')

    if len(mesh.faces) > 0 and len(mesh.vertices) > 0:
        v = mesh.vertices
        f = mesh.faces
        tris = v[f]

        # Normal-based Phong shading
        normals = mesh.face_normals
        light = np.array([0.4, 0.4, 0.8])
        light = light / np.linalg.norm(light)
        diffuse = np.clip(np.dot(normals, light) * 0.5 + 0.5, 0.15, 1.0)
        
        # Color gradient based on height (z)
        z_norm = (v[:, 2].mean() - global_min[2]) / (global_max[2] - global_min[2] + 1e-5)
        
        # Organic botanical palette (warm green for leaves, earthy bronze for stem/pot)
        colors = np.zeros((len(f), 4))
        colors[:, 0] = 0.25 * diffuse   # R
        colors[:, 1] = 0.75 * diffuse   # G
        colors[:, 2] = 0.45 * diffuse   # B
        colors[:, 3] = 0.95             # Alpha

        poly = Poly3DCollection(tris, facecolors=colors, edgecolors='#1e2d24', linewidths=0.2)
        ax.add_collection3d(poly)

    # Set standardized bounds across all frames
    ax.set_xlim([global_min[0], global_max[0]])
    ax.set_ylim([global_min[1], global_max[1]])
    ax.set_zlim([global_min[2], global_max[2]])
    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')
    
    # Render to numpy image
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    # Add overlay text using OpenCV
    h, w = rgb.shape[:2]
    # Header bar
    cv.rectangle(rgb, (0, 0), (w, 55), (20, 24, 30), -1)
    cv.putText(rgb, f"CMU Stochastic Geometry - 3D Polygon Mesh Evolution", 
               (20, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 210, 225), 2)
    cv.putText(rgb, f"Step: {step:,} / 300,000", 
               (20, 46), cv.FONT_HERSHEY_SIMPLEX, 0.55, (70, 220, 130), 2)
    
    # Vertex / Face count info
    n_v = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
    n_f = len(mesh.faces) if hasattr(mesh, 'faces') else 0
    cv.putText(rgb, f"Vertices: {n_v:,} | Faces: {n_f:,}", 
               (w - 300, 46), cv.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 190), 1)

    # Progress bar at bottom
    progress = step / 300000.0
    bar_w = int(w * progress)
    cv.rectangle(rgb, (0, h - 6), (w, h), (30, 35, 45), -1)
    cv.rectangle(rgb, (0, h - 6), (bar_w, h), (70, 220, 130), -1)

    return rgb

def main():
    mesh_dir = "cmu_reference/exp/nerf_ficus/gaussian_linearmixture_spatial_white/meshes"
    out_dir = "renders"
    os.makedirs(out_dir, exist_ok=True)

    files = get_mesh_files(mesh_dir)
    print(f"Found {len(files)} mesh files.")
    if not files:
        print("No meshes found!")
        return

    min_b, max_b = compute_global_bounds(files)
    print("Global bounds:", min_b, max_b)

    frames = []
    print("Rendering frames...")
    for idx, f in enumerate(files):
        step = int(os.path.basename(f).split('.')[0])
        # Smooth slight turntable rotation while evolving: 50 degrees to 85 degrees
        azim = 35 + (idx / len(files)) * 40
        frame_rgb = render_mesh_frame(f, step, min_b, max_b, azim=azim, elev=18)
        frames.append(frame_rgb)
        print(f"[{idx+1}/{len(files)}] Step {step} rendered.")

    # Save MP4 video
    mp4_path = os.path.join(out_dir, "cmu_ficus_mesh_evolution.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(mp4_path, fourcc, 10, (w, h))
    for f in frames:
        # Convert RGB to BGR for OpenCV
        out_video.write(cv.cvtColor(f, cv.COLOR_RGB2BGR))
    # Hold final frame for 2 seconds (20 frames at 10 fps)
    for _ in range(20):
        out_video.write(cv.cvtColor(frames[-1], cv.COLOR_RGB2BGR))
    out_video.release()
    print(f"Saved MP4 video to {mp4_path}")

    # Save animated GIF (every frame, with small size for snappy preview)
    gif_path = os.path.join(out_dir, "cmu_ficus_mesh_evolution.gif")
    pil_frames = [Image.fromarray(cv.resize(f, (500, 500))) for f in frames]
    # Add pause at the end
    for _ in range(10):
        pil_frames.append(pil_frames[-1])
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], 
                       duration=100, loop=0)
    print(f"Saved GIF to {gif_path}")

if __name__ == '__main__':
    main()
