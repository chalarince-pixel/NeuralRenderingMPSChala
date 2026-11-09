#!/usr/bin/env python3
"""
Prepares Meta Multiface Multi-Camera Dome Data for Neural SDF Surface Reconstruction.
Packs 38 calibrated camera angles at frozen instant t=0 (E001_Neutral_Eyes_Open, frame 000102)
into cameras_sphere.npz, image/*.png, and mask/*.png.
"""
import os
import glob
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIFACE_DIR = os.path.join(SCRIPT_DIR, "public_data", "multiface_6795937")
KRT_PATH = os.path.join(MULTIFACE_DIR, "m--20180227--0000--6795937--GHS", "KRT")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "public_data", "human_face_multiface")

def load_krt(path):
    cameras = {}
    with open(path, "r") as f:
        while True:
            name = f.readline().strip()
            if not name:
                break
            intrin = np.array([[float(x) for x in f.readline().split()] for _ in range(3)], dtype=np.float32)
            dist = np.array([float(x) for x in f.readline().split()], dtype=np.float32)
            extrin = np.array([[float(x) for x in f.readline().split()] for _ in range(3)], dtype=np.float32)
            f.readline()  # blank separator line
            cameras[name] = {
                "intrin": intrin,
                "dist": dist,
                "extrin": extrin,
            }
    return cameras

def build_dataset(selected_frame="000102"):
    print(f"Loading KRT camera matrices from: {KRT_PATH}")
    cameras = load_krt(KRT_PATH)
    sorted_cam_ids = sorted(list(cameras.keys()))
    print(f"Found {len(sorted_cam_ids)} calibrated camera viewpoints in Meta dome.")

    img_dir_out = os.path.join(OUTPUT_DIR, "image")
    mask_dir_out = os.path.join(OUTPUT_DIR, "mask")
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)

    # Compute camera centers to determine head center
    cam_centers = []
    for cam_id in sorted_cam_ids:
        Rt = cameras[cam_id]["extrin"]
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        c = -R.T @ t
        cam_centers.append(c)
    cam_centers = np.array(cam_centers)

    head_center = np.mean(cam_centers, axis=0)
    radii = np.linalg.norm(cam_centers - head_center, axis=1)
    dome_radius = np.mean(radii)
    print(f"Dome center: {head_center}, Dome radius: {dome_radius:.2f} mm")

    # Scale factor mapping head to unit sphere [-1, 1]
    s_world = 250.0  # mm per unit radius

    # Image resize factor
    orig_w, orig_h = 1334.0, 2048.0
    target_w = 800
    s_img = target_w / orig_w
    target_h = int(orig_h * s_img)

    camera_dict = {}
    extracted_count = 0

    for idx, cam_id in enumerate(sorted_cam_ids):
        cam_data = cameras[cam_id]
        K = cam_data["intrin"].copy()
        Rt = cam_data["extrin"].copy()
        R = Rt[:3, :3]
        t = Rt[:3, 3]

        # Scale K for resized image
        K[0, :] *= s_img
        K[1, :] *= s_img

        # Normalized projection matrix: P_norm = K_scaled @ [s_world * R | R @ head_center + t]
        P_norm = K @ np.hstack([s_world * R, (R @ head_center + t)[:, None]])
        world_mat = np.eye(4, dtype=np.float32)
        world_mat[:3, :4] = P_norm
        scale_mat = np.eye(4, dtype=np.float32)

        camera_dict[f"world_mat_{idx}"] = world_mat
        camera_dict[f"scale_mat_{idx}"] = scale_mat

        # Process image and mask
        src_path = os.path.join(MULTIFACE_DIR, "m--20180227--0000--6795937--GHS", "images", "E001_Neutral_Eyes_Open", cam_id, f"{selected_frame}.png")
        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            if img is not None:
                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                out_img_path = os.path.join(img_dir_out, f"{idx:06d}.png")
                cv2.imwrite(out_img_path, img_resized)

                # Clean silhouette mask
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                out_mask_path = os.path.join(mask_dir_out, f"{idx:06d}.png")
                cv2.imwrite(out_mask_path, mask_clean)
                extracted_count += 1

    npz_out = os.path.join(OUTPUT_DIR, "cameras_sphere.npz")
    np.savez(npz_out, **camera_dict)
    print(f"Saved {len(sorted_cam_ids)} normalized camera matrices to: {npz_out}")
    print(f"Processed {extracted_count}/{len(sorted_cam_ids)} multi-view images and masks.")

if __name__ == "__main__":
    build_dataset()
