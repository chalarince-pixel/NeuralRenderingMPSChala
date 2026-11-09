"""
Downloader for the NeRF Ficus Tree Multi-View Dataset.
Fetches real multi-view images (r_0.png, r_1.png, ...) and camera pose matrices
directly from the public NeRF synthetic repository on HuggingFace.
"""

import os
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor

BASE_HF_URL = "https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror/resolve/main"
TARGET_DIR = "data/nerf_synthetic/ficus"


def download_file(rel_path: str, dest_dir: str):
    url = f"{BASE_HF_URL}/{rel_path}"
    local_path = os.path.join(dest_dir, rel_path.replace("ficus/", ""))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        return local_path
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")
        return None


def fetch_ficus_dataset(num_train_views: int = 25):
    """
    Downloads transforms_train.json and the first `num_train_views` real multi-view photos.
    """
    print(f"[Dataset] Downloading Ficus Tree Dataset to -> {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. Download transforms_train.json
    print("[Dataset] Fetching camera pose matrices (transforms_train.json)...")
    transforms_path = download_file("ficus/transforms_train.json", TARGET_DIR)
    
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    print(f"[Dataset] Found {len(frames)} total camera frames in transforms_train.json.")

    # Select subset of views for fast, high-quality training
    selected_frames = frames[:num_train_views]
    file_list = []
    for frame in selected_frames:
        file_path = frame["file_path"]
        if not file_path.endswith(".png"):
            file_path = f"{file_path}.png"
        if file_path.startswith("./"):
            file_path = file_path[2:]
        file_list.append(f"ficus/{file_path}")

    # 2. Concurrently download selected images
    print(f"[Dataset] Downloading {len(file_list)} real multi-view photos...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_file, rel_file, TARGET_DIR) for rel_file in file_list]
        for f in futures:
            f.result()

    print(f"[Dataset] Successfully downloaded {len(file_list)} real Ficus images into {TARGET_DIR}!")
    return TARGET_DIR


if __name__ == "__main__":
    fetch_ficus_dataset(num_train_views=25)
