#!/usr/bin/env python3
"""
Lightweight REST API Microservice for 3D Mesh Generation.
Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import io
import os
import torch
import trimesh
from fastapi import FastAPI, Response, Query
from generate_mesh import load_sdf_model, extract_mesh_on_the_fly

app = FastAPI(
    title="3D Neural SDF Mesh Generator API",
    description="Generates on-the-fly .ply polygon meshes from trained neural weights.",
    version="1.0.0"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "ficus_model_300k.pth")

# Pre-load model at startup
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Initializing Neural Model on device: {device}")
model = load_sdf_model(CHECKPOINT_PATH, device)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "ficus_300k",
        "device": str(device)
    }

@app.get("/generate-mesh")
def generate_mesh(
    resolution: int = Query(default=128, ge=32, le=256, description="3D grid resolution (32 to 256)")
):
    """Generates and streams a .ply 3D mesh directly in response."""
    mesh = extract_mesh_on_the_fly(model, device, resolution=resolution)
    
    # Export to memory buffer
    buf = io.BytesIO()
    mesh.export(buf, file_type="ply")
    buf.seek(0)
    
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=ficus_mesh_res{resolution}.ply",
            "X-Vertex-Count": str(len(mesh.vertices)),
            "X-Face-Count": str(len(mesh.faces))
        }
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
