import logging
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Polygon Soup Pipeline API")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input" / "web_session"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "web_session"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "web_session"
FRONTEND_DIR = PROJECT_ROOT / "webapp" / "frontend"

@app.post("/api/reconstruct")
async def reconstruct(
    mode: str = Form(...),
    image: UploadFile = File(None),
    front: UploadFile = File(None),
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None)
):
    """
    Runs the 3D reconstruction pipeline.
    Routes to CRM for single mode, Unique3D for multi mode.
    """
    logger.info(f"Received request in mode: {mode}")

    # 1. Prepare directories
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Save uploaded images
    if mode == "single":
        if not image:
            return JSONResponse(status_code=400, content={"error": "No image provided for single mode"})
        front_path = INPUT_DIR / "front.png"
        with open(front_path, "wb") as f:
            f.write(await image.read())
        # For CRM, we just duplicate it to satisfy the 4-file check in stage0,
        # but CRM will only use front.png in stage2.
        for view in ["back.png", "left.png", "right.png"]:
            shutil.copy(front_path, INPUT_DIR / view)
        backend = "crm"
    else:
        if not all([front, back, left, right]):
            return JSONResponse(status_code=400, content={"error": "All 4 views required for multi mode"})
        for file_obj, name in [(front, "front.png"), (back, "back.png"), (left, "left.png"), (right, "right.png")]:
            with open(INPUT_DIR / name, "wb") as f:
                f.write(await file_obj.read())
        backend = "unique3d"

    # 3. Invoke the real pipeline
    cmd = [
        "python", str(PROJECT_ROOT / "main.py"),
        "--input", str(INPUT_DIR),
        "--output", str(OUTPUT_DIR),
        "--backend", backend
    ]

    logger.info(f"Running pipeline: {' '.join(cmd)}")

    try:
        # Timeout 30 minutes for real pipeline
        process = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800)

        if process.returncode != 0:
            logger.error(f"Pipeline failed: {process.stderr}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Pipeline execution failed",
                    "details": process.stderr[-500:],
                }
            )

        usd_file = OUTPUT_DIR / "scene_y_up.glb"
        if not usd_file.exists():
            return JSONResponse(status_code=500, content={"error": "Pipeline succeeded but output GLB not found."})

        return {"status": "success", "message": "Reconstruction complete", "download_url": "/api/download"}

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"error": "Pipeline execution timed out."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/download")
async def download_result():
    usd_file = OUTPUT_DIR / "scene_y_up.glb"
    if not usd_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path=usd_file, filename="reconstructed_scene.glb", media_type="model/gltf-binary")

# Serve static frontend files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
