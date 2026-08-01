import os
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

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
async def reconstruct(image: UploadFile = File(...)):
    """
    Accepts a single image, prepares the 4 required orthogonal views (by mocking for the demo),
    and runs the 3D reconstruction pipeline.
    """
    logger.info(f"Received image: {image.filename}")
    
    # 1. Prepare directories
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Save the uploaded image as front view
    content = await image.read()
    front_path = INPUT_DIR / "front.png"
    with open(front_path, "wb") as f:
        f.write(content)

    # 3. For this demo, since we require 4 views and don't have a 2D-to-3D multi-view generator 
    # running, we will duplicate the image for the other 3 views just to satisfy the pipeline input.
    for view in ["back.png", "left.png", "right.png"]:
        shutil.copy(front_path, INPUT_DIR / view)

    # 4. Invoke the pipeline
    cmd = [
        "python", str(PROJECT_ROOT / "main.py"),
        "--input", str(INPUT_DIR),
        "--output", str(OUTPUT_DIR)
    ]
    
    logger.info(f"Running pipeline: {' '.join(cmd)}")
    
    try:
        # We run it with a timeout, and capture output
        process = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
        
        if process.returncode != 0:
            logger.error(f"Pipeline failed: {process.stderr}")
            # If the failure is due to missing torch/models (which is currently true for Python 3.13),
            # we will return a 500 error with the reason.
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Pipeline execution failed",
                    "details": process.stderr[-500:], # Last 500 chars
                    "hint": "Ensure torch is installed and models are downloaded in external/ and checkpoints/"
                }
            )
            
        # Success! Return the URL to download the USD file
        # Check which axis was generated, default to y_up
        usd_file = OUTPUT_DIR / "scene_y_up.usda"
        if not usd_file.exists():
            return JSONResponse(status_code=500, content={"error": "Pipeline succeeded but output USD not found."})
            
        return {"status": "success", "message": "Reconstruction complete", "download_url": "/api/download"}
        
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"error": "Pipeline execution timed out."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/download")
async def download_result():
    usd_file = OUTPUT_DIR / "scene_y_up.usda"
    if not usd_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path=usd_file, filename="reconstructed_scene.usda")

# Serve static frontend files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
