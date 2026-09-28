"""HTTP API that classifies skin lesion photos.

Run on the local network (reachable from the phone):
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
Interactive docs: http://localhost:8000/docs
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from vibe_skin.config import load_config
from vibe_skin.predict import Predictor, load_image

DISCLAIMER = (
    "Research prototype. Not a medical device and not a diagnosis. "
    "See a dermatologist about any lesion that worries you or changes."
)

cfg = load_config(os.environ.get("VIBE_SKIN_API_CONFIG", "configs/api.yaml"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor(cfg["run_dir"])
    yield
    state.clear()


app = FastAPI(title="vibe_skin API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    predictor = state["predictor"]
    return {"status": "ok", "model": predictor.name, "device": str(predictor.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Photo of the lesion (JPEG or PNG)")) -> dict:
    data = await file.read()
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {cfg['max_upload_mb']} MB")
    try:
        image = load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the file as an image")

    t0 = time.perf_counter()
    result = state["predictor"].predict(image)
    return {
        **result,
        "model": state["predictor"].name,
        "inference_ms": round(1000 * (time.perf_counter() - t0), 1),
        "disclaimer": DISCLAIMER,
    }
