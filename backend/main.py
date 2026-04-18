"""FastAPI server: POST /api/identify for the frontend."""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import deps  # noqa: F401
from .identification import identify_from_pil, outcome_to_api_dict

app = FastAPI(title="Invasive species ID", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file (JPEG, PNG, etc.).")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}") from e

    outcome = identify_from_pil(image)
    payload = outcome_to_api_dict(outcome)
    if not payload.get("ok"):
        # Return 200 with structured error so the UI can show the message without treating it as HTTP failure
        return payload
    return payload
