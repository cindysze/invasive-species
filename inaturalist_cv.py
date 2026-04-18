"""Call iNaturalist Computer Vision API (species suggestions from your photo).

This uses the official ``POST /v1/computervision/score_image`` endpoint — a vision model,
not an LLM. You must authenticate with a JWT (see README in repo or env docs below).

API reference context: https://api.inaturalist.org/v1/docs/
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from config import DEFAULT_OBS_LAT, DEFAULT_OBS_LNG, INAT_COMPUTER_VISION_URL, PLANTAE_TAXON_ID
from plant_metadata import invasive_label

# Same taxon_metadata.json as the local model path (optional enrichment for known taxa).

_MODELS = Path(__file__).resolve().parent / "models" / "taxon_metadata.json"


def _load_taxon_meta() -> dict[str, Any]:
    if not _MODELS.is_file():
        return {}
    with open(_MODELS, encoding="utf-8") as f:
        return json.load(f)


def _auth_headers() -> dict[str, str]:
    token = (os.environ.get("INAT_JWT") or os.environ.get("INATURALIST_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "Set INAT_JWT to your iNaturalist JSON Web Token. "
            "While logged in at inaturalist.org, open https://www.inaturalist.org/users/api_token "
            "(token expires about every 24 hours)."
        )
    # Official examples use the raw JWT in Authorization; some clients use Bearer.
    if token.lower().startswith("bearer "):
        auth = token
    else:
        auth = f"Bearer {token}"
    return {"Authorization": auth, "Accept": "application/json"}


def score_image(
    image: Image.Image,
    *,
    lat: float | None = None,
    lng: float | None = None,
    plants_only: bool = True,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Upload an image and return the parsed JSON from iNaturalist (``results``, ``common_ancestor``, etc.).
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    headers = _auth_headers()
    files = {"image": ("upload.jpg", buf.getvalue(), "image/jpeg")}
    if lat is None:
        lat = float(os.environ["INAT_LAT"]) if os.environ.get("INAT_LAT") else DEFAULT_OBS_LAT
    if lng is None:
        lng = float(os.environ["INAT_LNG"]) if os.environ.get("INAT_LNG") else DEFAULT_OBS_LNG
    data: dict[str, str] = {
        "lat": str(lat),
        "lng": str(lng),
    }
    if plants_only:
        data["taxon_id"] = str(PLANTAE_TAXON_ID)

    r = requests.post(
        INAT_COMPUTER_VISION_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=timeout,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "iNaturalist returned 401 Unauthorized. Refresh INAT_JWT at "
            "https://www.inaturalist.org/users/api_token (log in first)."
        )
    r.raise_for_status()
    return r.json()


def _score_to_display(score: Any) -> float:
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def suggestions_for_app(image: Image.Image, top_k: int = 8) -> list[dict[str, Any]]:
    """
    Returns rows compatible with the Gradio formatter: confidence, names, invasive text, source tag.
    """
    raw = score_image(image)
    meta = _load_taxon_meta()
    rows: list[dict[str, Any]] = []

    for r in (raw.get("results") or [])[:top_k]:
        t = r.get("taxon") or {}
        tid = str(t.get("id", ""))
        sci = (t.get("name") or "").strip()
        common = (t.get("preferred_common_name") or "").strip()
        # After API processing, scores may appear under different keys (combined_score is common).
        raw = _score_to_display(
            r.get("combined_score")
            or r.get("score")
            or r.get("vision_score")
            or r.get("original_combined_score")
        )
        # For UI thresholds; iNat scores are usually in [0, 1] but not always.
        conf = raw if raw <= 1.0 else min(1.0, raw / (raw + 1.0))

        inv_short, inv_detail = invasive_label(sci)
        rec = meta.get(tid, {})
        if rec:
            inv_short = rec.get("invasive_summary", inv_short)
            inv_detail = rec.get("invasive_detail", inv_detail)

        rows.append(
            {
                "taxon_id": tid,
                "confidence": conf,
                "raw_score": raw,
                "scientific_name": sci,
                "common_name": common,
                "study_region": os.environ.get("INAT_REGION_LABEL", "San Diego County, California"),
                "invasive_summary": inv_short,
                "invasive_detail": inv_detail,
                "source": "iNaturalist Computer Vision API",
            }
        )

    if not rows:
        ca = raw.get("common_ancestor")
        if ca and isinstance(ca, dict):
            t = ca.get("taxon") or {}
            sci = (t.get("name") or "").strip()
            inv_short, inv_detail = invasive_label(sci)
            rows.append(
                {
                    "taxon_id": str(t.get("id", "")),
                    "confidence": _score_to_display(ca.get("score")),
                    "raw_score": _score_to_display(ca.get("score")),
                    "scientific_name": sci,
                    "common_name": (t.get("preferred_common_name") or "").strip(),
                    "study_region": os.environ.get("INAT_REGION_LABEL", "San Diego County, California"),
                    "invasive_summary": inv_short,
                    "invasive_detail": inv_detail,
                    "source": "iNaturalist (common ancestor only)",
                }
            )

    return rows
