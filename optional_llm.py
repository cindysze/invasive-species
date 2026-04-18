"""Optional OpenAI Chat Completions wrapper to turn structured IDs into a short summary paragraph."""

from __future__ import annotations

import json
import os
from typing import Any

import requests


def summarize_identification(rows: list[dict[str, Any]], region: str) -> str:
    """
    If OPENAI_API_KEY is set, returns a short natural-language summary of the top suggestions.
    Otherwise returns an empty string.
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or not rows:
        return ""

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    top = rows[:5]
    payload = [
        {
            "taxon_id": r.get("taxon_id"),
            "common_name": r.get("common_name"),
            "scientific_name": r.get("scientific_name"),
            "score": r.get("raw_score", r.get("confidence")),
            "invasive_note": r.get("invasive_summary"),
        }
        for r in top
    ]
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You help interpret plant identification API results for a public science demo. "
                    "Be concise, avoid claiming certainty, mention invasive risk only as local guidance."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Region focus: {region}.\n"
                    "Here are ranked species suggestions (from iNaturalist computer vision, not certainty):\n"
                    f"{json.dumps(payload, indent=2)}\n\n"
                    "Write 2–4 sentences summarizing likely candidates and reminding the user to verify ID."
                ),
            },
        ],
        "max_tokens": 220,
        "temperature": 0.4,
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return ""
