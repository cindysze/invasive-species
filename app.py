#!/usr/bin/env python3
"""Gradio UI (optional): wraps the same identification service as the FastAPI backend."""

from __future__ import annotations

import gradio as gr

from backend.identification import identify_from_pil
from config import REGION_SHORT


def predict_image(image):
    outcome = identify_from_pil(image)
    if not outcome.ok:
        return outcome.error or "Something went wrong."
    return outcome.markdown


def main() -> None:
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Plant photo"),
        outputs=gr.Markdown(label="Predictions"),
        title=f"Invasive plant demo — {REGION_SHORT}",
        description=(
            "**Same logic as the REST API** (`uvicorn backend.main:app`). "
            "Default: iNaturalist CV if `INAT_JWT` is set; else local model. "
            "Optional: `OPENAI_API_KEY` for LLM summary."
        ),
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
