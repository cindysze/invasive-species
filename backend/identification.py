"""Shared identification logic: iNaturalist CV and/or local model + invasive metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from . import deps  # noqa: F401 — adds project root to sys.path

from config import REGION_LONG, REGION_SHORT
from predict import load_classifier


def resolve_backend() -> str:
    b = (os.environ.get("BACKEND") or "auto").strip().lower()
    if b == "auto":
        if (os.environ.get("INAT_JWT") or os.environ.get("INATURALIST_API_TOKEN") or "").strip():
            return "inaturalist"
        return "local"
    if b in ("inat", "inaturalist", "inat_cv"):
        return "inaturalist"
    return "local"


def format_predictions_markdown(rows: list[dict], *, backend: str = "local") -> str:
    lines: list[str] = []
    if rows:
        reg = rows[0].get("study_region") or ""
        if reg:
            lines.append(f"**Scope:** {reg}")
            lines.append("")
        src = rows[0].get("source")
        if src and backend == "inaturalist":
            lines.append(f"**ID source:** {src}")
            lines.append("")
    for i, r in enumerate(rows, 1):
        name = r.get("common_name") or r.get("scientific_name") or "Unknown"
        sci = r.get("scientific_name", "")
        if backend == "inaturalist":
            raw = r.get("raw_score", r.get("confidence", 0))
            lines.append(f"**{i}. {name}** ({sci}) — iNaturalist score **{float(raw):.4f}**")
        else:
            pct = 100.0 * float(r["confidence"])
            lines.append(f"**{i}. {name}** ({sci}) — {pct:.1f}% local model confidence")
        lines.append(f"- Invasive note: {r.get('invasive_summary', '')}")
        lines.append(f"- {r.get('invasive_detail', '')}")
        lines.append("")
    return "\n".join(lines).strip()


@dataclass
class IdentifyOutcome:
    ok: bool
    backend: str = "local"
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    llm_summary: str | None = None
    markdown: str = ""
    error: str | None = None


def identify_from_pil(image) -> IdentifyOutcome:
    """Run full pipeline from a PIL image. Used by FastAPI and Gradio."""
    from PIL import Image

    if image is None:
        return IdentifyOutcome(ok=False, error="No image provided.")
    if not isinstance(image, Image.Image):
        return IdentifyOutcome(ok=False, error="Expected a PIL Image.")

    backend_name = resolve_backend()

    if backend_name == "inaturalist":
        try:
            from inaturalist_cv import suggestions_for_app
            from optional_llm import summarize_identification

            top = suggestions_for_app(image, top_k=8)
            summary = ""
            try:
                summary = summarize_identification(top, REGION_LONG)
            except Exception:
                pass
        except RuntimeError as e:
            return IdentifyOutcome(
                ok=False,
                backend="inaturalist",
                error=(
                    f"{e}\n\nAlternatively use BACKEND=local after training "
                    "(prepare_data.py + train.py)."
                ),
            )
        except Exception as e:
            return IdentifyOutcome(ok=False, backend="inaturalist", error=f"iNaturalist API error: {e}")

        text = format_predictions_markdown(top, backend="inaturalist")
        if summary:
            text += "\n\n---\n### Optional LLM summary (OpenAI)\n\n" + summary
        best = top[0] if top else {}
        raw = float(best.get("raw_score", best.get("confidence", 0)))
        caveat = (
            f"\n\n---\n*Suggestions from **iNaturalist Computer Vision**. "
            f"Geo hint: near {REGION_SHORT} (override with INAT_LAT / INAT_LNG). "
            "Scores are API outputs, not probabilities.*"
        )
        if raw < 0.08 and top:
            caveat = (
                "\n\n---\n**Low top score.** Try a sharper photo or adjust location hints." + caveat
            )
        text += caveat
        return IdentifyOutcome(
            ok=True,
            backend="inaturalist",
            suggestions=top,
            llm_summary=summary or None,
            markdown=text,
        )

    try:
        clf = load_classifier()
    except FileNotFoundError as e:
        return IdentifyOutcome(ok=False, backend="local", error=f"Model not found: {e}")

    top = clf.predict_topk(image, k=5)
    best = top[0]
    caveat = (
        f"\n\n---\n*Local model trained on iNaturalist export for {REGION_SHORT}; "
        "only knows species with enough training images.*"
    )
    if float(best["confidence"]) < 0.35:
        caveat = (
            "\n\n---\n**Low confidence.** Try a clearer photo of leaves/flowers." + caveat
        )
    text = format_predictions_markdown(top, backend="local") + caveat
    return IdentifyOutcome(ok=True, backend="local", suggestions=top, markdown=text)


def outcome_to_api_dict(o: IdentifyOutcome) -> dict[str, Any]:
    """JSON-serializable payload for the frontend."""
    if not o.ok:
        return {"ok": False, "error": o.error or "Unknown error"}
    return {
        "ok": True,
        "backend": o.backend,
        "suggestions": o.suggestions,
        "llm_summary": o.llm_summary,
        "markdown": o.markdown,
    }
