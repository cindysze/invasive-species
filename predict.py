"""Load the trained classifier and run inference on a PIL image or path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import CHECKPOINT_PATH, IMAGE_SIZE, LABEL_MAP_PATH, MODELS_DIR, REGION_LONG

TAXON_META_PATH = MODELS_DIR / "taxon_metadata.json"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_taxon_metadata() -> dict[str, Any]:
    if not TAXON_META_PATH.is_file():
        return {}
    with open(TAXON_META_PATH, encoding="utf-8") as f:
        return json.load(f)


_classifier_singleton: Optional["PlantClassifier"] = None


class PlantClassifier:
    def __init__(self, checkpoint_path: Path | None = None, label_map_path: Path | None = None):
        self.device = get_device()
        ckpt_path = checkpoint_path or CHECKPOINT_PATH
        map_path = label_map_path or LABEL_MAP_PATH

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"No model at {ckpt_path}. Train first: python prepare_data.py && python train.py"
            )

        blob = torch.load(ckpt_path, map_location=self.device)
        num_classes = int(blob["num_classes"])
        self.label_order: list[str] = blob["label_order"]

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)
        self.model.load_state_dict(blob["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose(
            [
                transforms.Resize(int(IMAGE_SIZE * 1.1)),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.taxon_meta = _load_taxon_metadata()

    @torch.no_grad()
    def predict_topk(self, image: Image.Image, k: int = 5) -> list[dict[str, Any]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.tf(image).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=0)
        topk = min(k, probs.numel())
        confs, idxs = torch.topk(probs, topk)

        out: list[dict[str, Any]] = []
        for conf, idx in zip(confs.tolist(), idxs.tolist()):
            tid = self.label_order[idx]
            meta = self.taxon_meta.get(tid, {})
            out.append(
                {
                    "taxon_id": tid,
                    "confidence": float(conf),
                    "scientific_name": meta.get("scientific_name", ""),
                    "common_name": meta.get("common_name", ""),
                    "study_region": meta.get("study_region", REGION_LONG),
                    "invasive_summary": meta.get("invasive_summary", ""),
                    "invasive_detail": meta.get("invasive_detail", ""),
                }
            )
        return out


def load_classifier() -> PlantClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = PlantClassifier()
    return _classifier_singleton
