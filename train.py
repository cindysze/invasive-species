#!/usr/bin/env python3
"""Fine-tune a ResNet backbone on the downloaded plant images."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATASET_DIR,
    EPOCHS,
    IMAGE_SIZE,
    LABEL_MAP_PATH,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders():
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(
            f"Missing {train_dir} or {val_dir}. Run prepare_data.py with network access first."
        )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    # Ensure class order matches between train and val
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise SystemExit("train/val class folders do not match — check dataset layout.")

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, train_ds = build_loaders()
    num_classes = len(train_ds.classes)
    print("Classes:", num_classes)

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for name, p in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            p.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    label_order = [idx_to_class[i] for i in range(num_classes)]

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS} train"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        model.eval()
        correct = 0
        total = 0
        vloss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                vloss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(1, total)
        print(
            f"epoch {epoch+1}: train_loss={running/len(train_loader.dataset):.4f} "
            f"val_loss={vloss/len(val_loader.dataset):.4f} val_acc={acc:.4f}"
        )
        if acc >= best_val:
            best_val = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "label_order": label_order,
                    "weights": "ResNet50_Weights.IMAGENET1K_V2",
                },
                CHECKPOINT_PATH,
            )

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"label_order": label_order, "class_to_idx": train_ds.class_to_idx}, f, indent=2)

    print(f"Saved checkpoint to {CHECKPOINT_PATH} (best val acc ~ {best_val:.4f})")
    print(f"Saved label map to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
