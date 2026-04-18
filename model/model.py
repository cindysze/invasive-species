"""
Invasive Species Image Classifier
==================================
Classes: spotted knapweed, artichoke thistle, carnation spurge, volutaria

Usage:
    # Step 1 - Download images
    python invasive_species_classifier.py --download

    # Step 2 - Train the model
    python invasive_species_classifier.py --train

    # Step 3 - Predict on a new image
    python invasive_species_classifier.py --predict path/to/image.jpg

    # Run all steps
    python invasive_species_classifier.py --download --train --predict path/to/image.jpg
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — update CSV paths if needed
# ─────────────────────────────────────────────
DATASETS = {
    "spotted_knapweed":  "spotted_knapwood_data.csv",
    "carnation_spurge":  "carnation_spurge_data.csv",
    "volutaria":         "volutaria_data.csv",
    "artichoke_thistle": "artichoke_thistle_data.csv",
}

IMAGE_DIR   = "images"
MODEL_PATH  = "invasive_species_classifier.keras"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_P1   = 10   # frozen base
EPOCHS_P2   = 10   # fine-tuning


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
def download_all(max_per_class=None):
    print("\n=== Downloading images ===")
    if max_per_class:
        print(f"  Capped at {max_per_class} images per class")
    for label, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path).dropna(subset=["image_url"])
        if max_per_class:
            df = df.head(max_per_class)
        label_dir = Path(IMAGE_DIR) / label
        label_dir.mkdir(parents=True, exist_ok=True)

        downloaded = skipped = failed = 0
        for _, row in df.iterrows():
            img_path = label_dir / f"{row['id']}.jpg"
            if img_path.exists():
                skipped += 1
                continue
            try:
                r = requests.get(row["image_url"], timeout=10)
                if r.status_code == 200:
                    img_path.write_bytes(r.content)
                    downloaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        print(f"  {label}: {downloaded} downloaded, {skipped} skipped, {failed} failed")

    print("Download complete.\n")


# ─────────────────────────────────────────────
# STEP 2 — TRAIN
# ─────────────────────────────────────────────
def train():
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\n=== Training model ===")

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=30,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
    )

    train_gen = train_datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
    )

    num_classes = len(train_gen.class_indices)
    print(f"\nClasses detected: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.n} | Validation samples: {val_gen.n}")

    # Class weights to handle imbalance
    class_counts = np.bincount(train_gen.classes)
    class_weights = {
        i: train_gen.n / (num_classes * count)
        for i, count in enumerate(class_counts)
    }
    print(f"Class weights: { {k: round(v, 2) for k, v in class_weights.items()} }")

    # Build model
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Phase 1 — train classification head only
    print(f"\n--- Phase 1: Training classification head ({EPOCHS_P1} epochs) ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_P1,
        class_weight=class_weights,
    )

    # Phase 2 — unfreeze last 30 layers for fine-tuning
    print(f"\n--- Phase 2: Fine-tuning last 30 layers ({EPOCHS_P2} epochs) ---")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_P2,
        class_weight=class_weights,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Save class index mapping
    import json
    mapping_path = "class_indices.json"
    with open(mapping_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}\n")


# ─────────────────────────────────────────────
# STEP 3 — PREDICT
# ─────────────────────────────────────────────
def predict(img_path: str):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import json

    print(f"\n=== Predicting: {img_path} ===")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at '{MODEL_PATH}'. Run --train first.")
        sys.exit(1)

    # Load class mapping
    mapping_path = "class_indices.json"
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            class_indices = json.load(f)
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    else:
        class_names = sorted(DATASETS.keys())

    model = load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    predicted = class_names[np.argmax(probs)]

    print("\nConfidence scores:")
    for cls, prob in zip(class_names, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<22} {prob:5.1%}  {bar}")
    print(f"\n→ Predicted class: {predicted}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invasive Species Classifier")
    parser.add_argument("--download", action="store_true", help="Download images from CSVs")
    parser.add_argument("--max-per-class", type=int, metavar="N", help="Max images to download per class (default: all)")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--predict",  type=str, metavar="IMAGE", help="Predict on an image file")
    args = parser.parse_args()

    if not any([args.download, args.train, args.predict]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        download_all(max_per_class=args.max_per_class)

    if args.train:
        train()

    if args.predict:
        predict(args.predict)