#!/usr/bin/env python3
"""Download images from the iNaturalist export and build train/val folders."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

import json

from config import (
    CSV_PATH,
    DATASET_DIR,
    MIN_SAMPLES_PER_CLASS,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_FRACTION,
)
from plant_metadata import build_record

TAXON_META_PATH = MODELS_DIR / "taxon_metadata.json"


def _safe_filename(url: str, obs_id: str) -> str:
    path = urlparse(url).path
    ext = Path(path).suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    return f"{obs_id}{ext}"


def download_one(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES_PER_CLASS,
        help="Minimum images per taxon to include in training.",
    )
    parser.add_argument(
        "--limit-downloads",
        type=int,
        default=0,
        help="If >0, cap total downloaded images (debug).",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("image_url") or "").strip()
            if not url:
                continue
            tid = str(row.get("taxon_id") or "").strip()
            if not tid:
                continue
            rows.append(row)

    by_taxon: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_taxon[str(row["taxon_id"])].append(row)

    selected = {t for t, rs in by_taxon.items() if len(rs) >= args.min_samples}
    print(
        f"Taxa with >= {args.min_samples} images: {len(selected)} "
        f"(from {len(by_taxon)} taxa total)"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Full metadata for all taxa in CSV (for the app to describe predictions).
    all_meta: dict[str, dict] = {}
    for row in rows:
        tid = str(row["taxon_id"])
        if tid in all_meta:
            continue
        all_meta[tid] = build_record(
            tid,
            row.get("scientific_name") or "",
            row.get("common_name") or "",
        )
    with open(TAXON_META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote {TAXON_META_PATH}")

    train_root = DATASET_DIR / "train"
    val_root = DATASET_DIR / "val"
    if train_root.exists():
        shutil.rmtree(train_root)
    if val_root.exists():
        shutil.rmtree(val_root)

    tasks: list[tuple[str, Path]] = []
    for tid in sorted(selected):
        class_rows = by_taxon[tid]
        paths = list(class_rows)
        random.shuffle(paths)
        n = len(paths)
        n_val = max(1, round(n * (1.0 - TRAIN_FRACTION)))
        if n <= 1:
            train_rows = paths
            val_rows = paths
        else:
            n_val = min(n_val, n - 1)
            val_rows = paths[-n_val:]
            train_rows = paths[:-n_val]

        for row in train_rows:
            dest_dir = train_root / tid
            tasks.append(
                (row["image_url"], dest_dir / _safe_filename(row["image_url"], str(row["id"])))
            )
        for row in val_rows:
            dest_dir = val_root / tid
            tasks.append(
                (row["image_url"], dest_dir / _safe_filename(row["image_url"], str(row["id"])))
            )

    if args.limit_downloads > 0:
        tasks = tasks[: args.limit_downloads]

    ok = 0
    for url, dest in tqdm(tasks, desc="Downloading"):
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        if download_one(url, dest):
            ok += 1

    print(f"Downloaded / verified {ok} of {len(tasks)} image files under {DATASET_DIR}")


if __name__ == "__main__":
    main()
