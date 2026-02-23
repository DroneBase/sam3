"""
Convert per-image Encord COCO annotation files into a single unified COCO JSON
compatible with SAM3 training (sam3.train.data.coco_json_loaders.COCO_FROM_JSON).

This script:
1. Reads all per-image *.coco.json annotation files
2. Matches each to its corresponding image by the `info.description` field
3. Only processes images that have a matching annotation file
4. Merges everything into a standard COCO JSON with global image/annotation IDs
5. Splits into train/test sets (80/20 by default)
6. Creates symlinks to the original images in train/ and test/ directories
7. Outputs _annotations.coco.json in each split directory

Usage:
    python scripts/prepare_solar_dataset.py \
        --img-dir /path/to/images_all \
        --ann-dir /path/to/aa9b3817-... \
        --output-dir /path/to/output/solar_array \
        --train-ratio 0.8 \
        --seed 42
"""

import argparse
import json
import os
import glob
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_per_image_annotations(ann_dir: str, img_dir: str):
    """
    Load all per-image COCO JSONs and match them to actual image files.

    Returns:
        list of dicts: Each dict has keys:
            - image_filename: str (e.g., "site_31684-pass_09-FLIR-125.png")
            - image_info: dict (width, height)
            - annotations: list of annotation dicts
            - categories: list of category dicts
    """
    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.coco.json")))
    available_images = set(os.listdir(img_dir))

    entries = []
    skipped = 0

    for ann_file in ann_files:
        with open(ann_file, "r") as f:
            data = json.load(f)

        # The actual image filename is stored in info.description
        image_filename = data["info"]["description"]

        # Skip if the image doesn't exist
        if image_filename not in available_images:
            skipped += 1
            print(f"  [SKIP] No image found for: {image_filename}")
            continue

        image_info = data["images"][0]  # Each file has exactly 1 image
        annotations = data["annotations"]
        categories = data["categories"]

        entries.append({
            "image_filename": image_filename,
            "image_info": image_info,
            "annotations": annotations,
            "categories": categories,
        })

    print(f"Loaded {len(entries)} annotated images ({skipped} skipped - no matching image)")
    return entries


def build_unified_coco(entries, split_name="train"):
    """
    Build a single unified COCO-format JSON from the per-image entries.

    The output format matches what SAM3's COCO_FROM_JSON expects:
    {
        "images": [...],       # Each with id, file_name, height, width
        "annotations": [...],  # Each with id, image_id, category_id, bbox (xywh), segmentation, area, iscrowd
        "categories": [...]    # Unified category list
    }
    """
    # Collect all unique categories
    cat_map = {}
    for entry in entries:
        for cat in entry["categories"]:
            cat_map[cat["id"]] = cat

    categories = sorted(cat_map.values(), key=lambda c: c["id"])

    images = []
    annotations = []
    global_ann_id = 0

    for img_id, entry in enumerate(entries):
        # Image entry: file_name is just the filename (relative to img_folder)
        img_entry = {
            "id": img_id,
            "file_name": entry["image_filename"],
            "height": entry["image_info"]["height"],
            "width": entry["image_info"]["width"],
        }
        images.append(img_entry)

        # Annotations for this image
        for ann in entry["annotations"]:
            new_ann = {
                "id": global_ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],  # Already in COCO xywh format
                "area": ann["area"],
                "segmentation": ann["segmentation"],  # Polygon format
                "iscrowd": ann.get("iscrowd", 0),
            }
            annotations.append(new_ann)
            global_ann_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    print(f"  [{split_name}] {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    return coco_json


def create_image_links(entries, img_dir, output_dir):
    """
    Create symlinks (or copies) of images into the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for entry in entries:
        src = os.path.join(img_dir, entry["image_filename"])
        dst = os.path.join(output_dir, entry["image_filename"])

        if not os.path.exists(dst):
            # Use symlink for efficiency; fall back to copy if symlinks not supported
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-image Encord COCO annotations to unified SAM3-compatible COCO JSON"
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="Path to directory containing all images",
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        required=True,
        help="Path to directory containing per-image .coco.json annotation files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the SAM3-compatible dataset (will contain train/ and test/ subdirs)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Dataset Preparation - Solar Array Segmentation")
    print("=" * 60)

    # Step 1: Load and match annotations
    print("\n[1/4] Loading per-image annotations...")
    entries = load_per_image_annotations(args.ann_dir, args.img_dir)

    if len(entries) == 0:
        print("ERROR: No matching image-annotation pairs found!")
        return

    # Step 2: Train/test split
    print(f"\n[2/4] Splitting dataset (train={args.train_ratio:.0%}, test={1 - args.train_ratio:.0%})...")
    random.seed(args.seed)
    shuffled = list(entries)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * args.train_ratio)
    train_entries = shuffled[:split_idx]
    test_entries = shuffled[split_idx:]

    print(f"  Train: {len(train_entries)} images")
    print(f"  Test:  {len(test_entries)} images")

    # Step 3: Build unified COCO JSONs
    print("\n[3/4] Building unified COCO annotation files...")
    train_coco = build_unified_coco(train_entries, "train")
    test_coco = build_unified_coco(test_entries, "test")

    # Step 4: Write outputs
    print("\n[4/4] Writing output files...")
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Write annotation JSONs
    train_ann_path = os.path.join(train_dir, "_annotations.coco.json")
    test_ann_path = os.path.join(test_dir, "_annotations.coco.json")

    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f, indent=2)
    print(f"  Wrote: {train_ann_path}")

    with open(test_ann_path, "w") as f:
        json.dump(test_coco, f, indent=2)
    print(f"  Wrote: {test_ann_path}")

    # Create image symlinks
    create_image_links(train_entries, args.img_dir, train_dir)
    print(f"  Linked {len(train_entries)} images to {train_dir}")

    create_image_links(test_entries, args.img_dir, test_dir)
    print(f"  Linked {len(test_entries)} images to {test_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("DONE! Dataset structure created:")
    print(f"  {args.output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── _annotations.coco.json  ({len(train_entries)} images, {len(train_coco['annotations'])} annotations)")
    print(f"  │   └── <image symlinks>")
    print(f"  └── test/")
    print(f"      ├── _annotations.coco.json  ({len(test_entries)} images, {len(test_coco['annotations'])} annotations)")
    print(f"      └── <image symlinks>")
    print(f"\nCategories: {[c['name'] for c in train_coco['categories']]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
