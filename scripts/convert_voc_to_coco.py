# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Script to convert Pascal VOC annotations to COCO JSON format for SAM3 training

"""
Pascal VOC to COCO JSON Converter for SAM3 Training

This script converts Pascal VOC XML annotations to COCO JSON format
required by SAM3 training pipeline.

Usage:
python scripts/convert_voc_to_coco.py \
    --voc_dir /Users/sayandebroy/Developer/zeitview/datasets/solar_anomaly_detection/normalized_images/hotspot_small/train \
    --output_json /Users/sayandebroy/Developer/zeitview/datasets/solar_anomaly_detection/normalized_images/hotspot_small/train/_annotations_coco.json

python convert_voc_to_coco.py --voc_dir /Users/sayandebroy/Developer/zeitview/datasets/solar_anomaly_detection/normalized_images/hotspot_small/test --output_json /Users/sayandebroy/Developer/zeitview/datasets/solar_anomaly_detection/normalized_images/hotspot_small/test/_annotations_coco.json
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm


def parse_voc_annotation(xml_path: str) -> Dict:
    """
    Parse a Pascal VOC XML annotation file.
    
    Args:
        xml_path: Path to the XML annotation file
        
    Returns:
        Dictionary containing parsed annotation data
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = obj.find('difficult')
        difficult = int(difficult.text) if difficult is not None else 0
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # COCO format uses [x, y, width, height]
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        
        objects.append({
            'name': name,
            'bbox': bbox,
            'area': area,
            'difficult': difficult
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions from file."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_voc_to_coco(
    voc_dir: str,
    output_json: str,
    images_subdir: str = "images",
    annotations_subdir: str = "annotations"
) -> None:
    """
    Convert Pascal VOC annotations to COCO JSON format.
    
    Args:
        voc_dir: Root directory containing images and annotations subdirectories
        output_json: Output path for COCO JSON file
        images_subdir: Name of images subdirectory
        annotations_subdir: Name of annotations subdirectory
    """
    voc_dir = Path(voc_dir)
    images_dir = voc_dir / images_subdir
    annotations_dir = voc_dir / annotations_subdir
    
    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track categories
    category_name_to_id = {}
    
    # Get all annotation files
    annotation_files = sorted(annotations_dir.glob("*.xml"))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    image_id = 0
    annotation_id = 0
    
    for xml_path in tqdm(annotation_files, desc="Converting annotations"):
        try:
            # Parse VOC annotation
            voc_data = parse_voc_annotation(str(xml_path))
            
            # Determine image path
            image_filename = voc_data['filename']
            image_path = images_dir / image_filename
            
            # Handle case where filename doesn't include extension
            if not image_path.exists():
                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tiff', '.TIFF', '.tif', '.TIF']:
                    potential_path = images_dir / (Path(image_filename).stem + ext)
                    if potential_path.exists():
                        image_path = potential_path
                        image_filename = potential_path.name
                        break
            
            if not image_path.exists():
                print(f"Warning: Image not found for {xml_path.name}, skipping...")
                continue
            
            # Get actual image dimensions (in case XML is incorrect)
            try:
                actual_width, actual_height = get_image_size(str(image_path))
            except Exception as e:
                print(f"Warning: Could not read image {image_path}: {e}")
                actual_width, actual_height = voc_data['width'], voc_data['height']
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": actual_width,
                "height": actual_height
            })
            
            # Add annotations for this image
            for obj in voc_data['objects']:
                # Get or create category ID
                category_name = obj['name']
                if category_name not in category_name_to_id:
                    cat_id = len(category_name_to_id) + 1  # COCO uses 1-indexed categories
                    category_name_to_id[category_name] = cat_id
                    coco_data["categories"].append({
                        "id": cat_id,
                        "name": category_name,
                        "supercategory": category_name
                    })
                
                cat_id = category_name_to_id[category_name]
                
                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": obj['bbox'],  # [x, y, width, height]
                    "area": obj['area'],
                    "iscrowd": 0
                })
                annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            continue
    
    # Sort categories by ID
    coco_data["categories"] = sorted(coco_data["categories"], key=lambda x: x["id"])
    
    # Save COCO JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    for cat in coco_data["categories"]:
        count = sum(1 for ann in coco_data["annotations"] if ann["category_id"] == cat["id"])
        print(f"  - {cat['name']}: {count} annotations")
    print(f"\nOutput saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotations to COCO JSON format for SAM3 training"
    )
    parser.add_argument(
        "--voc_dir",
        type=str,
        required=True,
        help="Root directory containing 'images' and 'annotations' subdirectories"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output path for COCO JSON file"
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="images",
        help="Name of images subdirectory (default: 'images')"
    )
    parser.add_argument(
        "--annotations_subdir",
        type=str,
        default="annotations",
        help="Name of annotations subdirectory (default: 'annotations')"
    )
    
    args = parser.parse_args()
    
    convert_voc_to_coco(
        voc_dir=args.voc_dir,
        output_json=args.output_json,
        images_subdir=args.images_subdir,
        annotations_subdir=args.annotations_subdir
    )


if __name__ == "__main__":
    main()
