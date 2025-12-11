"""
Generate RGBD (4-channel) dataset by combining RGB tiles with depth maps.
Uses Depth Anything Large (best performing depth model) to generate the depth channel.
Saves as 4-channel TIFF images for native Ultralytics multi-channel support.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from depth_vision.factory import DepthEstimatorFactory
from depth_vision.utils import normalize_depth


# Configuration
DEPTH_MODEL = "depth_anything"
DEPTH_CONFIG = {"model_size": "large"}

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
SOURCE_DATASET = DATASETS_DIR / "dataset_yolov8_V1"
OUTPUT_DATASET = DATASETS_DIR / "dataset_rgbd"


def create_rgbd_image(rgb_image: np.ndarray, depth_estimator) -> np.ndarray:
    """
    Create 4-channel RGBD image from RGB image.

    Args:
        rgb_image: BGR image (OpenCV format)
        depth_estimator: Initialized depth estimator

    Returns:
        4-channel BGRD image (uint8)
    """
    # Generate depth map
    depth_map = depth_estimator.estimate(rgb_image)

    # Normalize depth to 0-255
    depth_normalized = normalize_depth(depth_map)

    # Stack RGB + Depth as 4th channel (BGRD)
    # OpenCV reads as BGR, so result is BGRD
    rgbd = np.dstack([rgb_image, depth_normalized])

    return rgbd


def generate_rgbd_dataset():
    """Generate the full RGBD dataset."""
    print("=" * 60)
    print("RGBD Dataset Generation")
    print("=" * 60)
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Depth model: {DEPTH_MODEL} ({DEPTH_CONFIG})")
    print("=" * 60)

    # Create output directory structure
    for split in ["train", "valid"]:
        (OUTPUT_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)

    # Initialize depth estimator
    print("\nInitializing Depth Anything Large estimator...")
    estimator = DepthEstimatorFactory.create(DEPTH_MODEL, **DEPTH_CONFIG)
    print("Estimator ready!")

    # Process each split
    for split in ["train", "valid"]:
        print(f"\n{'='*40}")
        print(f"Processing {split} split")
        print(f"{'='*40}")

        src_images_dir = SOURCE_DATASET / split / "images"
        src_labels_dir = SOURCE_DATASET / split / "labels"
        dst_images_dir = OUTPUT_DATASET / split / "images"
        dst_labels_dir = OUTPUT_DATASET / split / "labels"

        # Get all source images
        image_files = list(src_images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} images")

        # Process each image
        for img_path in tqdm(image_files, desc=f"Generating RGBD ({split})"):
            # Read RGB image
            rgb_image = cv2.imread(str(img_path))
            if rgb_image is None:
                print(f"Warning: Could not read {img_path.name}")
                continue

            # Create RGBD image
            try:
                rgbd_image = create_rgbd_image(rgb_image, estimator)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

            # Save as TIFF (supports 4 channels natively)
            output_filename = img_path.stem + ".tiff"
            output_path = dst_images_dir / output_filename
            cv2.imwrite(str(output_path), rgbd_image)

            # Copy corresponding label file
            label_filename = img_path.stem + ".txt"
            src_label = src_labels_dir / label_filename
            if src_label.exists():
                shutil.copy2(src_label, dst_labels_dir / label_filename)

        print(f"Saved {len(image_files)} RGBD images to {dst_images_dir}")

    # Create data.yaml with channels: 4
    data_yaml_content = f"""# YOLOv8 RGBD Dataset Configuration
# 4-channel input: RGB + Depth (from Depth Anything Large)
# Generated automatically by generate_rgbd_dataset.py

# Dataset path (absolute)
path: {OUTPUT_DATASET}

# Train and validation image paths (relative to 'path')
train: train/images
val: valid/images

# Number of classes
nc: 1

# Class names
names:
  0: sandbag

# IMPORTANT: 4-channel input (RGBD)
channels: 4
"""

    with open(OUTPUT_DATASET / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Output: {OUTPUT_DATASET}")
    print(f"data.yaml created with channels: 4")
    print(f"{'='*60}")

    # Cleanup GPU memory
    del estimator
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return OUTPUT_DATASET


if __name__ == "__main__":
    generate_rgbd_dataset()
