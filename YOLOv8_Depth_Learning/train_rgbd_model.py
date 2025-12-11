"""
Train YOLOv8 segmentation model on RGBD (4-channel) dataset.
Uses same hyperparameters as other depth models for fair comparison,
but disables HSV augmentations since they would corrupt the depth channel.
"""

import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs" / "segment"

# Dataset path
RGBD_DATASET = DATASETS_DIR / "dataset_rgbd" / "data.yaml"

# Training configuration - same as train_all_depth_models.py
TRAINING_CONFIG = {
    "epochs": 1500,
    "patience": 150,
    "imgsz": 640,
    "batch": 32,
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": True,
    "optimizer": "auto",
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "dropout": 0.0,
    # Geometric augmentations (work for all channels)
    "degrees": 15.0,
    "mosaic": 1.0,
    "fliplr": 0.5,
    "flipud": 0.2,
    "scale": 0.5,
    "mixup": 0.1,
    "close_mosaic": 10,
    # DISABLED: HSV augmentations would corrupt depth channel
    "hsv_h": 0.0,  # Was 0.015
    "hsv_s": 0.0,  # Was 0.7
    "hsv_v": 0.0,  # Was 0.4
}


def train_rgbd_model():
    """Train YOLOv8 segmentation on RGBD dataset."""

    # Check dataset exists
    if not RGBD_DATASET.exists():
        print(f"ERROR: Dataset not found at {RGBD_DATASET}")
        print("Please run generate_rgbd_dataset.py first!")
        return None

    # Generate unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rgbd_depth_anything_large_{timestamp}"

    print("=" * 60)
    print("RGBD Model Training")
    print("=" * 60)
    print(f"Dataset: {RGBD_DATASET}")
    print(f"Run name: {run_name}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Batch size: {TRAINING_CONFIG['batch']}")
    print(f"Image size: {TRAINING_CONFIG['imgsz']}")
    print(f"HSV augmentations: DISABLED (to preserve depth channel)")
    print("=" * 60)

    # Load model
    model = YOLO("yolov8n-seg.pt")

    # Train
    results = model.train(
        data=str(RGBD_DATASET),
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        verbose=True,
        **TRAINING_CONFIG,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {RUNS_DIR / run_name}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_rgbd_model()
