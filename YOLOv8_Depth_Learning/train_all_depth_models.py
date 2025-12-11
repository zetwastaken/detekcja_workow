"""
Batch training script for all depth model datasets.
Trains YOLOv8 segmentation model on each generated depth dataset
with identical hyperparameters for fair comparison.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"

# Training hyperparameters (same as original yolo_learning.py)
TRAINING_CONFIG = {
    "epochs": 500,
    "patience": 150,
    "imgsz": 640,
    "batch": 32,
    "lr0": 0.01,
    "lrf": 0.01,
    "task": "segment",
    "cache": False,
    "device": 0,  # GPU
    "cos_lr": True,
    "optimizer": "auto",
    "save": True,
    "save_period": 10,
    # Augmentation parameters
    "degrees": 15.0,
    "mosaic": 1.0,
    "fliplr": 0.5,
    "flipud": 0.2,
    "scale": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "copy_paste": 0.0,
    "close_mosaic": 10,
    "amp": True,
    "mixup": 0.1,
    "dropout": 0.0,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
}


def get_depth_datasets() -> list:
    """
    Find all depth datasets in the datasets directory.
    Returns list of (dataset_name, data_yaml_path) tuples.
    """
    datasets = []

    for dataset_dir in DATASETS_DIR.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_depth_"):
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                # Extract model name from dataset name
                model_name = dataset_dir.name.replace("dataset_depth_", "")
                datasets.append((model_name, data_yaml))

    return sorted(datasets)


def train_single_dataset(
    model_name: str, data_yaml: Path, base_model: str = "yolov8n-seg.pt"
) -> dict:
    """
    Train YOLOv8 on a single depth dataset.

    Args:
        model_name: Name of the depth model (for naming the run)
        data_yaml: Path to the dataset's data.yaml
        base_model: Base YOLOv8 model to use

    Returns:
        Results dictionary from training
    """
    print(f"\n{'='*60}")
    print(f"Training on dataset: {model_name}")
    print(f"Data config: {data_yaml}")
    print(f"{'='*60}")

    # Initialize model
    model = YOLO(base_model)

    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"depth_{model_name}_{timestamp}"

    # Train
    results = model.train(
        data=str(data_yaml), name=run_name, exist_ok=True, **TRAINING_CONFIG
    )

    return {"model_name": model_name, "run_name": run_name, "results": results}


def main(datasets_to_train: list = None, base_model: str = "yolov8n-seg.pt"):
    """
    Train YOLOv8 on all depth datasets.

    Args:
        datasets_to_train: List of specific model names to train on.
                          If None, train on all available depth datasets.
        base_model: Base YOLOv8 model to use
    """
    print("=" * 60)
    print("BATCH TRAINING - ALL DEPTH MODEL DATASETS")
    print("=" * 60)

    # Find all depth datasets
    all_datasets = get_depth_datasets()

    if not all_datasets:
        print("No depth datasets found!")
        print(
            f"Run generate_depth_datasets.py first to create datasets in {DATASETS_DIR}"
        )
        return

    print(f"\nFound {len(all_datasets)} depth datasets:")
    for name, path in all_datasets:
        print(f"  - {name}: {path}")

    # Filter if specific datasets requested
    if datasets_to_train:
        all_datasets = [(n, p) for n, p in all_datasets if n in datasets_to_train]
        print(f"\nFiltered to {len(all_datasets)} datasets")

    # Train each dataset
    results = []
    failed = []

    for model_name, data_yaml in all_datasets:
        try:
            result = train_single_dataset(model_name, data_yaml, base_model)
            results.append(result)
            print(f"✓ Completed training: {model_name}")
        except Exception as e:
            print(f"✗ Failed training {model_name}: {e}")
            import traceback

            traceback.print_exc()
            failed.append((model_name, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nSuccessfully trained {len(results)} models:")
    for r in results:
        print(f"  ✓ {r['model_name']} -> {r['run_name']}")

    if failed:
        print(f"\nFailed trainings ({len(failed)}):")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")

    print(f"\nResults saved in: {RUNS_DIR / 'segment'}")

    return results, failed


def train_also_rgb_baseline(base_model: str = "yolov8n-seg.pt"):
    """
    Train on original RGB dataset as baseline for comparison.
    """
    print("\n" + "=" * 60)
    print("TRAINING RGB BASELINE")
    print("=" * 60)

    rgb_dataset = DATASETS_DIR / "dataset_yolov8_V1" / "data.yaml"

    if not rgb_dataset.exists():
        print(f"RGB dataset not found at: {rgb_dataset}")
        return None

    result = train_single_dataset("rgb_baseline", rgb_dataset, base_model)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch train YOLOv8 on all depth datasets"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Specific depth models to train on (default: all)",
        default=None,
    )
    parser.add_argument(
        "--base-model",
        "-b",
        default="yolov8n-seg.pt",
        help="Base YOLOv8 model to use (default: yolov8n-seg.pt)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available datasets and exit"
    )
    parser.add_argument(
        "--include-rgb", action="store_true", help="Also train on RGB baseline dataset"
    )

    args = parser.parse_args()

    if args.list:
        datasets = get_depth_datasets()
        print("Available depth datasets:")
        for name, path in datasets:
            print(f"  - {name}")
        sys.exit(0)

    # Train on depth datasets
    main(args.models, args.base_model)

    # Optionally train RGB baseline
    if args.include_rgb:
        train_also_rgb_baseline(args.base_model)
