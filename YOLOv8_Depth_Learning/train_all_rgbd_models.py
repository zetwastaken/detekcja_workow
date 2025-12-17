"""
Batch train YOLOv8 segmentation models on all available RGBD (4-channel) datasets.
Uses the same hyperparameters as depth models, but disables HSV augmentations
to avoid corrupting the depth channel.
"""

from pathlib import Path
from datetime import datetime
from typing import List

from ultralytics import YOLO


# Configuration - use main detekcja_workow folders
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT.parent / "datasets"
RUNS_DIR = PROJECT_ROOT.parent / "runs"

# Training configuration - matches depth training but HSV is disabled
TRAINING_CONFIG = {
    "epochs": 5500,
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
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
}


def get_rgbd_datasets() -> list:
    """
    Find all RGBD datasets in the datasets directory.
    Returns list of (dataset_name, data_yaml_path) tuples.
    """
    datasets = []

    for dataset_dir in DATASETS_DIR.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_rgbd"):
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                if dataset_dir.name == "dataset_rgbd":
                    model_name = "default"
                else:
                    model_name = dataset_dir.name.replace("dataset_rgbd_", "")
                datasets.append((model_name, data_yaml))

    return sorted(datasets)


def train_single_dataset(
    model_name: str, data_yaml: Path, base_model: str = "yolov8n-seg.pt"
) -> dict:
    """
    Train YOLOv8 on a single RGBD dataset.

    Args:
        model_name: Name of the depth model (for naming the run)
        data_yaml: Path to the dataset's data.yaml
        base_model: Base YOLOv8 model to use

    Returns:
        Results dictionary from training
    """
    print(f"\n{'='*60}")
    print(f"Training on RGBD dataset: {model_name}")
    print(f"Data config: {data_yaml}")
    print(f"{'='*60}")

    model = YOLO(base_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rgbd_{model_name}_{timestamp}"

    results = model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR / "segment"),
        name=run_name,
        exist_ok=True,
        verbose=True,
        **TRAINING_CONFIG,
    )

    return {"model_name": model_name, "run_name": run_name, "results": results}


def main(datasets_to_train: List[str] = None, base_model: str = "yolov8n-seg.pt"):
    """
    Train YOLOv8 on all RGBD datasets.

    Args:
        datasets_to_train: List of specific model names to train on.
                          If None, train on all available RGBD datasets.
        base_model: Base YOLOv8 model to use
    """
    print("=" * 60)
    print("BATCH TRAINING - ALL RGBD DATASETS")
    print("=" * 60)

    all_datasets = get_rgbd_datasets()

    if not all_datasets:
        print("No RGBD datasets found!")
        print(
            f"Run generate_rgbd_datasets.py first to create datasets in {DATASETS_DIR}"
        )
        return

    print(f"\nFound {len(all_datasets)} RGBD datasets:")
    for name, path in all_datasets:
        print(f"  - {name}: {path}")

    if datasets_to_train:
        all_datasets = [(n, p) for n, p in all_datasets if n in datasets_to_train]
        print(f"\nFiltered to {len(all_datasets)} datasets")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch train YOLOv8 on all RGBD datasets"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Specific RGBD models to train on (default: all)",
        default=None,
    )
    parser.add_argument(
        "--base-model",
        "-b",
        default="yolov8n-seg.pt",
        help="Base YOLOv8 model to use (default: yolov8n-seg.pt)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available RGBD datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        datasets = get_rgbd_datasets()
        print("Available RGBD datasets:")
        for name, path in datasets:
            print(f"  - {name}")
        exit(0)

    main(args.models, args.base_model)
