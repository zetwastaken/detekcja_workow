"""
Automated pipeline for generating depth map datasets for all depth estimation models.
This script:
1. Generates depth maps (with colormap) for all images in data/ folder
2. Tiles them using the same parameters as original dataset
3. Filters tiles matching choosen_V1 selection
4. Creates separate YOLOv8 datasets for each depth model
"""

import cv2
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from depth_vision.utils import visualize_depth
from utils.dataset_utils import copy_labels, create_dataset_structure, write_data_yaml
from utils.image_utils import create_depth_estimator
from utils.tiling import tile_image_with_names


# All depth model configurations to test
DEPTH_MODELS = {
    # MiDaS variants
    "midas_DPT_Large": {"type": "midas", "model_type": "DPT_Large"},
    # Depth Anything V2 variants
    "depth_anything_large": {"type": "depth_anything", "model_size": "large"},
    # ZoeDepth variants
    "zoedepth_NK": {"type": "zoedepth", "model_type": "NK"},
    "zoedepth_N": {"type": "zoedepth", "model_type": "N"},
    "zoedepth_K": {"type": "zoedepth", "model_type": "K"},
    # Marigold variants
    "marigold_lcm": {"type": "marigold", "variant": "lcm"},
    "marigold_default": {"type": "marigold", "variant": "default"},
}

# Tiling parameters (same as original dataset)
TILE_SIZE = 640
OVERLAP = 80

# Paths - use main detekcja_workow folders
DATA_DIR = PROJECT_ROOT.parent / "data"
DATASETS_DIR = PROJECT_ROOT.parent / "datasets"
TILING_DIR = PROJECT_ROOT.parent / "tiling"
CHOOSEN_V1_DIR = TILING_DIR / "choosen_V1"
SOURCE_DATASET = DATASETS_DIR / "dataset_yolov8_V1"

# Output directories
DEPTH_MAPS_DIR = PROJECT_ROOT.parent / "depth_maps"
DEPTH_TILES_DIR = PROJECT_ROOT.parent / "depth_tiles"


def get_selected_filenames() -> Tuple[set, set]:
    """
    Get filenames from train and valid splits of original dataset.
    Returns (train_files, valid_files) as sets of filenames.
    """
    train_dir = SOURCE_DATASET / "train" / "images"
    valid_dir = SOURCE_DATASET / "valid" / "images"

    train_files = {f.name for f in train_dir.glob("*.jpg")}
    valid_files = {f.name for f in valid_dir.glob("*.jpg")}

    return train_files, valid_files


def process_single_model(
    model_name: str,
    model_config: dict,
    train_files: set,
    valid_files: set,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> Path:
    """
    Process a single depth model: generate depth maps, tile, and create dataset.
    Returns path to created dataset.
    """
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    # Create output directories for this model
    model_depth_dir = DEPTH_MAPS_DIR / model_name
    model_tiles_dir = DEPTH_TILES_DIR / model_name
    dataset_dir = DATASETS_DIR / f"dataset_depth_{model_name}"

    model_depth_dir.mkdir(parents=True, exist_ok=True)
    model_tiles_dir.mkdir(parents=True, exist_ok=True)
    create_dataset_structure(dataset_dir)

    # Initialize depth estimator
    estimator_type = model_config.pop("type")
    print(f"Initializing {estimator_type} estimator...")
    try:
        estimator = create_depth_estimator(estimator_type, **model_config)
    except Exception as e:
        print(f"ERROR: Failed to create estimator: {e}")
        return None

    # Get list of source images
    img_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG")
    source_images = [f for f in DATA_DIR.iterdir() if f.suffix in img_formats]

    print(f"Found {len(source_images)} source images")

    # All tiles that will be generated
    all_tiles = {}

    # Process each source image
    for img_path in tqdm(source_images, desc="Generating depth maps"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path.name}")
            continue

        # Generate depth map
        try:
            depth_map = estimator.estimate(image)
        except Exception as e:
            print(f"Error generating depth for {img_path.name}: {e}")
            continue

        # Apply colormap
        depth_colored = visualize_depth(depth_map, colormap=colormap)

        # Save full depth map
        depth_filename = f"{img_path.stem}_depth.png"
        cv2.imwrite(str(model_depth_dir / depth_filename), depth_colored)

        # Tile the depth map
        base_name = img_path.stem
        tiles = tile_image_with_names(depth_colored, base_name, TILE_SIZE, OVERLAP)
        all_tiles.update(tiles)

    print(f"Generated {len(all_tiles)} tiles total")

    # Filter tiles and copy to dataset
    train_count = 0
    valid_count = 0

    for tile_name, tile_img in all_tiles.items():
        if tile_name in train_files:
            # Save to train
            cv2.imwrite(str(dataset_dir / "train" / "images" / tile_name), tile_img)
            cv2.imwrite(str(model_tiles_dir / tile_name), tile_img)
            train_count += 1
        elif tile_name in valid_files:
            # Save to valid
            cv2.imwrite(str(dataset_dir / "valid" / "images" / tile_name), tile_img)
            cv2.imwrite(str(model_tiles_dir / tile_name), tile_img)
            valid_count += 1

    print(f"Copied {train_count} tiles to train, {valid_count} tiles to valid")

    # Copy labels from source dataset
    copy_labels(SOURCE_DATASET, dataset_dir)

    # Create data.yaml
    write_data_yaml(
        dataset_dir,
        class_names=["sandbag"],
        header=f"YOLOv8 Depth Dataset Configuration - {model_name}",
        generator="generate_depth_datasets.py",
    )

    print(f"Dataset created at: {dataset_dir}")

    # Cleanup: unload model to free GPU memory
    del estimator
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dataset_dir


def main(models_to_process: List[str] = None):
    """
    Main function to generate all depth datasets.

    Args:
        models_to_process: List of model names to process. If None, process all.
    """
    print("=" * 60)
    print("DEPTH MAP DATASET GENERATION PIPELINE")
    print("=" * 60)

    # Get train/valid splits from source dataset
    train_files, valid_files = get_selected_filenames()
    print(f"Source dataset: {len(train_files)} train, {len(valid_files)} valid images")

    # Create base output directories
    DEPTH_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_TILES_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which models to process
    if models_to_process is None:
        models_to_process = list(DEPTH_MODELS.keys())

    print(f"\nWill process {len(models_to_process)} depth models:")
    for name in models_to_process:
        print(f"  - {name}")

    # Process each model
    successful = []
    failed = []

    for model_name in models_to_process:
        if model_name not in DEPTH_MODELS:
            print(f"Unknown model: {model_name}")
            failed.append(model_name)
            continue

        # Make a copy of config to avoid modifying original
        model_config = DEPTH_MODELS[model_name].copy()

        try:
            dataset_path = process_single_model(
                model_name, model_config, train_files, valid_files
            )
            if dataset_path:
                successful.append((model_name, dataset_path))
            else:
                failed.append(model_name)
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback

            traceback.print_exc()
            failed.append(model_name)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nSuccessfully created {len(successful)} datasets:")
    for name, path in successful:
        print(f"  ✓ {name}: {path}")

    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")

    return successful, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate depth map datasets for all depth models"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Specific models to process (default: all)",
        choices=list(DEPTH_MODELS.keys()),
        default=None,
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available depth models:")
        for name, config in DEPTH_MODELS.items():
            print(f"  - {name}: {config}")
        sys.exit(0)

    main(args.models)
