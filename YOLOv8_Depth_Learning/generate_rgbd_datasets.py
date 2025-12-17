"""
Generate RGBD (4-channel) datasets by combining RGB images with depth maps
for one or more depth estimation models. Images are tiled using the same
parameters as the depth datasets to keep splits aligned.
"""

import argparse
import shutil
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import create_dataset_structure, write_data_yaml
from utils.image_utils import create_depth_estimator, create_rgbd_image
from utils.tiling import tile_image_with_names


# Depth models to generate RGBD datasets for (mirrors generate_depth_datasets.py)
DEPTH_MODELS: Dict[str, Dict] = {
    "midas_DPT_Large": {"type": "midas", "model_type": "DPT_Large"},
    "depth_anything_large": {"type": "depth_anything", "model_size": "large"},
    "zoedepth_NK": {"type": "zoedepth", "model_type": "NK"},
    "zoedepth_N": {"type": "zoedepth", "model_type": "N"},
    "zoedepth_K": {"type": "zoedepth", "model_type": "K"},
    "marigold_lcm": {"type": "marigold", "variant": "lcm"},
    "marigold_default": {"type": "marigold", "variant": "default"},
}

# Tiling parameters (match training/depth generation)
TILE_SIZE = 640
OVERLAP = 80

# Paths - use main detekcja_workow folders
DATASETS_DIR = PROJECT_ROOT.parent / "datasets"
SOURCE_DATASET = DATASETS_DIR / "dataset_yolov8_V1"  # for split selection + labels
DATA_DIR = PROJECT_ROOT.parent / "data"
RGBD_TILES_DIR = PROJECT_ROOT.parent / "rgbd_tiles"


def get_selected_filenames() -> Tuple[set, set]:
    """
    Get filenames from train and valid splits of original dataset.
    Returns (train_files, valid_files) as sets of basenames (stems).
    """
    train_dir = SOURCE_DATASET / "train" / "images"
    valid_dir = SOURCE_DATASET / "valid" / "images"

    train_files = {f.stem for f in train_dir.glob("*.jpg")} | {
        f.stem for f in train_dir.glob("*.JPG")
    }
    valid_files = {f.stem for f in valid_dir.glob("*.jpg")} | {
        f.stem for f in valid_dir.glob("*.JPG")
    }

    return train_files, valid_files


def process_single_model(model_name: str, model_config: Dict) -> Path | None:
    """Generate an RGBD dataset for a single depth estimator."""
    print(f"\n{'='*60}")
    print(f"Processing RGBD model: {model_name}")
    print(f"{'='*60}")

    # Train/valid selection based on existing tiled dataset
    train_files, valid_files = get_selected_filenames()
    print(f"Split selection: {len(train_files)} train, {len(valid_files)} valid tiles")

    output_dataset = DATASETS_DIR / f"dataset_rgbd_{model_name}"
    create_dataset_structure(output_dataset)
    model_tiles_dir = RGBD_TILES_DIR / model_name
    model_tiles_dir.mkdir(parents=True, exist_ok=True)

    estimator_type = model_config.pop("type")
    print(f"Initializing {estimator_type} estimator...")
    try:
        estimator = create_depth_estimator(estimator_type, **model_config)
    except Exception as e:
        print(f"ERROR: Failed to create estimator: {e}")
        return None

    for split in ["train", "valid"]:
        print(f"\n{'='*40}")
        print(f"Processing {split} split")
        print(f"{'='*40}")

        dst_images_dir = output_dataset / split / "images"
        dst_labels_dir = output_dataset / split / "labels"

        # Use raw images (same source as depth pipeline)
        image_files = []
        for ext in (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.bmp",
            "*.tiff",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ):
            image_files.extend(DATA_DIR.glob(ext))
        print(f"Found {len(image_files)} raw images")

        saved_count = 0

        for img_path in tqdm(
            image_files, desc=f"Generating RGBD ({model_name}/{split})"
        ):
            rgb_image = cv2.imread(str(img_path))
            if rgb_image is None:
                print(f"Warning: Could not read {img_path.name}")
                continue

            try:
                rgbd_image = create_rgbd_image(rgb_image, estimator)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

            # Tile RGBD image and save only tiles belonging to this split
            tiles = tile_image_with_names(
                rgbd_image, img_path.stem, TILE_SIZE, OVERLAP, extension=".tiff"
            )

            for tile_name, tile_img in tiles.items():
                tile_stem = Path(tile_name).stem
                if (split == "train" and tile_stem not in train_files) or (
                    split == "valid" and tile_stem not in valid_files
                ):
                    continue

                output_path = dst_images_dir / tile_name
                cv2.imwrite(str(output_path), tile_img)
                saved_count += 1
                # Also save a copy into shared tiles folder
                cv2.imwrite(str(model_tiles_dir / tile_name), tile_img)

                # Copy matching label if it exists
                label_name = Path(tile_name).with_suffix(".txt").name
                src_label = SOURCE_DATASET / split / "labels" / label_name
                if src_label.exists():
                    shutil.copy2(src_label, dst_labels_dir / label_name)

        print(f"Saved {saved_count} RGBD tiles to {dst_images_dir}")

    write_data_yaml(
        output_dataset,
        class_names=["sandbag"],
        channels=4,
        header=f"YOLOv8 RGBD Dataset Configuration - {model_name}",
        generator="generate_rgbd_datasets.py",
    )

    print(f"✓ Dataset created at: {output_dataset}")

    # Cleanup GPU memory
    del estimator
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dataset


def generate_rgbd_datasets(
    models_to_process: List[str] | None = None,
) -> tuple[list, list]:
    """Generate RGBD datasets for the requested depth models (default: all)."""
    print("=" * 60)
    print("RGBD DATASET GENERATION (multi-model)")
    print("=" * 60)
    print(f"Source dataset: {SOURCE_DATASET}")

    if models_to_process is None:
        models_to_process = list(DEPTH_MODELS.keys())

    print(f"\nWill generate RGBD datasets for {len(models_to_process)} depth models:")
    for name in models_to_process:
        print(f"  - {name}")

    successful = []
    failed = []

    for model_name in models_to_process:
        if model_name not in DEPTH_MODELS:
            print(f"Unknown model: {model_name}")
            failed.append(model_name)
            continue

        model_config = DEPTH_MODELS[model_name].copy()

        try:
            dataset_path = process_single_model(model_name, model_config)
            if dataset_path:
                successful.append((model_name, dataset_path))
            else:
                failed.append(model_name)
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback

            traceback.print_exc()
            failed.append(model_name)

    print("\n" + "=" * 60)
    print("RGBD GENERATION COMPLETE")
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
    parser = argparse.ArgumentParser(
        description="Generate RGBD datasets for one or more depth estimators"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Specific depth models to process (default: all)",
        choices=list(DEPTH_MODELS.keys()),
        default=None,
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available depth models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available depth models:")
        for name, config in DEPTH_MODELS.items():
            print(f"  - {name}: {config}")
        sys.exit(0)

    generate_rgbd_datasets(args.models)
