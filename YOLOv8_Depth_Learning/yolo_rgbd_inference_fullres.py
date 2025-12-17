"""
Full-resolution RGBD inference pipeline.

This script performs end-to-end inference on full-resolution RGB images:
1. Reads RGB images from data/ directory
2. Generates RGBD (4-channel) images using depth estimation
3. Tiles RGBD images into 640x640 patches with overlap
4. Runs YOLO prediction on tiles
5. Reassembles prediction results back into full-resolution images
6. Saves results with original image names
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_utils import create_depth_estimator, create_rgbd_image
from utils.tiling import tile_image, tile_coordinates
from utils.depth_model_config import (
    extract_depth_model_from_weights,
    is_rgbd_model,
    get_depth_model_name_from_weights,
)

# Paths
ROOT_DIR = PROJECT_ROOT
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs" / "segment"
OUTPUT_DIR = ROOT_DIR / "output" / "fullres_rgbd_predictions"
RGBD_TILES_DIR = ROOT_DIR / "rgbd_tiles"

# Tiling parameters (match training)
TILE_SIZE = 640
OVERLAP = 80

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def reassemble_predictions(
    tiles_info: List[Tuple[int, int, np.ndarray]],
    original_shape: Tuple[int, int],
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """
    Reassemble prediction tiles back into a full-resolution image.

    Args:
        tiles_info: List of (y_start, x_start, predicted_tile) tuples
        original_shape: (height, width) of the original image
        tile_size: Size of each tile

    Returns:
        Reassembled full-resolution prediction image
    """
    height, width = original_shape

    # Determine number of channels from first tile
    if tiles_info:
        channels = tiles_info[0][2].shape[2] if len(tiles_info[0][2].shape) == 3 else 1
    else:
        channels = 3

    # Create output image and weight map for blending
    if channels == 1:
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
    else:
        output = np.zeros((height, width, channels), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)

    # Create blending weights (higher weight in center, lower at edges)
    tile_weights = np.ones((tile_size, tile_size), dtype=np.float32)

    # Apply distance-based weights to reduce edge artifacts
    # Add 1 to distance to avoid zero weights at edges
    for i in range(tile_size):
        for j in range(tile_size):
            dist_from_edge = min(i, j, tile_size - 1 - i, tile_size - 1 - j)
            normalized_dist = min((dist_from_edge + 1) / (OVERLAP / 2 + 1), 1.0)
            tile_weights[i, j] = max(normalized_dist, 0.1)  # Minimum weight of 0.1

    # Blend tiles
    for y_start, x_start, tile in tiles_info:
        y_end = min(y_start + tile_size, height)
        x_end = min(x_start + tile_size, width)

        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Handle edge tiles that may be smaller
        tile_cropped = tile[:tile_h, :tile_w]
        weights_cropped = tile_weights[:tile_h, :tile_w]

        if channels == 1:
            output[y_start:y_end, x_start:x_end] += (
                tile_cropped.astype(np.float32) * weights_cropped
            )
            weights[y_start:y_end, x_start:x_end] += weights_cropped
        else:
            for c in range(channels):
                output[y_start:y_end, x_start:x_end, c] += (
                    tile_cropped[:, :, c].astype(np.float32) * weights_cropped
                )
            weights[y_start:y_end, x_start:x_end] += weights_cropped

    # Normalize by weights
    if channels == 1:
        mask = weights > 0
        output[mask] /= weights[mask]
    else:
        for c in range(channels):
            mask = weights > 0
            output[:, :, c][mask] /= weights[mask]

    return output.astype(np.uint8)


def find_all_rgbd_model_weights() -> List[Path]:
    """
    Find all RGBD model weights in the runs directory.

    Returns:
        List of paths to all RGBD model weights, sorted by modification time (newest first)
    """
    rgbd_candidates = sorted(
        RUNS_DIR.glob("rgbd_**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return rgbd_candidates


def find_model_weights(model_type: str = "auto") -> Path:
    """
    Find model weights.

    Args:
        model_type: Type of model to find ("rgbd", "rgb", or "auto")
                   "auto" tries RGBD first, falls back to RGB

    Returns:
        Path to model weights
    """
    if model_type == "rgbd" or model_type == "auto":
        # Try to find RGBD model
        rgbd_candidates = find_all_rgbd_model_weights()
        if rgbd_candidates:
            return rgbd_candidates[0]

        if model_type == "rgbd":
            raise FileNotFoundError(
                f"No RGBD model weights found under {RUNS_DIR}. "
                "Please train an RGBD model first using train_all_rgbd_models.py"
            )

    # Fall back to RGB model or if explicitly requested
    rgb_candidates = sorted(
        RUNS_DIR.glob("yolov8_**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not rgb_candidates:
        raise FileNotFoundError(
            f"No model weights found under {RUNS_DIR}. " "Please train a model first."
        )

    return rgb_candidates[0]


def main(weights_path: Optional[Path] = None):
    """Main inference pipeline.

    Args:
        weights_path: Path to model weights. If None, auto-detect the most recent model.
    """
    print("=" * 80)
    print("Full-Resolution RGBD Inference Pipeline")
    print("=" * 80)

    # Check if data directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Collect input images
    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(DATA_DIR.glob(f"*{ext}"))
        image_files.extend(DATA_DIR.glob(f"*{ext.upper()}"))

    if not image_files:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {DATA_DIR}")

    # Load model (try RGBD first, fall back to RGB)
    if weights_path is None:
        weights_path = find_model_weights("auto")
    uses_rgbd = is_rgbd_model(weights_path)
    model_type = "RGBD" if uses_rgbd else "RGB"
    print(f"Loading {model_type} model from: {weights_path}")
    model = YOLO(weights_path)

    # Initialize depth estimator (auto-detect from trained model)
    estimator = None
    depth_model_name = None
    cached_tiles_dir = None

    if uses_rgbd:
        print("Detecting depth model configuration from trained weights...")
        depth_model_name = get_depth_model_name_from_weights(weights_path)
        depth_model_type, depth_config = extract_depth_model_from_weights(weights_path)

        # Check if cached RGBD tiles exist for this depth model
        if depth_model_name:
            potential_cache_dir = RGBD_TILES_DIR / depth_model_name
            if potential_cache_dir.exists() and any(potential_cache_dir.iterdir()):
                cached_tiles_dir = potential_cache_dir
                print(f"Found cached RGBD tiles at: {cached_tiles_dir}")
            else:
                print(f"No cached tiles found, will generate RGBD on-the-fly")
                print(
                    f"Initializing {depth_model_type} estimator with config: {depth_config}"
                )
                estimator = create_depth_estimator(depth_model_type, **depth_config)
                print("Depth estimator ready!")
        else:
            print(
                f"Initializing {depth_model_type} estimator with config: {depth_config}"
            )
            estimator = create_depth_estimator(depth_model_type, **depth_config)
            print("Depth estimator ready!")
    else:
        print("RGB model detected - skipping depth estimation")

    # Create output directory with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract model name from weights path (e.g., rgbd_depth_anything_large_20251211_153418)
    model_run_name = weights_path.parent.parent.name
    run_output_dir = OUTPUT_DIR / f"{model_run_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(image_files)} images...")
    print("=" * 80)

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read RGB image
            rgb_image = cv2.imread(str(img_path))
            if rgb_image is None:
                print(f"\n⚠️  Could not read {img_path.name}, skipping...")
                continue

            original_shape = rgb_image.shape[:2]
            image_stem = img_path.stem

            # Try to load cached RGBD tiles or generate on-the-fly
            tiles = []

            if cached_tiles_dir is not None:
                # Load tiles from cache
                y_starts, x_starts = tile_coordinates(
                    original_shape, TILE_SIZE, OVERLAP
                )
                all_tiles_found = True
                cached_tiles = []

                for row_idx, y_start in enumerate(y_starts):
                    for col_idx, x_start in enumerate(x_starts):
                        tile_filename = (
                            f"{image_stem}_R{row_idx:03d}_C{col_idx:03d}.tiff"
                        )
                        tile_path = cached_tiles_dir / tile_filename

                        if tile_path.exists():
                            # Load 4-channel RGBD tile (TIFF preserves all channels)
                            tile_img = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
                            if tile_img is not None:
                                cached_tiles.append((tile_img, y_start, x_start))
                            else:
                                all_tiles_found = False
                                break
                        else:
                            all_tiles_found = False
                            break
                    if not all_tiles_found:
                        break

                if all_tiles_found and cached_tiles:
                    tiles = cached_tiles
                else:
                    # Fallback: generate RGBD on-the-fly if some tiles missing
                    if estimator is None:
                        depth_model_type, depth_config = (
                            extract_depth_model_from_weights(weights_path)
                        )
                        print(
                            f"\n  Some cached tiles missing, initializing {depth_model_type} estimator..."
                        )
                        estimator = create_depth_estimator(
                            depth_model_type, **depth_config
                        )

                    input_image = create_rgbd_image(rgb_image, estimator)
                    tiles = tile_image(input_image, TILE_SIZE, OVERLAP)

            elif uses_rgbd and estimator is not None:
                # Generate RGBD on-the-fly
                input_image = create_rgbd_image(rgb_image, estimator)
                tiles = tile_image(input_image, TILE_SIZE, OVERLAP)
            else:
                # RGB only
                tiles = tile_image(rgb_image, TILE_SIZE, OVERLAP)

            # Run predictions on tiles
            tile_predictions = []
            for tile_img, y_start, x_start in tiles:
                # Use tile directly (channels already match model type)
                model_input = tile_img

                # Run YOLO prediction on this tile
                results = model.predict(
                    source=model_input,
                    save=False,
                    verbose=False,
                    imgsz=TILE_SIZE,
                    conf=0.25,
                    iou=0.7,
                )

                # Get the predicted image (with boxes/masks drawn)
                if results and hasattr(results[0], "plot"):
                    pred_tile = results[0].plot()
                else:
                    # Fallback: use RGB channels if no predictions
                    pred_tile = tile_img[:, :, :3]

                tile_predictions.append((y_start, x_start, pred_tile))

            # Reassemble predictions into full image
            full_prediction = reassemble_predictions(
                tile_predictions, original_shape, TILE_SIZE
            )

            # Save the result with original filename
            output_path = run_output_dir / img_path.name
            cv2.imwrite(str(output_path), full_prediction)

        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("✓ Inference complete!")
    print(f"✓ Results saved to: {run_output_dir}")
    print("=" * 80)

    # Cleanup GPU memory
    if estimator is not None:
        del estimator
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_output_dir


def run_all_models():
    """Run inference on all available RGBD models."""
    print("=" * 80)
    print("Running inference on ALL RGBD models")
    print("=" * 80)

    all_weights = find_all_rgbd_model_weights()

    if not all_weights:
        print("No RGBD models found!")
        return

    print(f"Found {len(all_weights)} RGBD models:")
    for i, w in enumerate(all_weights, 1):
        model_name = w.parent.parent.name
        print(f"  {i}. {model_name}")

    print()
    results = []

    for i, weights_path in enumerate(all_weights, 1):
        model_name = weights_path.parent.parent.name
        print(f"\n{'#'*80}")
        print(f"# Model {i}/{len(all_weights)}: {model_name}")
        print(f"{'#'*80}")

        try:
            output_dir = main(weights_path)
            results.append((model_name, "✓ Success", str(output_dir)))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((model_name, "❌ Failed", str(e)))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - All Models Inference")
    print("=" * 80)
    for model_name, status, info in results:
        print(f"{status} {model_name}")
        print(f"   -> {info}")
    print("=" * 80)


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - Set your options here
    # ========================================
    RUN_ALL_MODELS = True  # Set to True to run on all models, False for single model
    SPECIFIC_MODEL = None  # Set to model name or path, or None for auto-detect
    # ========================================

    if RUN_ALL_MODELS:
        run_all_models()
    elif SPECIFIC_MODEL:
        # Check if it's a full path or just a model name
        model_path = Path(SPECIFIC_MODEL)
        if not model_path.exists():
            # Try to find by name
            potential_path = RUNS_DIR / SPECIFIC_MODEL / "weights" / "best.pt"
            if potential_path.exists():
                model_path = potential_path
            else:
                print(f"Model not found: {SPECIFIC_MODEL}")
                sys.exit(1)
        main(model_path)
    else:
        main()
