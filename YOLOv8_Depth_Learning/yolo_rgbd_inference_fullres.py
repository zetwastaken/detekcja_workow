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

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_vision.factory import DepthEstimatorFactory
from depth_vision.utils import normalize_depth

# Paths
ROOT_DIR = PROJECT_ROOT
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs" / "segment"
OUTPUT_DIR = ROOT_DIR / "output" / "fullres_rgbd_predictions"

# Tiling parameters (match training)
TILE_SIZE = 640
OVERLAP = 80
STRIDE = TILE_SIZE - OVERLAP

# Configuration
DEPTH_MODEL = "depth_anything"
DEPTH_CONFIG = {"model_size": "large"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    rgbd = np.dstack([rgb_image, depth_normalized])

    return rgbd.astype(np.uint8)


def tile_image(
    image: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Tile an image into overlapping patches.

    Returns a list of (tile, y_start, x_start) tuples.
    """
    stride = tile_size - overlap
    height, width = image.shape[:2]
    tiles: List[Tuple[np.ndarray, int, int]] = []

    # Generate x coordinates
    x_starts = []
    x = 0
    while x <= width - tile_size:
        x_starts.append(x)
        x += stride
    if x_starts and x_starts[-1] < width - tile_size:
        x_starts.append(width - tile_size)
    elif not x_starts and width >= tile_size:
        x_starts.append(0)

    # Generate y coordinates
    y_starts = []
    y = 0
    while y <= height - tile_size:
        y_starts.append(y)
        y += stride
    if y_starts and y_starts[-1] < height - tile_size:
        y_starts.append(height - tile_size)
    elif not y_starts and height >= tile_size:
        y_starts.append(0)

    # Extract tiles with their positions
    for y_start in y_starts:
        for x_start in x_starts:
            tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tiles.append((tile, y_start, x_start))

    return tiles


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
            output[y_start:y_end, x_start:x_end] += tile_cropped.astype(np.float32) * weights_cropped
            weights[y_start:y_end, x_start:x_end] += weights_cropped
        else:
            for c in range(channels):
                output[y_start:y_end, x_start:x_end, c] += tile_cropped[:, :, c].astype(np.float32) * weights_cropped
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
        rgbd_candidates = sorted(
            RUNS_DIR.glob("rgbd_**/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if rgbd_candidates:
            return rgbd_candidates[0]
        
        if model_type == "rgbd":
            raise FileNotFoundError(
                f"No RGBD model weights found under {RUNS_DIR}. "
                "Please train an RGBD model first using train_rgbd_model.py"
            )
    
    # Fall back to RGB model or if explicitly requested
    rgb_candidates = sorted(
        RUNS_DIR.glob("yolov8_**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    if not rgb_candidates:
        raise FileNotFoundError(
            f"No model weights found under {RUNS_DIR}. "
            "Please train a model first."
        )
    
    return rgb_candidates[0]


def main():
    """Main inference pipeline."""
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
    weights_path = find_model_weights("auto")
    model_type = "RGBD" if "rgbd" in weights_path.parent.parent.name.lower() else "RGB"
    print(f"Loading {model_type} model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Initialize depth estimator
    print("Initializing Depth Anything Large estimator...")
    estimator = DepthEstimatorFactory.create(DEPTH_MODEL, **DEPTH_CONFIG)
    print("Estimator ready!")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_DIR / f"run_{timestamp}"
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
            
            # Generate RGBD image
            rgbd_image = create_rgbd_image(rgb_image, estimator)
            
            # Tile the RGBD image
            tiles = tile_image(rgbd_image, TILE_SIZE, OVERLAP)
            
            # Run predictions on tiles
            tile_predictions = []
            for tile_img, y_start, x_start in tiles:
                # For RGB models, use only first 3 channels
                if model_type == "RGB":
                    model_input = tile_img[:, :, :3]
                else:
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
                if results and hasattr(results[0], 'plot'):
                    pred_tile = results[0].plot()
                else:
                    # Fallback: use RGB channels if no predictions
                    pred_tile = tile_img[:, :, :3]
                
                tile_predictions.append((y_start, x_start, pred_tile))
            
            # Reassemble predictions into full image
            full_prediction = reassemble_predictions(
                tile_predictions,
                original_shape,
                TILE_SIZE
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
    del estimator
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
