"""
RGBD inference pipeline with tiled detection.

This script performs end-to-end inference on full-resolution RGB images:
1. Reads RGB images from data/ directory
2. Generates RGBD (4-channel) images using depth estimation
3. Uses numpy-based tiling (640x640 tiles with 80px overlap)
4. Runs YOLO prediction on each tile directly (preserves 4-channel RGBD)
5. Applies NMS to merge overlapping predictions
6. Saves results with original image names and detection counts

Note: This implementation uses manual tiling instead of SAHI to correctly
handle 4-channel RGBD images (SAHI's PIL-based slicing corrupts them).

Supports:
- Single model inference with auto-detection of depth model from trained weights
- Batch inference across all available RGBD models
- Comparison report generation across models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_vision.factory import DepthEstimatorFactory
from depth_vision.utils import normalize_depth
from utils.depth_model_config import (
    DEPTH_MODEL_CONFIGS,
    DEFAULT_DEPTH_CONFIG,
    extract_depth_model_from_weights,
    get_depth_model_name_from_weights,
    get_depth_config,
)
from utils.tiling import tile_image, tile_coordinates

# Paths
ROOT_DIR = PROJECT_ROOT
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs" / "segment"
OUTPUT_DIR = ROOT_DIR / "output" / "sahi_rgbd_predictions"

# Tiling parameters (matching training config)
TILE_SIZE = 640
OVERLAP = 80  # pixels, same as training

# Image extensions to process
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


def find_all_rgbd_models() -> List[Tuple[str, Path]]:
    """
    Find all trained RGBD models in the runs directory.

    Returns:
        List of tuples (model_name, weights_path) sorted by modification time (newest first)
    """
    models = []

    # Find all rgbd_* runs with weights
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        if not run_name.startswith("rgbd_"):
            continue

        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            continue

        # Extract model name from run directory
        depth_model_name = get_depth_model_name_from_weights(weights_path)
        if depth_model_name is None:
            depth_model_name = "unknown"

        models.append((depth_model_name, weights_path, weights_path.stat().st_mtime))

    # Sort by modification time (newest first) and remove duplicates (keep newest)
    models.sort(key=lambda x: x[2], reverse=True)

    # Deduplicate by model name, keeping the newest run for each model
    seen = set()
    unique_models = []
    for model_name, weights_path, _ in models:
        if model_name not in seen:
            seen.add(model_name)
            unique_models.append((model_name, weights_path))

    return unique_models


def find_model_weights(
    model_name: Optional[str] = None, model_type: str = "auto"
) -> Path:
    """
    Find model weights for a specific depth model or the latest model.

    Args:
        model_name: Specific depth model name (e.g., "depth_anything_large", "zoedepth_K")
                   If None, finds the latest trained model.
        model_type: Type of model to find ("rgbd", "rgb", or "auto")
                   "auto" tries RGBD first, falls back to RGB

    Returns:
        Path to model weights
    """
    if model_type == "rgbd" or model_type == "auto":
        # If specific model requested, look for matching runs
        if model_name:
            # Pattern: rgbd_<model_name>_<timestamp>
            pattern = f"rgbd_{model_name}_*/weights/best.pt"
            rgbd_candidates = sorted(
                RUNS_DIR.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if rgbd_candidates:
                return rgbd_candidates[0]

            # Also check default dataset pattern
            if model_name == "default" or model_name == "depth_anything_large":
                rgbd_candidates = sorted(
                    RUNS_DIR.glob("rgbd_default_*/weights/best.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if rgbd_candidates:
                    return rgbd_candidates[0]

            raise FileNotFoundError(
                f"No RGBD model weights found for '{model_name}' under {RUNS_DIR}. "
                f"Available models: {[m for m, _ in find_all_rgbd_models()]}"
            )

        # No specific model - find any RGBD model (newest first)
        rgbd_candidates = sorted(
            RUNS_DIR.glob("rgbd_*/weights/best.pt"),
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


def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    
    Args:
        boxes: Array of boxes in xyxy format, shape (N, 4)
        scores: Array of scores, shape (N,)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def run_inference_for_model(
    weights_path: Path,
    depth_model_name: str,
    depth_estimator_type: str,
    depth_config: Dict,
    image_files: List[Path],
    output_base_dir: Path,
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Run tiled inference for a single model using manual numpy-based tiling.
    
    This bypasses SAHI's PIL-based slicing which corrupts 4-channel RGBD images.

    Args:
        weights_path: Path to YOLO model weights
        depth_model_name: Name of the depth model (for output naming)
        depth_estimator_type: Type of depth estimator (e.g., "depth_anything")
        depth_config: Configuration dict for the depth estimator
        image_files: List of input image paths
        output_base_dir: Base output directory for results
        confidence: Detection confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        Dictionary with inference statistics
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference: {depth_model_name}")
    print(f"Weights: {weights_path}")
    print(f"Depth estimator: {depth_estimator_type} with config {depth_config}")
    print(f"{'=' * 80}")

    # Initialize depth estimator
    print(f"Initializing {depth_estimator_type} estimator...")
    estimator = DepthEstimatorFactory.create(depth_estimator_type, **depth_config)
    print("Estimator ready!")

    # Load YOLO model directly
    model = YOLO(str(weights_path))
    
    # Get class names from model
    category_names = []
    if hasattr(model, 'names'):
        category_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_base_dir / f"{depth_model_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(image_files)} images with manual tiling...")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Overlap: {OVERLAP}px")
    print("=" * 80)

    # Statistics tracking
    total_detections = 0
    image_stats = []

    # Process each image
    for img_path in tqdm(image_files, desc=f"Processing ({depth_model_name})"):
        try:
            # Read RGB image
            rgb_image = cv2.imread(str(img_path))
            if rgb_image is None:
                print(f"\n⚠️  Could not read {img_path.name}, skipping...")
                continue

            original_shape = rgb_image.shape[:2]

            # Generate RGBD image (4-channel BGRD)
            rgbd_image = create_rgbd_image(rgb_image, estimator)

            # Manual tiling using numpy (preserves all 4 channels!)
            tiles = tile_image(rgbd_image, TILE_SIZE, OVERLAP)
            
            # Collect all detections from tiles
            all_boxes = []
            all_scores = []
            all_class_ids = []
            all_masks = []
            
            for tile_img, y_start, x_start in tiles:
                # Run YOLO prediction directly on tile (numpy array)
                results = model.predict(
                    source=tile_img,
                    save=False,
                    verbose=False,
                    conf=confidence,
                    imgsz=TILE_SIZE,
                )
                
                if not results or results[0].boxes is None:
                    continue
                    
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Shift coordinates to full-image space
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box
                    # Apply offset for tile position
                    x1 += x_start
                    y1 += y_start
                    x2 += x_start
                    y2 += y_start
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, original_shape[1]))
                    y1 = max(0, min(y1, original_shape[0]))
                    x2 = max(0, min(x2, original_shape[1]))
                    y2 = max(0, min(y2, original_shape[0]))
                    
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    all_class_ids.append(class_id)
            
            # Apply NMS to merge overlapping predictions from different tiles
            if all_boxes:
                boxes_array = np.array(all_boxes)
                scores_array = np.array(all_scores)
                keep_indices = apply_nms(boxes_array, scores_array, iou_threshold)
                
                final_boxes = [all_boxes[i] for i in keep_indices]
                final_scores = [all_scores[i] for i in keep_indices]
                final_class_ids = [all_class_ids[i] for i in keep_indices]
            else:
                final_boxes = []
                final_scores = []
                final_class_ids = []

            # Count detections
            num_detections = len(final_boxes)
            total_detections += num_detections

            # Store stats
            image_stats.append(
                {
                    "filename": img_path.name,
                    "detections": num_detections,
                }
            )

            # Draw detections on original RGB image and save
            output_path = run_output_dir / img_path.name
            annotated_image = rgb_image.copy()

            for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Get category name
                if 0 <= class_id < len(category_names):
                    category = category_names[class_id]
                else:
                    category = str(class_id)

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence
                label = f"{category}: {score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

            # Save annotated image
            cv2.imwrite(str(output_path), annotated_image)

        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Write statistics file
    stats_file = run_output_dir / "detection_stats.txt"
    avg_detections = total_detections / len(image_stats) if image_stats else 0

    with open(stats_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Tiled RGBD Detection Statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Depth Model: {depth_model_name}\n")
        f.write(f"Depth Estimator: {depth_estimator_type}\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"\nTile Configuration:\n")
        f.write(f"  Tile size: {TILE_SIZE}x{TILE_SIZE}\n")
        f.write(f"  Overlap: {OVERLAP}px\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Image Detection Counts:\n")
        f.write("=" * 80 + "\n")

        for stat in image_stats:
            f.write(f"{stat['filename']}: {stat['detections']} detections\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total images processed: {len(image_stats)}\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per image: {avg_detections:.2f}\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Inference complete for {depth_model_name}!")
    print(f"✓ Results saved to: {run_output_dir}")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Average detections per image: {avg_detections:.2f}")

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": depth_model_name,
        "weights_path": str(weights_path),
        "depth_estimator": depth_estimator_type,
        "total_images": len(image_stats),
        "total_detections": total_detections,
        "avg_detections": avg_detections,
        "output_dir": str(run_output_dir),
        "image_stats": image_stats,
    }


def generate_comparison_report(
    results: List[Dict], output_dir: Path, timestamp: str
) -> Path:
    """
    Generate a comparison report across multiple models.

    Args:
        results: List of inference results from run_inference_for_model
        output_dir: Directory to save the report
        timestamp: Timestamp string for the report filename

    Returns:
        Path to the generated CSV report
    """
    # Create comparison dataframe
    comparison_data = []
    for result in results:
        comparison_data.append(
            {
                "Model": result["model_name"],
                "Depth Estimator": result["depth_estimator"],
                "Total Images": result["total_images"],
                "Total Detections": result["total_detections"],
                "Avg Detections": result["avg_detections"],
                "Weights Path": result["weights_path"],
            }
        )

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Total Detections", ascending=False)

    # Save CSV
    csv_path = output_dir / f"comparison_rgbd_inference_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Report saved to: {csv_path}")

    return csv_path


def collect_input_images() -> List[Path]:
    """Collect all input images from the data directory."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(DATA_DIR.glob(f"*{ext}"))
        image_files.extend(DATA_DIR.glob(f"*{ext.upper()}"))

    if not image_files:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    return sorted(image_files)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SAHI-based RGBD inference with automatic depth model detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with auto-detected depth model (from latest trained weights)
  python yolo_rgbd_inference_sahi.py

  # Run inference with a specific depth model
  python yolo_rgbd_inference_sahi.py --depth_model depth_anything_large

  # Run inference with specific weights file
  python yolo_rgbd_inference_sahi.py --weights /path/to/best.pt

  # Run inference across ALL trained RGBD models and generate comparison
  python yolo_rgbd_inference_sahi.py --all_models

  # List available trained models
  python yolo_rgbd_inference_sahi.py --list_models

Available depth models: {}
""".format(
            ", ".join(DEPTH_MODEL_CONFIGS.keys())
        ),
    )

    parser.add_argument(
        "--depth_model",
        type=str,
        default=None,
        help="Specific depth model to use (e.g., depth_anything_large, zoedepth_K). "
        "If not specified, auto-detects from trained weights.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to specific model weights. If not specified, finds the latest "
        "trained RGBD model matching the depth_model.",
    )

    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run inference across ALL available trained RGBD models and generate "
        "a comparison report.",
    )

    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all available trained RGBD models and exit.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )

    return parser.parse_args()


def main():
    """Main entry point for SAHI-based RGBD inference pipeline."""
    args = parse_args()

    print("=" * 80)
    print("SAHI-based RGBD Inference Pipeline")
    print("=" * 80)

    # List models mode
    if args.list_models:
        print("\nAvailable trained RGBD models:")
        print("-" * 40)
        models = find_all_rgbd_models()
        if not models:
            print("No trained RGBD models found.")
        else:
            for model_name, weights_path in models:
                print(f"  • {model_name}")
                print(f"    Weights: {weights_path}")
        print("-" * 40)
        print(f"\nSupported depth estimators: {list(DEPTH_MODEL_CONFIGS.keys())}")
        return

    # Collect input images
    image_files = collect_input_images()
    print(f"Found {len(image_files)} images in {DATA_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Batch mode: run all models
    if args.all_models:
        print("\n🔄 Batch mode: Running inference for ALL trained RGBD models")
        models = find_all_rgbd_models()

        if not models:
            print("❌ No trained RGBD models found!")
            return

        print(f"Found {len(models)} trained models: {[m for m, _ in models]}")

        all_results = []
        for model_name, weights_path in models:
            try:
                # Get depth config for this model
                depth_type, depth_config = get_depth_config(model_name)

                result = run_inference_for_model(
                    weights_path=weights_path,
                    depth_model_name=model_name,
                    depth_estimator_type=depth_type,
                    depth_config=depth_config,
                    image_files=image_files,
                    output_base_dir=OUTPUT_DIR,
                    confidence=args.confidence,
                )
                all_results.append(result)

            except Exception as e:
                print(f"\n❌ Failed to process {model_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Generate comparison report
        if all_results:
            generate_comparison_report(all_results, OUTPUT_DIR, timestamp)

        print("\n" + "=" * 80)
        print(f"✓ Batch inference complete! Processed {len(all_results)} models.")
        print("=" * 80)
        return

    # Single model mode
    if args.weights:
        # Use specified weights
        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Auto-detect depth model from weights
        depth_type, depth_config = extract_depth_model_from_weights(weights_path)
        depth_model_name = get_depth_model_name_from_weights(weights_path) or "unknown"

    elif args.depth_model:
        # Find weights for specified depth model
        weights_path = find_model_weights(model_name=args.depth_model, model_type="rgbd")
        depth_type, depth_config = get_depth_config(args.depth_model)
        depth_model_name = args.depth_model

    else:
        # Auto-detect: use latest trained RGBD model
        weights_path = find_model_weights(model_type="auto")

        # Detect depth model from the found weights
        depth_type, depth_config = extract_depth_model_from_weights(weights_path)
        depth_model_name = get_depth_model_name_from_weights(weights_path) or "auto"

    print(f"Using depth model: {depth_model_name}")
    print(f"Depth estimator: {depth_type} with config {depth_config}")
    print(f"Weights: {weights_path}")

    # Run inference
    result = run_inference_for_model(
        weights_path=weights_path,
        depth_model_name=depth_model_name,
        depth_estimator_type=depth_type,
        depth_config=depth_config,
        image_files=image_files,
        output_base_dir=OUTPUT_DIR,
        confidence=args.confidence,
    )

    print("\n" + "=" * 80)
    print("✓ Inference complete!")
    print(f"✓ Results saved to: {result['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
