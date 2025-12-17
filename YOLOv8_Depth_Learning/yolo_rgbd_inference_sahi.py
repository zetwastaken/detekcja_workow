"""
RGBD inference pipeline using SAHI for sliced detection.

This script performs end-to-end inference on full-resolution RGB images:
1. Reads RGB images from data/ directory
2. Generates RGBD (4-channel) images using depth estimation
3. Uses SAHI (Slicing Aided Hyper Inference) for tiling and detection
4. Aggregates detections and counts detected items
5. Saves results with original image names and detection counts

SAHI provides better handling of small objects and overlapping tiles compared
to manual tiling approaches.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# SAHI imports
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image_as_pil
except ImportError:
    print("ERROR: SAHI is not installed. Please install it with: pip install sahi")
    sys.exit(1)

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
OUTPUT_DIR = ROOT_DIR / "output" / "sahi_rgbd_predictions"

# SAHI parameters
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.125  # 80/640 = 0.125
OVERLAP_WIDTH_RATIO = 0.125

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


class RGBDDetectionModel(AutoDetectionModel):
    """
    Custom SAHI detection model wrapper for RGBD images.
    
    Handles 4-channel RGBD inputs by converting them for YOLO inference.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, 
                 device: str = None, use_rgbd: bool = True):
        """
        Initialize RGBD detection model.
        
        Args:
            model_path: Path to YOLO weights
            confidence_threshold: Confidence threshold for predictions
            device: Device to run on ('cuda:0', 'cpu', etc.)
            use_rgbd: Whether to use 4-channel RGBD (True) or 3-channel RGB (False)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_rgbd = use_rgbd
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Set model parameters
        self.category_mapping = None
        self.category_names = []
        
        # Get class names from model
        if hasattr(self.model, 'names'):
            self.category_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
    
    def perform_inference(self, image: np.ndarray):
        """
        Perform inference on image.
        
        Args:
            image: Input image (can be RGB or RGBD)
        
        Returns:
            Prediction results
        """
        # If image has 4 channels and we're not using RGBD, drop the depth channel
        if image.shape[-1] == 4 and not self.use_rgbd:
            image = image[:, :, :3]
        
        # Run YOLO prediction
        results = self.model.predict(
            source=image,
            save=False,
            verbose=False,
            conf=self.confidence_threshold,
            device=self.device,
        )
        
        return results
    
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list = [[0, 0]],
        full_shape=None,
    ):
        """
        Convert YOLO predictions to SAHI object prediction format.
        
        Args:
            shift_amount_list: Amount to shift predictions (for tiling)
            full_shape: Shape of full image
        
        Returns:
            List of ObjectPrediction objects
        """
        from sahi.prediction import ObjectPrediction
        
        predictions = self._original_predictions[0]  # YOLO returns list with single result
        object_prediction_list = []
        
        # Handle case where no detections
        if predictions.boxes is None or len(predictions.boxes) == 0:
            return object_prediction_list
        
        shift_amount = shift_amount_list[0]
        
        # Extract boxes, scores, and class IDs
        boxes = predictions.boxes.xyxy.cpu().numpy()
        scores = predictions.boxes.conf.cpu().numpy()
        class_ids = predictions.boxes.cls.cpu().numpy().astype(int)
        
        # Extract masks if available
        masks = None
        if hasattr(predictions, 'masks') and predictions.masks is not None:
            masks = predictions.masks.data.cpu().numpy()
        
        for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            
            # Apply shift for tiled predictions
            x1 += shift_amount[0]
            y1 += shift_amount[1]
            x2 += shift_amount[0]
            y2 += shift_amount[1]
            
            # Get category name
            category_name = self.category_names[class_id] if class_id < len(self.category_names) else str(class_id)
            
            # Create bbox in COCO format [x, y, width, height]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            # Create mask if available
            mask = None
            if masks is not None and idx < len(masks):
                mask = masks[idx]
            
            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=int(class_id),
                category_name=category_name,
                bool_mask=mask,
                score=float(score),
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            
            object_prediction_list.append(object_prediction)
        
        return object_prediction_list


def main():
    """Main SAHI-based inference pipeline."""
    print("=" * 80)
    print("SAHI-based RGBD Inference Pipeline")
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
    
    # Initialize depth estimator
    print("Initializing Depth Anything Large estimator...")
    estimator = DepthEstimatorFactory.create(DEPTH_MODEL, **DEPTH_CONFIG)
    print("Estimator ready!")
    
    # Initialize SAHI detection model
    use_rgbd = (model_type == "RGBD")
    detection_model = RGBDDetectionModel(
        model_path=str(weights_path),
        confidence_threshold=0.25,
        use_rgbd=use_rgbd,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stats file
    stats_file = run_output_dir / "detection_stats.txt"
    
    print(f"\nProcessing {len(image_files)} images with SAHI...")
    print(f"Slice size: {SLICE_WIDTH}x{SLICE_HEIGHT}")
    print(f"Overlap ratio: {OVERLAP_WIDTH_RATIO}")
    print("=" * 80)
    
    # Statistics tracking
    total_detections = 0
    image_stats = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read RGB image
            rgb_image = cv2.imread(str(img_path))
            if rgb_image is None:
                print(f"\n⚠️  Could not read {img_path.name}, skipping...")
                continue
            
            # Generate RGBD image
            rgbd_image = create_rgbd_image(rgb_image, estimator)
            
            # Save temporary RGBD image for SAHI processing
            temp_rgbd_path = run_output_dir / f"temp_{img_path.stem}.tiff"
            cv2.imwrite(str(temp_rgbd_path), rgbd_image)
            
            # Perform sliced prediction with SAHI
            result = get_sliced_prediction(
                image=str(temp_rgbd_path),
                detection_model=detection_model,
                slice_height=SLICE_HEIGHT,
                slice_width=SLICE_WIDTH,
                overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
                overlap_width_ratio=OVERLAP_WIDTH_RATIO,
                verbose=0,
            )
            
            # Count detections
            num_detections = len(result.object_prediction_list)
            total_detections += num_detections
            
            # Store stats
            image_stats.append({
                'filename': img_path.name,
                'detections': num_detections,
            })
            
            # Export visualization
            output_path = run_output_dir / img_path.name
            result.export_visuals(
                export_dir=str(run_output_dir),
                file_name=img_path.stem,
                rect_th=2,
                text_size=0.5,
                text_th=2,
            )
            
            # Rename the output file to match original name
            exported_file = run_output_dir / f"{img_path.stem}.png"
            if exported_file.exists():
                # Convert to same format as input if needed
                img_result = cv2.imread(str(exported_file))
                cv2.imwrite(str(output_path), img_result)
                exported_file.unlink()  # Remove temporary PNG
            
            # Clean up temporary RGBD file
            if temp_rgbd_path.exists():
                temp_rgbd_path.unlink()
            
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write statistics
    with open(stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SAHI-based RGBD Detection Statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {weights_path}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"\nSlice Configuration:\n")
        f.write(f"  Slice size: {SLICE_WIDTH}x{SLICE_HEIGHT}\n")
        f.write(f"  Overlap ratio: {OVERLAP_WIDTH_RATIO}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Image Detection Counts:\n")
        f.write("=" * 80 + "\n")
        
        for stat in image_stats:
            f.write(f"{stat['filename']}: {stat['detections']} detections\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total images processed: {len(image_stats)}\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per image: {total_detections / len(image_stats):.2f}\n")
        f.write("=" * 80 + "\n")
    
    print("\n" + "=" * 80)
    print("✓ Inference complete!")
    print(f"✓ Results saved to: {run_output_dir}")
    print(f"✓ Total images processed: {len(image_stats)}")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Average detections per image: {total_detections / len(image_stats):.2f}")
    print(f"✓ Detection statistics saved to: {stats_file}")
    print("=" * 80)
    
    # Cleanup GPU memory
    del estimator
    del detection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
