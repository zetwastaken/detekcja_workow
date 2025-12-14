"""
SAHI Sliced Inference for YOLOv8 Segmentation Model
Performs sliced inference on large images for improved detection of small objects (sandbags)
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
import cv2


class YOLOv8SahiInference:
    """SAHI-based sliced inference for YOLOv8 segmentation models"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        device: str = "cuda:0",
        image_size: int = 640
    ):
        """
        Initialize SAHI inference with YOLOv8 model
        
        Args:
            model_path: Path to trained YOLOv8 model weights (.pt file)
            confidence_threshold: Confidence threshold for predictions
            device: Device to run inference on ('cuda:0' or 'cpu')
            image_size: Image size for model inference
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size
        
        # Initialize SAHI detection model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=image_size
        )
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Device: {device}")
        print(f"✓ Confidence threshold: {confidence_threshold}")
    
    def predict_single_image(
        self,
        image_path: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        postprocess_type: str = "NMS",
        postprocess_match_threshold: float = 0.7,
        postprocess_class_agnostic: bool = True,
        verbose: int = 1
    ):
        """
        Perform sliced prediction on a single image
        
        Args:
            image_path: Path to input image
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Overlap ratio for height (0.0-1.0)
            overlap_width_ratio: Overlap ratio for width (0.0-1.0)
            postprocess_type: Type of postprocessing ('NMS' or 'GREEDYNMM')
            postprocess_match_threshold: IoU threshold for matching
            postprocess_class_agnostic: Whether to perform class-agnostic NMS
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
            
        Returns:
            SAHI prediction result
        """
        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            verbose=verbose
        )
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        save_visualizations: bool = True,
        save_txt: bool = True,
        postprocess_type: str = "NMS",
        postprocess_match_threshold: float = 0.7,
        postprocess_class_agnostic: bool = True
    ):
        """
        Perform sliced prediction on multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Overlap ratio for height
            overlap_width_ratio: Overlap ratio for width
            save_visualizations: Whether to save visualized predictions
            save_txt: Whether to save predictions in YOLO format
            postprocess_type: Type of postprocessing
            postprocess_match_threshold: IoU threshold for matching
            postprocess_class_agnostic: Whether to perform class-agnostic NMS
        """
        # Create output directories
        output_path = Path(output_dir)
        vis_dir = output_path / "visualizations"
        labels_dir = output_path / "labels"
        
        if save_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)
        if save_txt:
            labels_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary = []
        
        print(f"\n🚀 Processing {len(image_paths)} images...")
        print(f"📊 Slice size: {slice_width}x{slice_height}")
        print(f"🔄 Overlap: {overlap_width_ratio*100}% x {overlap_height_ratio*100}%")
        print(f"📁 Output directory: {output_dir}\n")
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Get prediction
                result = self.predict_single_image(
                    image_path=img_path,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    postprocess_type=postprocess_type,
                    postprocess_match_threshold=postprocess_match_threshold,
                    postprocess_class_agnostic=postprocess_class_agnostic,
                    verbose=0
                )
                
                img_name = Path(img_path).stem
                num_detections = len(result.object_prediction_list)
                
                # Save visualization
                if save_visualizations:
                    # vis_path = vis_dir / f"{img_name}_pred.jpg"
                    result.export_visuals(export_dir=str(vis_dir), file_name=f"{img_name}_pred.jpg",text_size=0.8, hide_labels=True)
                    # result.export_visuals(
                    #     export_dir=str(vis_dir), 
                    #     file_name=f"{img_name}_pred.jpg",
                    #     text_size=0.3,  # Smaller text size (default is 0.5)
                    #     text_th=1       # Thinner text thickness (default is 2)
                    # )
                
                
                # Save labels in YOLO format
                if save_txt:
                    label_path = labels_dir / f"{img_name}.txt"
                    self._save_yolo_labels(result, label_path, img_path)
                
                results_summary.append({
                    'image': img_name,
                    'detections': num_detections
                })
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results_summary.append({
                    'image': Path(img_path).stem,
                    'detections': 0,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n✅ Processing complete!")
        print(f"Total images: {len(image_paths)}")
        print(f"Total detections: {sum(r['detections'] for r in results_summary)}")
        
        return results_summary
    
    def _save_yolo_labels(self, result, label_path: Path, image_path: str):
        """
        Save predictions in YOLO segmentation format
        
        Args:
            result: SAHI prediction result
            label_path: Path to save label file
            image_path: Path to original image (to get dimensions)
        """
        # Read image to get dimensions
        image = read_image(image_path)
        img_height, img_width = image.shape[:2]
        
        with open(label_path, 'w') as f:
            for pred in result.object_prediction_list:
                class_id = pred.category.id
                
                # Check if segmentation mask exists
                if hasattr(pred, 'mask') and pred.mask is not None:
                    # Get segmentation points
                    segmentation = pred.mask.segmentation
                    
                    if segmentation and len(segmentation) > 0:
                        # Normalize coordinates
                        normalized_points = []
                        for point in segmentation:
                            x_norm = point[0] / img_width
                            y_norm = point[1] / img_height
                            normalized_points.extend([x_norm, y_norm])
                        
                        # Write to file: class_id x1 y1 x2 y2 ... xn yn
                        line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
                        f.write(line + "\n")
                else:
                    # Fallback to bounding box if no segmentation
                    bbox = pred.bbox
                    x_center = (bbox.minx + bbox.maxx) / 2 / img_width
                    y_center = (bbox.miny + bbox.maxy) / 2 / img_height
                    width = (bbox.maxx - bbox.minx) / img_width
                    height = (bbox.maxy - bbox.miny) / img_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def predict_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
        **kwargs
    ):
        """
        Perform sliced prediction on all images in a directory or a single image file
        
        Args:
            input_dir: Directory containing input images OR path to a single image file
            output_dir: Directory to save results
            image_extensions: Tuple of valid image extensions
            **kwargs: Additional arguments for predict_batch
        """
        input_path = Path(input_dir)
        
        # Check if input is a file or directory
        if input_path.is_file():
            # Single image file
            if input_path.suffix.lower() in image_extensions:
                image_paths = [str(input_path)]
            else:
                print(f"⚠️ File {input_dir} is not a supported image format")
                print(f"   Supported formats: {image_extensions}")
                return []
        elif input_path.is_dir():
            # Directory with multiple images
            # Find all images (case-insensitive)
            image_paths = []
            for ext in image_extensions:
                # Use case-insensitive glob pattern
                image_paths.extend(list(input_path.glob(f"*{ext}")))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for p in image_paths:
                p_lower = str(p).lower()
                if p_lower not in seen:
                    seen.add(p_lower)
                    unique_paths.append(str(p))
            
            image_paths = unique_paths
            
            if not image_paths:
                print(f"⚠️ No images found in {input_dir}")
                return []
        else:
            print(f"⚠️ Path {input_dir} does not exist")
            return []
        
        return self.predict_batch(
            image_paths=image_paths,
            output_dir=output_dir,
            **kwargs
        )


def main():
    """Example usage of SAHI sliced inference"""
    
    # Configuration
    MODEL_PATH = "S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/runs/segment/yolov8_sandbag_seg_v5/weights/best.pt"  # Adjust to your model
    # INPUT_DIR = "S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/data/IMG_2022_1pK.JPG"  # Directory with images to process
    INPUT_DIR = "S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/data"  # Directory with images to process
    OUTPUT_DIR = "S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/runs/segment/sahi_predictions"
    
    # SAHI parameters
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_HEIGHT_RATIO = 0.2  # 20% overlap
    OVERLAP_WIDTH_RATIO = 0.2   # 20% overlap
    CONFIDENCE_THRESHOLD = 0.4
    
    # Initialize inference
    print("🔧 Initializing SAHI inference...")
    inferencer = YOLOv8SahiInference(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cpu",  # Change to 'cpu' if no GPU
        image_size=640
    )
    
    # Run inference on directory
    results = inferencer.predict_from_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        save_visualizations=True,
        save_txt=True,
        postprocess_type="NMS",
        postprocess_match_threshold=0.7,  # Higher IoU threshold for better NMS
        postprocess_class_agnostic=True
    )
    
    print("\n📊 Results summary:")
    for result in results[:10]:  # Show first 10
        print(f"  {result['image']}: {result['detections']} detections")


if __name__ == "__main__":
    main()
