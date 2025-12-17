# SAHI-based RGBD Inference Pipeline

This script provides an advanced inference pipeline for full-resolution RGB images using SAHI (Slicing Aided Hyper Inference) for improved object detection with RGBD-enhanced YOLO models.

## Overview

SAHI is a computer vision library that improves object detection performance on high-resolution images by:
- Intelligently slicing images into overlapping tiles
- Running detection on each tile
- Merging results with sophisticated non-maximum suppression
- Handling small objects better than single-pass inference

The pipeline:
1. Reads RGB images from the `data/` directory
2. Generates RGBD (4-channel) images using Depth Anything Large depth estimation
3. Uses SAHI to slice RGBD images into 640x640 patches with configurable overlap
4. Runs YOLO prediction on each tile
5. Aggregates detections across tiles using SAHI's smart merging
6. **Counts and reports the number of detected items per image**
7. Saves results with original filenames and detection statistics

## Key Advantages over Manual Tiling

- **Better handling of objects on tile boundaries**: SAHI's NMS merges detections across tiles
- **Configurable overlap**: Easy to adjust overlap ratios for optimal performance
- **Detection counting**: Automatic counting and reporting of detected items
- **Statistics tracking**: Generates detailed statistics file with per-image detection counts
- **Production-ready**: SAHI is a well-tested library used in many computer vision applications

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Ultralytics YOLO
- SAHI >= 0.11.0
- OpenCV
- NumPy
- Trained YOLO model weights (RGBD or RGB)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Simply run the script from the YOLOv8_Depth_Learning directory:

```bash
cd YOLOv8_Depth_Learning
python yolo_rgbd_inference_sahi.py
```

The script will:
- Look for images in `../data/` directory
- Automatically find the latest trained model (prefers RGBD models, falls back to RGB)
- Process all images using SAHI sliced prediction
- Count detections for each image
- Save results to `../output/sahi_rgbd_predictions/run_<timestamp>/`
- Generate a `detection_stats.txt` file with detailed statistics

### Input Images

Place your full-resolution images in the `data/` directory. Supported formats:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tif`, `.tiff`

### Output

Results are saved to: `output/sahi_rgbd_predictions/run_<timestamp>/`

For each processed image:
- **Visualization**: Image with detection boxes, masks, and labels
- **Statistics**: Detection counts in `detection_stats.txt`

#### Detection Statistics File

The `detection_stats.txt` file includes:
```
================================================================================
SAHI-based RGBD Detection Statistics
================================================================================
Timestamp: 20231217_143022
Model: runs/segment/rgbd_depth_anything_large_20231215/weights/best.pt
Model Type: RGBD

Slice Configuration:
  Slice size: 640x640
  Overlap ratio: 0.125

================================================================================
Per-Image Detection Counts:
================================================================================
DJI_0398_2pD.JPG: 15 detections
IMG_2015_1pK.JPG: 23 detections
...

================================================================================
Total images processed: 10
Total detections: 187
Average detections per image: 18.70
================================================================================
```

## Configuration

You can modify the SAHI parameters at the top of the script:

```python
# SAHI parameters
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.125  # 80/640 = 0.125 (12.5% overlap)
OVERLAP_WIDTH_RATIO = 0.125
```

**Overlap Ratio Guide:**
- `0.0`: No overlap (may miss objects on boundaries)
- `0.125`: 12.5% overlap (80 pixels for 640x640 tiles) - **default, good balance**
- `0.25`: 25% overlap (160 pixels) - better for small objects, slower
- `0.5`: 50% overlap (320 pixels) - maximum coverage, much slower

## Technical Details

### SAHI Integration

The script includes a custom `RGBDDetectionModel` class that:
- Wraps YOLO models for SAHI compatibility
- Handles 4-channel RGBD inputs
- Converts YOLO predictions to SAHI's ObjectPrediction format
- Supports both RGBD (4-channel) and RGB (3-channel) models

### Depth Estimation

Uses **Depth Anything Large** model for depth map generation:
- High-quality monocular depth estimation
- Optimized for outdoor scenes
- Same model used for RGBD dataset generation

### Model Compatibility

The script supports both:
- **RGBD models** (4-channel input): Uses full RGBD tiles
- **RGB models** (3-channel input): Automatically extracts RGB channels from RGBD tiles

Auto-detection prioritizes RGBD models but falls back to RGB models if no RGBD models are found.

## Example

```bash
# Ensure images are in data/
ls ../data/
# DJI_0398_2pD.JPG
# IMG_2015_1pK.JPG
# ...

# Run SAHI-based inference
python yolo_rgbd_inference_sahi.py

# Output:
# ================================================================================
# SAHI-based RGBD Inference Pipeline
# ================================================================================
# Found 5 images in /path/to/data
# Loading RGBD model from: runs/segment/rgbd_depth_anything_large/weights/best.pt
# Initializing Depth Anything Large estimator...
# Estimator ready!
#
# Processing 5 images with SAHI...
# Slice size: 640x640
# Overlap ratio: 0.125
# ================================================================================
# Processing images: 100%|██████████| 5/5 [00:45<00:00,  9.12s/it]
#
# ================================================================================
# ✓ Inference complete!
# ✓ Results saved to: output/sahi_rgbd_predictions/run_20231217_143022/
# ✓ Total images processed: 5
# ✓ Total detections: 87
# ✓ Average detections per image: 17.40
# ✓ Detection statistics saved to: output/sahi_rgbd_predictions/run_20231217_143022/detection_stats.txt
# ================================================================================

# Check results
ls ../output/sahi_rgbd_predictions/run_20231217_143022/
# DJI_0398_2pD.JPG  (with predictions overlaid)
# IMG_2015_1pK.JPG  (with predictions overlaid)
# detection_stats.txt  (detection counts)
# ...
```

## Performance Notes

- GPU strongly recommended for both depth estimation and detection
- SAHI processing is slower than single-pass inference but provides better results
- Processing time: ~15-60 seconds per image (depending on resolution, overlap ratio, and GPU)
- Memory usage: Proportional to image resolution and overlap ratio

## Troubleshooting

### No model weights found

```
FileNotFoundError: No model weights found under runs/segment.
```

**Solution**: Train a model first using:
- `train_rgbd_model.py` for RGBD models, or
- Standard RGB model training scripts

### SAHI import error

```
ERROR: SAHI is not installed. Please install it with: pip install sahi
```

**Solution**: Install SAHI:
```bash
pip install sahi
```

### Out of memory

If you encounter CUDA out of memory errors with large images:
1. Reduce overlap ratio (e.g., from 0.125 to 0.0)
2. Process fewer images at a time
3. Reduce image resolution before processing
4. Use CPU mode (slower but requires less memory)

## Comparison with Manual Tiling

| Feature | SAHI (this script) | Manual Tiling (`yolo_rgbd_inference_fullres.py`) |
|---------|-------------------|--------------------------------------------------|
| Tile merging | Smart NMS-based merging | Weighted blending |
| Boundary handling | Excellent (merges overlapping detections) | Good (blends visually) |
| Detection counting | Built-in, per-image and total | Not available |
| Statistics | Detailed stats file | No statistics |
| Configuration | Simple overlap ratio | Manual overlap pixels |
| Performance | Slightly slower | Faster |
| Library dependency | SAHI required | No extra dependencies |

## Related Scripts

- `generate_rgbd_dataset.py` - Generate RGBD training dataset
- `train_rgbd_model.py` - Train YOLO on RGBD data
- `yolo_rgbd_inference_fullres.py` - Manual tiling approach (alternative to SAHI)
- `yolo_depth_inference.py` - Inference with more control options

## References

- [SAHI GitHub](https://github.com/obss/sahi)
- [SAHI Documentation](https://sahi.readthedocs.io/)
- Paper: "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection" (2022)
