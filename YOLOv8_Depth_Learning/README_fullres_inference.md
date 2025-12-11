# Full-Resolution RGBD Inference Pipeline

This script performs end-to-end inference on full-resolution RGB images using RGBD-enhanced YOLO models.

## Overview

The pipeline:
1. Reads RGB images from the `data/` directory
2. Generates RGBD (4-channel) images using Depth Anything Large depth estimation
3. Tiles RGBD images into 640x640 patches with 80px overlap
4. Runs YOLO prediction on each tile
5. Reassembles prediction tiles back into full-resolution images using weighted blending
6. Saves results with original filenames

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Ultralytics YOLO
- OpenCV
- NumPy
- Trained YOLO model weights (RGBD or RGB)

## Usage

### Basic Usage

Simply run the script from the YOLOv8_Depth_Learning directory:

```bash
cd YOLOv8_Depth_Learning
python yolo_rgbd_inference_fullres.py
```

The script will:
- Look for images in `../data/` directory
- Automatically find the latest trained model (prefers RGBD models, falls back to RGB)
- Process all images and save results to `../output/fullres_rgbd_predictions/run_<timestamp>/`

### Input Images

Place your full-resolution images in the `data/` directory. Supported formats:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tif`, `.tiff`

### Output

Results are saved to: `output/fullres_rgbd_predictions/run_<timestamp>/`

Each processed image is saved with its original filename, containing:
- Detection boxes
- Segmentation masks
- Confidence scores

## Technical Details

### Tile Processing

- **Tile size**: 640x640 pixels (matches training size)
- **Overlap**: 80 pixels (ensures smooth transitions)
- **Blending**: Distance-weighted blending to reduce edge artifacts

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

# Run inference
python yolo_rgbd_inference_fullres.py

# Check results
ls ../output/fullres_rgbd_predictions/run_20231211_143022/
# DJI_0398_2pD.JPG  (with predictions overlaid)
# IMG_2015_1pK.JPG  (with predictions overlaid)
# ...
```

## Performance Notes

- GPU strongly recommended for depth estimation
- Processing time: ~10-30 seconds per image (depending on resolution and GPU)
- Memory usage: Proportional to image resolution

## Troubleshooting

### No model weights found

```
FileNotFoundError: No model weights found under runs/segment.
```

**Solution**: Train a model first using:
- `train_rgbd_model.py` for RGBD models, or
- Standard RGB model training scripts

### Out of memory

If you encounter CUDA out of memory errors with large images:
1. Process images in smaller batches
2. Reduce image resolution before processing
3. Use CPU mode (slower but requires less memory)

## Related Scripts

- `generate_rgbd_dataset.py` - Generate RGBD training dataset
- `train_rgbd_model.py` - Train YOLO on RGBD data
- `yolo_depth_inference.py` - Inference with more control options
- `yolo_depth_inference_fullres.py` - Simpler non-tiled inference wrapper
