# MiDaS Depth Vision

Monocular depth estimation using Intel's MiDaS model.

## Setup

### 1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### CLI (Command Line Interface)

Process images from command line:

```bash
# Basic usage
python -m depth_vision.midas -i input.jpg

# Specify model type
python -m depth_vision.midas -i input.jpg -m DPT_Hybrid

# Use different colormap
python -m depth_vision.midas -i input.jpg -c VIRIDIS

# Save both raw and colored depth maps
python -m depth_vision.midas -i input.jpg --both

# Process multiple images
python -m depth_vision.midas -i image1.jpg image2.jpg image3.jpg

# Specify output path
python -m depth_vision.midas -i input.jpg -o depth_output.png
```

**Available Models:**
- `DPT_Large` - Highest accuracy, slowest (default)
- `DPT_Hybrid` - Good balance of speed and accuracy
- `MiDaS_small` - Fastest, lower accuracy

**Available Colormaps:**
- `INFERNO` (default)
- `VIRIDIS`
- `PLASMA`
- `MAGMA`
- `JET`
- `TURBO`
- `HOT`

### Python API

Use in your Python code:

```python
from depth_vision.midas import generate_depth_map, MiDaSDepthEstimator
import cv2

# Method 1: Simple function (loads model each time)
image = cv2.imread("input.jpg")
depth_map = generate_depth_map(image, model_type="DPT_Large")

# Method 2: Persistent estimator (efficient for multiple images)
estimator = MiDaSDepthEstimator(model_type="DPT_Large")

# Process single image
depth_map = estimator.estimate(image)

# Or get depth map with visualization
depth_map, colored_depth = estimator.estimate_with_visualization(image)

# Save results
cv2.imwrite("depth_raw.png", depth_map)
cv2.imwrite("depth_colored.png", colored_depth)
```

### Return Depth Map from Function

```python
from depth_vision.midas import MiDaSDepthEstimator
import cv2

# Initialize once
estimator = MiDaSDepthEstimator(model_type="DPT_Large")

# Use in your code
def process_image(image_path):
    image = cv2.imread(image_path)
    depth_map = estimator.estimate(image)
    return depth_map

# Use it
depth = process_image("my_image.jpg")
print(f"Depth map shape: {depth.shape}")
```

## Project Structure

```
depth_vision/
  midas/
    __init__.py      - Module exports
    __main__.py      - CLI entry point
    midas.py         - Core MiDaS implementation
    cli.py           - Command-line interface
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+

## Notes

- First run will download the MiDaS model from torch hub (~500MB for DPT_Large)
- GPU acceleration is automatically used if CUDA is available
- Output depth maps are inverse depth (higher values = closer objects)
