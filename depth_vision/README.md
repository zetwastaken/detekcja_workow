# Depth Vision - Multi-Model Framework

Monocular depth estimation using multiple state-of-the-art models:
- **MiDaS** - Intel's robust depth estimation model
- **Depth Anything V2** - Latest high-performance depth estimator
- **ZoeDepth** - High-quality metric depth estimation

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

## Available Models

### MiDaS (Intel)
- `DPT_Large` - Highest accuracy, ~1.3GB
- `DPT_Hybrid` - Balanced accuracy/speed, ~800MB (recommended)
- `MiDaS_small` - Fastest, ~80MB

### Depth Anything V2
- `small` - Fast inference, ~100MB
- `base` - Balanced, ~400MB  
- `large` - Highest accuracy, ~1.3GB

### ZoeDepth
- `NK` - General purpose (indoor + outdoor), ~350MB (recommended)
- `N` - Indoor scenes (NYU dataset), ~350MB
- `K` - Outdoor/driving scenes (KITTI dataset), ~350MB

## Usage

### Using the Factory Pattern (Recommended)

```python
from depth_vision import create_depth_estimator, visualize_depth
import cv2

# Create a MiDaS estimator
estimator = create_depth_estimator("midas", model_type="DPT_Hybrid")

# Or create a Depth Anything estimator
estimator = create_depth_estimator("depth_anything", model_size="small")

# Or create a ZoeDepth estimator
estimator = create_depth_estimator("zoedepth", model_type="NK")

# Process image
image = cv2.imread("input.jpg")
depth_map = estimator.estimate(image)

# Visualize
colored_depth = visualize_depth(depth_map)
cv2.imwrite("depth_output.png", colored_depth)
```

### Direct Model Usage

```python
from depth_vision.estimators import MiDaSDepthEstimator, DepthAnythingEstimator, ZoeDepthEstimator

# MiDaS
midas = MiDaSDepthEstimator(model_type="DPT_Hybrid")
depth = midas.estimate(image)

# Depth Anything
depth_anything = DepthAnythingEstimator(model_size="base")
depth = depth_anything.estimate(image)

# ZoeDepth
zoedepth = ZoeDepthEstimator(model_type="NK")
depth = zoedepth.estimate(image)
```

### Comparing Multiple Models

```python
from depth_vision import DepthEstimatorFactory
import cv2

image = cv2.imread("input.jpg")

# Compare different models
models = [
    ("midas", {"model_type": "DPT_Hybrid"}),
    ("depth_anything", {"model_size": "small"}),
    ("zoedepth", {"model_type": "NK"}),
]

for model_name, config in models:
    estimator = DepthEstimatorFactory.create(model_name, **config)
    depth = estimator.estimate(image)
    
    # Get model info
    info = estimator.get_model_info()
    print(f"{info['name']} - Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
```

### CLI (Command Line Interface)

Process images from command line (MiDaS):

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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+
- transformers 4.35+ (for Depth Anything)
- timm 0.9+ (for vision models)

## Project Structure

```
depth_vision/
  __init__.py           - Package exports
  base.py               - Base estimator class
  factory.py            - Factory for creating estimators
  utils.py              - Utility functions (visualization, etc.)
  estimators/
    __init__.py
    midas.py            - MiDaS implementation
    depth_anything.py   - Depth Anything V2 implementation
    zoedepth.py         - ZoeDepth implementation
```

## Notes

- First run will download models from HuggingFace/PyTorch Hub
  - MiDaS DPT_Large: ~1.3GB
  - Depth Anything Small: ~100MB
  - ZoeDepth: ~350MB
- GPU acceleration is automatically used if CUDA is available
- Output depth maps:
  - MiDaS & Depth Anything: inverse depth (higher values = closer objects)
  - ZoeDepth: inverted metric depth (higher values = closer objects)

## Examples

See `examples/examples_multi_model.py` for comprehensive usage examples:
- Basic usage with different models
- Comparing model outputs
- Batch processing
- Custom estimator implementation
