# Depth Pro Setup Guide

## About Depth Pro

Depth Pro is Apple's ML Research foundation model for zero-shot metric monocular depth estimation. It produces high-resolution depth maps (up to 2.25 megapixels) with sharp boundaries and fine-grained details in less than a second.

**Paper:** [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)

## Features

- ✅ Zero-shot metric depth estimation
- ✅ High-resolution output (up to 2.25 megapixels)
- ✅ Sharp boundary delineation
- ✅ Fast inference (~0.3s on GPU)
- ✅ Metric depth values in meters
- ✅ Focal length estimation (if not provided)

## Installation

### Step 1: Install required packages

```bash
pip install huggingface-hub
pip install git+https://github.com/apple/ml-depth-pro.git
```

### Step 2: Model download (automatic)

The pretrained model will be **automatically downloaded from HuggingFace** on first use and cached locally (similar to other models like Depth Anything, MiDaS, etc.). No manual download needed!

The model will be cached in your HuggingFace cache directory (typically `~/.cache/huggingface/hub/`).

**Alternative: Manual download**

If you prefer to download manually:

```bash
# Download from HuggingFace
huggingface-cli download apple/DepthPro depth_pro.pt
```

## Usage

### Basic Usage

```python
from depth_vision.factory import DepthEstimatorFactory
import cv2

# Create Depth Pro estimator
estimator = DepthEstimatorFactory.create("depth_pro")

# Load and process image
image = cv2.imread("path/to/image.jpg")
depth_map = estimator.estimate(image)

# depth_map contains metric depth in meters
print(f"Depth range: {depth_map.min():.2f}m to {depth_map.max():.2f}m")
```

### Get Model Information

```python
info = estimator.get_model_info()
print(info)
# Output:
# {
#     'name': 'Depth Pro',
#     'variant': 'foundation',
#     'framework': 'PyTorch',
#     'device': 'cuda:0',
#     'properties': {
#         'size': '~2GB',
#         'accuracy': 'highest',
#         'speed': 'fast',
#         'metric_depth': True,
#         'max_resolution': '2.25MP'
#     },
#     'repository': 'apple/ml-depth-pro',
#     'paper': 'https://arxiv.org/abs/2410.02073'
# }
```

### Compare with Other Models

```python
from depth_vision.factory import create_depth_estimator
import cv2

image = cv2.imread("path/to/image.jpg")

# Compare different models
models = ["depth_pro", "depth_anything", "midas"]

for model_name in models:
    estimator = create_depth_estimator(model_name)
    depth_map = estimator.estimate(image)
    print(f"{model_name}: depth range [{depth_map.min():.2f}, {depth_map.max():.2f}]")
```

## Key Differences from Other Models

| Feature | Depth Pro | Depth Anything | MiDaS | ZoeDepth |
|---------|-----------|----------------|-------|----------|
| Metric Depth | ✅ Yes (meters) | ❌ Relative | ❌ Relative | ✅ Yes |
| Resolution | Up to 2.25MP | Variable | Variable | Variable |
| Speed (GPU) | ~0.3s | ~0.2-0.5s | ~0.3-0.8s | ~0.4s |
| Boundary Quality | Excellent | Very Good | Good | Good |
| Model Size | ~2GB | ~100MB-1.3GB | ~80MB-1.3GB | ~350MB |

## Troubleshooting

### Import Error

If you see:
```
ImportError: Depth Pro is not installed
```

Install the package:
```bash
pip install git+https://github.com/apple/ml-depth-pro.git
```

### HuggingFace Hub Error

If you see:
```
ImportError: huggingface_hub is required to download Depth Pro
```

Install huggingface-hub:
```bash
pip install huggingface-hub
```

### Model Download Issues

If the automatic download fails, you can:

1. **Check your internet connection**

2. **Manually download from HuggingFace:**
   ```bash
   huggingface-cli download apple/DepthPro depth_pro.pt
   ```

3. **Set HuggingFace cache directory (optional):**
   ```bash
   export HF_HOME=/path/to/your/cache
   ```

4. **Clear cache and retry:**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--apple--DepthPro
   # Then run your script again
   ```

## Citation

If you use Depth Pro in your research, please cite:

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

## License

Depth Pro is released by Apple Inc. under their license terms. Please check the [official repository](https://github.com/apple/ml-depth-pro) for license details.
