# AI Assistant Instructions for detekcja_workow

Sandbag detection project using YOLOv8 segmentation with depth estimation augmentation.

## Project Architecture

### Three-Tier Pipeline System
1. **Depth Estimation** (`depth_vision/`) - Factory-pattern framework supporting 4 model families (MiDaS, Depth Anything V2, ZoeDepth, Marigold)
2. **Dataset Generation** (`YOLOv8_Depth_Learning/`) - Automated pipelines creating depth-only and RGBD (4-channel) datasets
3. **Training & Evaluation** - Batch training scripts with standardized hyperparameters for fair model comparison

### Key Data Flow
```
data/ (raw images)
  → depth_vision estimators → depth_maps/ (visualized depth)
  → tiling → depth_tiles/ (640×640, 80px overlap)
  → dataset generation → datasets/dataset_depth_* or dataset_rgbd_*
  → YOLO training → runs/segment/
  → comparison scripts → comparison_results/
```

## Critical Conventions

### Path Management
- **Always use `PROJECT_ROOT`** resolved from `Path(__file__).resolve().parent`
- Multi-level scripts access shared resources via `PROJECT_ROOT.parent` (workspace root)
- Example: `DATASETS_DIR = PROJECT_ROOT.parent / "datasets"`
- Dataset `data.yaml` must contain absolute paths for YOLOv8

### Tiling Standards
- Fixed parameters: 640×640 tiles, 80px overlap (stride=560)
- Use `utils/tiling.py::tile_image_with_names()` for consistent naming: `basename_y<Y>_x<X>.jpg`
- Train/valid splits determined by original `dataset_yolov8_V1` selection
- Depth tiles must match RGB tile selection for aligned RGBD datasets

### Depth Estimator Integration
- All estimators inherit `BaseDepthEstimator` (abstract: `load_model()`, `estimate()`)
- Create via factory: `DepthEstimatorFactory.create("midas", model_type="DPT_Large")`
- Input: BGR numpy array (OpenCV convention)
- Output: Raw depth maps (higher values = closer objects)
- Visualize with `depth_vision.utils.visualize_depth(depth, colormap=cv2.COLORMAP_INFERNO)`

### Training Configuration
- Base model: `yolov8n-seg.pt` for segmentation
- Hyperparameters standardized across all experiments (see `TRAINING_CONFIG` in `YOLOv8_Depth_Learning/train_all_depth_models.py`)
- Key settings: 500 epochs, patience=150, batch=32, imgsz=640, cosine LR
- Augmentation: degrees=15°, mosaic=1.0, fliplr=0.5, flipud=0.2, mixup=0.1

### Dataset Structure
```
datasets/dataset_<type>_<model>/
  data.yaml           # Absolute paths, nc: 1, names: [sandbag]
  train/
    images/          # .jpg tiles
    labels/          # .txt YOLO format (class x_center y_center width height)
  valid/
    images/
    labels/
```

## Development Workflow

### Code Quality
- Auto-formatting: Black (88 char line length) - run `black .`
- Linting: Pylint with `.pylintrc` config - run `pylint --rcfile=.pylintrc`
- CI: GitHub Actions workflow `.github/workflows/pylint.yml` runs on all pushes

### Common Tasks

**Generate new depth datasets:**
```bash
cd YOLOv8_Depth_Learning
python generate_depth_datasets.py  # Creates dataset_depth_* for all models
```

**Train all depth models:**
```bash
python train_all_depth_models.py  # Batch trains all depth datasets
```

**Compare results:**
```bash
python compare_models.py --type depth  # Generates comparison CSV/plots
```

**Add new depth estimator:**
1. Create class in `depth_vision/estimators/` inheriting `BaseDepthEstimator`
2. Register in `depth_vision/factory.py::_estimators`
3. Add config to `DEPTH_MODELS` dict in generation scripts
4. Run full pipeline (generate → train → compare)

## Model Configuration Reference

### Depth Models (depth-only, 1-channel grayscale)
- `midas_DPT_Large`, `midas_DPT_Hybrid`, `midas_small`
- `depth_anything_large`, `depth_anything_base`, `depth_anything_small`
- `zoedepth_NK` (indoor+outdoor), `zoedepth_N` (indoor), `zoedepth_K` (outdoor)
- `marigold_lcm` (fast diffusion), `marigold_default` (accurate but slow)

### RGBD Models (4-channel RGB+Depth)
- Same naming as depth: `dataset_rgbd_<model_name>`
- Generated via `generate_rgbd_datasets.py` using `create_rgbd_image()`
- Requires `channels: 4` in data.yaml for YOLOv8

## Project-Specific Notes

- Comments and documentation may be in Polish (e.g., "worki" = sandbags)
- Single class detection: `nc: 1, names: [sandbag]`
- GPU training default: `device=0`, override with `device='cpu'` for testing
- Results tracking: Training runs auto-named with timestamps in `runs/segment/`
- Comparison scripts auto-discover runs by naming pattern (`depth_*` vs RGB)
