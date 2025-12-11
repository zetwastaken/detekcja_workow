"""
YOLOv8 inference helper for depth-trained sandbag models.

Supports running models trained on depth colormap tiles (3-channel) and RGBD
models (4-channel) with optional on-the-fly depth generation from RGB inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure repo root is on sys.path so depth_vision imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_vision.factory import DepthEstimatorFactory
from depth_vision.utils import normalize_depth, visualize_depth
from generate_depth_datasets import DEPTH_MODELS

# Paths
ROOT_DIR = PROJECT_ROOT  # detekcja_workow/
DATASETS_DIR = ROOT_DIR / "datasets"
RGBD_DATASET_DIR = DATASETS_DIR / "dataset_rgbd"
RUNS_DIR = ROOT_DIR / "runs" / "segment"
DEPTH_TILES_DIR = ROOT_DIR / "depth_tiles"
PREDICTIONS_DIR = ROOT_DIR / "runs" / "prediction"

# Tiling parameters (match training)
TILE_SIZE = 640
OVERLAP = 80

# Defaults (now biased to RGBD workflow)
DEFAULT_MODEL_KEY = "depth_anything_large"
DEFAULT_RUN_TYPE = "rgbd"  # "depth" (colored depth map) or "rgbd" (4-channel)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLORMAPS: Dict[str, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "jet": cv2.COLORMAP_JET,
}

# VS Code / no-CLI presets: edit these and run the file directly.
RUN_WITH_PRESET_WHEN_NO_ARGS = True
PRESET_SOURCE = None  # If None, falls back to defaults per run_type/model
PRESET_SETTINGS = {
    "model_key": DEFAULT_MODEL_KEY,
    "run_type": DEFAULT_RUN_TYPE,
    "run_prefix": None,
    "weights": None,
    "source": (
        [PRESET_SOURCE] if PRESET_SOURCE else None
    ),  # can be folder, glob, or list of files
    "from_rgb": False,  # set True to generate depth from RGB
    "tile": None,  # None -> auto default (depth=True, rgbd without from_rgb=False)
    "rgbd_split": "valid",
    "limit": None,
    "conf": 0.25,
    "iou": 0.7,
    "colormap": "inferno",
    "name": None,
}


def tile_image(
    image: np.ndarray,
    base_name: str,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
    ext: str = ".jpg",
) -> List[Tuple[str, np.ndarray]]:
    """
    Tile an image into overlapping patches.

    Returns a list of (filename, tile) tuples without writing to disk.
    """
    stride = tile_size - overlap
    height, width = image.shape[:2]
    tiles: List[Tuple[str, np.ndarray]] = []

    x_starts = []
    x = 0
    while x <= width - tile_size:
        x_starts.append(x)
        x += stride
    if x_starts and x_starts[-1] < width - tile_size:
        x_starts.append(width - tile_size)
    elif not x_starts and width >= tile_size:
        x_starts.append(0)

    y_starts = []
    y = 0
    while y <= height - tile_size:
        y_starts.append(y)
        y += stride
    if y_starts and y_starts[-1] < height - tile_size:
        y_starts.append(height - tile_size)
    elif not y_starts and height >= tile_size:
        y_starts.append(0)

    suffix = ext if ext.startswith(".") else f".{ext}"
    for i, y_start in enumerate(y_starts):
        for j, x_start in enumerate(x_starts):
            tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tile_filename = f"{base_name}_R{i:03d}_C{j:03d}{suffix}"
            tiles.append((tile_filename, tile))

    return tiles


def collect_image_paths(
    sources: Sequence[str] | None, default_dir: Path | None
) -> List[Path]:
    """
    Expand user-provided sources into a flat list of image paths.
    Supports files, folders, and globs.
    """
    paths: List[Path] = []

    if sources:
        for src in sources:
            src_str = str(src)
            if "*" in src_str or "?" in src_str:
                matches = sorted(Path().glob(src_str))
                paths.extend([m.resolve() for m in matches if m.is_file()])
                continue

            path = Path(src_str).expanduser()
            if path.is_dir():
                for file in sorted(path.iterdir()):
                    if file.suffix.lower() in IMAGE_EXTS:
                        paths.append(file.resolve())
            elif path.is_file():
                paths.append(path.resolve())
    elif default_dir and default_dir.exists():
        for file in sorted(default_dir.iterdir()):
            if file.suffix.lower() in IMAGE_EXTS:
                paths.append(file.resolve())

    if not paths:
        raise FileNotFoundError(
            f"No input images found. Provide --source or ensure {default_dir} exists."
        )

    return paths


def resolve_weights(
    model_key: str,
    run_type: str,
    run_prefix: str | None,
    weights_override: str | None,
) -> Path:
    """Pick weights file based on override, run prefix, or latest matching run."""
    if weights_override:
        weights_path = Path(weights_override).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        return weights_path

    if run_prefix:
        glob_pattern = f"{run_prefix}_*/weights/best.pt"
    else:
        prefix = f"{run_type}_{model_key}".rstrip("_")
        glob_pattern = f"{prefix}_*/weights/best.pt"

    candidates = sorted(
        RUNS_DIR.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not candidates:
        raise FileNotFoundError(
            f"No weights found under {RUNS_DIR} matching '{glob_pattern}'. "
            "Pass --weights or --run-prefix explicitly."
        )

    return candidates[0]


def build_depth_estimator(model_key: str):
    """Create a depth estimator matching the training dataset config."""
    if model_key not in DEPTH_MODELS:
        available = ", ".join(sorted(DEPTH_MODELS.keys()))
        raise ValueError(
            f"Unknown depth model '{model_key}'. Available options: {available}"
        )

    cfg = DEPTH_MODELS[model_key]
    estimator_type = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return DepthEstimatorFactory.create(estimator_type, **kwargs)


def generate_depth_inputs(
    source_images: Iterable[Path],
    estimator,
    mode: str,
    tile: bool,
    output_dir: Path,
    colormap: int,
) -> List[Path]:
    """Convert RGB sources into depth-map or RGBD inputs expected by the model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for img_path in source_images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  Skipping unreadable image: {img_path}")
            continue

        depth_map = estimator.estimate(image)

        if mode == "rgbd":
            depth_norm = normalize_depth(depth_map)
            depth_input = np.dstack([image, depth_norm]).astype(np.uint8)
            ext = ".tiff"
        else:
            depth_input = visualize_depth(depth_map, colormap=colormap)
            ext = ".jpg"

        if tile:
            tiles = tile_image(depth_input, img_path.stem, TILE_SIZE, OVERLAP, ext)
            for filename, tile_img in tiles:
                out_path = output_dir / filename
                cv2.imwrite(str(out_path), tile_img)
                saved.append(out_path)
        else:
            out_path = output_dir / f"{img_path.stem}{ext}"
            cv2.imwrite(str(out_path), depth_input)
            saved.append(out_path)

    if not saved:
        raise RuntimeError("No depth inputs were generated.")

    return saved


def load_rgbd_images(image_paths: Iterable[Path]) -> List[np.ndarray]:
    """
    Load RGBD images preserving all 4 channels (BGRD).
    """
    loaded: List[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️  Skipping unreadable image: {path}")
            continue

        if img.ndim == 2:
            print(f"⚠️  Skipping grayscale image (expected 4 channels): {path}")
            continue

        if img.shape[2] < 4:
            print(f"⚠️  Skipping 3-channel image (expected 4 channels): {path}")
            continue

        if img.shape[2] > 4:
            img = img[:, :, :4]

        # Preserve original name for Ultralytics saving logic
        try:
            img.filename = str(path)
        except Exception:
            pass

        loaded.append(img)

    if not loaded:
        raise RuntimeError("No valid 4-channel RGBD images were loaded.")

    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference for depth-trained sandbag models."
    )
    parser.add_argument(
        "--model-key",
        default=DEFAULT_MODEL_KEY,
        help="Depth dataset key (matches depth_tiles/<key> and run prefix).",
    )
    parser.add_argument(
        "--run-type",
        choices=["depth", "rgbd"],
        default=DEFAULT_RUN_TYPE,
        help="Use depth-map (3-channel) or RGBD (4-channel) model weights.",
    )
    parser.add_argument(
        "--run-prefix",
        help="Override run prefix used to pick the newest weights (e.g., depth_midas_DPT_Large).",
    )
    parser.add_argument(
        "--weights",
        help="Explicit path to a weights file. Overrides run discovery.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        help="Images, folder, or glob. Required with --from-rgb; otherwise defaults to depth_tiles/<model-key>.",
    )
    parser.add_argument(
        "--from-rgb",
        action="store_true",
        help="Generate depth inputs from RGB sources using the matching estimator.",
    )
    parser.add_argument(
        "--rgbd-split",
        choices=["train", "valid"],
        default="valid",
        help="Default dataset split when run_type=rgbd and no source provided.",
    )
    parser.add_argument(
        "--no-tile",
        dest="tile",
        action="store_false",
        help="Run on full frames instead of 640x640 depth tiles.",
    )
    parser.add_argument(
        "--tile",
        dest="tile",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(tile=None)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on number of images processed.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--colormap",
        choices=sorted(COLORMAPS.keys()),
        default="inferno",
        help="Colormap for depth visualization (depth mode only).",
    )
    parser.add_argument(
        "--name",
        help="Custom prediction folder name under runs/segment/.",
    )
    return parser.parse_args()


def _namespace_from_settings(settings: Dict[str, Any]) -> argparse.Namespace:
    """Create argparse-like namespace from preset settings."""
    # Ensure any Path values become strings for predict()
    normalized = settings.copy()
    if normalized.get("source") is not None:
        normalized["source"] = [
            str(p) if isinstance(p, Path) else p for p in normalized["source"]
        ]
    return argparse.Namespace(**normalized)


def main(settings: Dict[str, Any] | None = None) -> None:
    # If settings provided or no CLI args, use presets for IDE runs; otherwise parse CLI.
    if settings is not None:
        args = _namespace_from_settings(settings)
    elif RUN_WITH_PRESET_WHEN_NO_ARGS and not any(
        a.startswith("-") for a in sys.argv[1:]
    ):
        args = _namespace_from_settings(PRESET_SETTINGS)
        print("Using preset settings (no CLI args detected).")
    else:
        args = parse_args()

    # Choose tile default if unset (rgbd -> no tiling by default to keep filenames)
    if args.tile is None:
        args.tile = False if args.run_type == "rgbd" else True

    weights_path = resolve_weights(
        model_key=args.model_key,
        run_type=args.run_type,
        run_prefix=args.run_prefix,
        weights_override=args.weights,
    )

    # Naming: include trained run folder and timestamp when not provided
    weights_run_folder = weights_path.parent.parent.name
    default_name = (
        f"{weights_run_folder}_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    predict_name = args.name or default_name

    if args.from_rgb and not args.source:
        raise ValueError("Provide --source when generating depth from RGB.")

    def default_source_for(run_type: str, model_key: str, split: str) -> Path | None:
        if run_type == "rgbd":
            candidate = RGBD_DATASET_DIR / split / "images"
            return candidate if candidate.exists() else None
        if not args.from_rgb:
            candidate = DEPTH_TILES_DIR / model_key
            return candidate if candidate.exists() else None
        return None

    default_source_dir = default_source_for(
        args.run_type, args.model_key, getattr(args, "rgbd_split", "valid")
    )
    source_images = collect_image_paths(args.source, default_source_dir)
    print(
        f"Using input images from: {default_source_dir if args.source is None else args.source}"
    )
    if args.limit:
        source_images = source_images[: args.limit]

    if args.from_rgb:
        estimator = build_depth_estimator(args.model_key)
        generated_dir = ROOT_DIR / "output" / "depth_inference_inputs" / predict_name
        source_images = generate_depth_inputs(
            source_images,
            estimator=estimator,
            mode=args.run_type,
            tile=args.tile,
            output_dir=generated_dir,
            colormap=COLORMAPS[args.colormap],
        )

    # Prepare sources respecting 4-channel RGBD inputs
    if args.run_type == "rgbd":
        rgbd_images = load_rgbd_images(source_images)
        if args.tile:
            predict_sources: List[np.ndarray] = []
            for path, img in zip(source_images, rgbd_images):
                tiles = tile_image(
                    img, Path(path).stem, TILE_SIZE, OVERLAP, ext=".tiff"
                )
                for tile_name, tile_img in tiles:
                    try:
                        tile_img.filename = tile_name
                    except Exception:
                        pass
                    predict_sources.append(tile_img)
        else:
            predict_sources = rgbd_images
    else:
        predict_sources = [str(p) for p in source_images]

    model = YOLO(weights_path)
    results = model.predict(
        source=predict_sources,
        project=str(PREDICTIONS_DIR),
        name=predict_name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        imgsz=TILE_SIZE,
        conf=args.conf,
        iou=args.iou,
        line_width=2,
    )

    save_dir = results[0].save_dir if results else PREDICTIONS_DIR
    print(f"\n✓ Using weights: {weights_path}")
    print(f"✓ Saved predictions to: {save_dir}")


if __name__ == "__main__":
    main()
