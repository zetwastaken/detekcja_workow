"""
Centralized depth model configuration.

Maps depth model names (as used in dataset names and run directories)
to the corresponding depth estimator types and configuration parameters.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


# Mapping from depth model name to (estimator_type, config_dict)
# This mirrors the DEPTH_MODELS in generate_rgbd_datasets.py
DEPTH_MODEL_CONFIGS: Dict[str, Tuple[str, Dict]] = {
    "midas_DPT_Large": ("midas", {"model_type": "DPT_Large"}),
    "midas_DPT_Hybrid": ("midas", {"model_type": "DPT_Hybrid"}),
    "midas_small": ("midas", {"model_type": "MiDaS_small"}),
    "depth_anything_large": ("depth_anything", {"model_size": "large"}),
    "depth_anything_base": ("depth_anything", {"model_size": "base"}),
    "depth_anything_small": ("depth_anything", {"model_size": "small"}),
    "zoedepth_NK": ("zoedepth", {"model_type": "NK"}),
    "zoedepth_N": ("zoedepth", {"model_type": "N"}),
    "zoedepth_K": ("zoedepth", {"model_type": "K"}),
    "marigold_lcm": ("marigold", {"variant": "lcm"}),
    "marigold_default": ("marigold", {"variant": "default"}),
}

# Default fallback configuration
DEFAULT_DEPTH_CONFIG = ("depth_anything", {"model_size": "large"})


def get_depth_config(model_name: str) -> Tuple[str, Dict]:
    """
    Get estimator type and config for a given depth model name.

    Args:
        model_name: Name of the depth model (e.g., "depth_anything_large", "zoedepth_K")

    Returns:
        Tuple of (estimator_type, config_dict)
    """
    return DEPTH_MODEL_CONFIGS.get(model_name, DEFAULT_DEPTH_CONFIG)


def extract_depth_model_from_dataset_path(data_path: str) -> Optional[str]:
    """
    Extract depth model name from a dataset path.

    Args:
        data_path: Path to the dataset's data.yaml file
                   (e.g., ".../datasets/dataset_rgbd_depth_anything_large/data.yaml")

    Returns:
        Depth model name (e.g., "depth_anything_large") or None if not found
    """
    path = Path(data_path)
    dataset_name = path.parent.name  # e.g., "dataset_rgbd_depth_anything_large"

    # Try to extract model name from dataset directory name
    # Patterns: dataset_rgbd_<model_name> or dataset_depth_<model_name>
    if dataset_name.startswith("dataset_rgbd_"):
        return dataset_name[13:]  # Remove "dataset_rgbd_"
    elif dataset_name.startswith("dataset_depth_"):
        return dataset_name[14:]  # Remove "dataset_depth_"

    return None


def extract_depth_model_from_run_name(run_name: str) -> Optional[str]:
    """
    Extract depth model name from a training run directory name.

    Args:
        run_name: Name of the run directory
                  (e.g., "rgbd_depth_anything_large_20251211_153418")

    Returns:
        Depth model name (e.g., "depth_anything_large") or None if not found
    """
    # Patterns: rgbd_<model_name>_<timestamp> or depth_<model_name>_<timestamp>
    # Remove timestamp suffix (format: YYYYMMDD_HHMMSS)
    timestamp_pattern = r"_\d{8}_\d{6}$"
    run_name_no_timestamp = re.sub(timestamp_pattern, "", run_name)

    if run_name_no_timestamp.startswith("rgbd_"):
        return run_name_no_timestamp[5:]  # Remove "rgbd_"
    elif run_name_no_timestamp.startswith("depth_"):
        return run_name_no_timestamp[6:]  # Remove "depth_"

    return None


def extract_depth_model_from_weights(weights_path: Path) -> Tuple[str, Dict]:
    """
    Extract depth model configuration from a trained model's weights path.

    This function reads the args.yaml file from the training run and
    extracts the depth model information from the dataset path.

    Args:
        weights_path: Path to the model weights file (e.g., ".../weights/best.pt")

    Returns:
        Tuple of (estimator_type, config_dict)
    """
    model_name = get_depth_model_name_from_weights(weights_path)

    # Get configuration for the detected model
    if model_name and model_name in DEPTH_MODEL_CONFIGS:
        return get_depth_config(model_name)
    elif model_name:
        print(f"  Warning: Unknown depth model '{model_name}', using default")
        return DEFAULT_DEPTH_CONFIG
    else:
        print(
            "  Warning: Could not detect depth model, using default (depth_anything large)"
        )
        return DEFAULT_DEPTH_CONFIG


def get_depth_model_name_from_weights(weights_path: Path) -> Optional[str]:
    """
    Extract the depth model name from a trained model's weights path.

    Args:
        weights_path: Path to the model weights file (e.g., ".../weights/best.pt")

    Returns:
        Depth model name (e.g., "depth_anything_large") or None if not found
    """
    # Get the run directory (parent of 'weights' folder)
    run_dir = weights_path.parent.parent
    args_file = run_dir / "args.yaml"

    model_name = None

    # Method 1: Try to extract from args.yaml (most reliable)
    if args_file.exists():
        try:
            with open(args_file, "r") as f:
                args = yaml.safe_load(f)

            if "data" in args:
                model_name = extract_depth_model_from_dataset_path(args["data"])
                if model_name:
                    print(f"  Detected depth model from dataset: {model_name}")
        except Exception as e:
            print(f"  Warning: Could not read args.yaml: {e}")

    # Method 2: Fallback to parsing run directory name
    if not model_name:
        run_name = run_dir.name
        model_name = extract_depth_model_from_run_name(run_name)
        if model_name:
            print(f"  Detected depth model from run name: {model_name}")

    return model_name


def is_rgbd_model(weights_path: Path) -> bool:
    """
    Check if a model was trained on RGBD data (4-channel input).

    Args:
        weights_path: Path to the model weights file

    Returns:
        True if the model was trained on RGBD data, False otherwise
    """
    run_dir = weights_path.parent.parent
    run_name = run_dir.name.lower()

    # Check run name for RGBD or depth indicators
    if "rgbd_" in run_name or "depth_" in run_name:
        return True

    # Check args.yaml for dataset path
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        try:
            with open(args_file, "r") as f:
                args = yaml.safe_load(f)

            if "data" in args:
                data_path = args["data"].lower()
                if "rgbd" in data_path or "depth" in data_path:
                    return True
        except Exception:
            pass

    return False
