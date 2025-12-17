"""Image utilities shared across RGBD generation and inference pipelines."""

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from depth_vision.factory import DepthEstimatorFactory
from depth_vision.utils import normalize_depth


def create_depth_estimator(model_name: str, **config: Any) -> Any:
    """Create and return a depth estimator instance."""
    return DepthEstimatorFactory.create(model_name, **config)


def create_rgbd_image(rgb_image: np.ndarray, depth_estimator: Any) -> np.ndarray:
    """
    Create a 4-channel RGBD image from an RGB input.

    Args:
        rgb_image: BGR image (OpenCV format).
        depth_estimator: Initialized depth estimator with an .estimate method.

    Returns:
        4-channel BGRD image (uint8).
    """
    depth_map = depth_estimator.estimate(rgb_image)
    depth_normalized = normalize_depth(depth_map)
    rgbd = np.dstack([rgb_image, depth_normalized])
    return rgbd.astype(np.uint8)
