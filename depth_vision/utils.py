"""
Utility functions for depth map processing and visualization.
"""

import cv2
import numpy as np


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize depth map to 0-255 range.

    Args:
        depth_map: Raw depth map

    Returns:
        Normalized depth map as uint8 (0-255 range)
    """
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
    return normalized.astype(np.uint8)


def visualize_depth(
    depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
) -> np.ndarray:
    """
    Apply colormap to depth map for visualization.

    Args:
        depth_map: Raw depth map
        colormap: cv2.COLORMAP_* (INFERNO, VIRIDIS, PLASMA, JET, etc.)

    Returns:
        Colored depth map (BGR format)

    Example:
        depth_map = estimator.estimate(image)
        colored = visualize_depth(depth_map, colormap=cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("depth_colored.png", colored)
    """
    normalized = normalize_depth(depth_map)
    colored = cv2.applyColorMap(normalized, colormap)
    return colored
