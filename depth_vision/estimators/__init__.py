"""
Depth estimators package.
Contains implementations of various depth estimation models.
"""

from .midas import MiDaSDepthEstimator
from .depth_anything import DepthAnythingEstimator
from .zoedepth import ZoeDepthEstimator
from .marigold import MarigoldDepthEstimator

__all__ = [
    "MiDaSDepthEstimator",
    "DepthAnythingEstimator",
    "ZoeDepthEstimator",
    "MarigoldDepthEstimator",
]
