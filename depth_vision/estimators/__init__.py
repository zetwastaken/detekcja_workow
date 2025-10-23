"""
Depth estimators package.
Contains implementations of various depth estimation models.
"""

from .midas import MiDaSDepthEstimator
from .depth_anything import DepthAnythingEstimator

__all__ = ["MiDaSDepthEstimator", "DepthAnythingEstimator"]
