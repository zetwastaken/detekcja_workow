"""
Depth Anything estimator implementation (placeholder).
This is a template for adding the Depth Anything model.
"""

import numpy as np
from typing import Dict, Any

from ..base import BaseDepthEstimator


class DepthAnythingEstimator(BaseDepthEstimator):
    """
    Depth Anything depth estimator.

    NOTE: This is a placeholder implementation showing how to add new models.
    To implement:
    1. Install required dependencies (transformers, etc.)
    2. Implement load_model() to load Depth Anything
    3. Implement estimate() for inference
    4. Add to factory.py

    Example future usage:
        estimator = DepthEstimatorFactory.create(
            "depth_anything",
            model_size="small"  # small, base, large
        )
    """

    def __init__(self, model_size: str = "small", **kwargs):
        """
        Initialize Depth Anything estimator.

        Args:
            model_size: Model size variant (small, base, large)
            **kwargs: Additional configuration
        """
        super().__init__(model_size=model_size, **kwargs)
        self.model_size = model_size
        # self.load_model()  # Uncomment when implemented

    def load_model(self) -> None:
        """Load Depth Anything model."""
        raise NotImplementedError(
            "Depth Anything is not yet implemented. "
            "To add it:\n"
            "1. pip install transformers torch\n"
            "2. Load model from HuggingFace\n"
            "3. Implement estimate() method"
        )

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Generate depth map using Depth Anything."""
        raise NotImplementedError("Depth Anything not yet implemented")

    def get_model_info(self) -> Dict[str, Any]:
        """Get Depth Anything model information."""
        return {
            "name": "Depth Anything",
            "variant": self.model_size,
            "status": "not_implemented",
            "repository": "LiheYoung/Depth-Anything",
        }
