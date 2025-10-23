"""
Base classes for depth estimation.
Defines the interface that all depth estimators must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseDepthEstimator(ABC):
    """
    Abstract base class for all depth estimators.

    All depth estimation models must inherit from this class
    and implement the required methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the depth estimator.

        Args:
            **kwargs: Model-specific configuration parameters
        """
        self.config = kwargs
        self.device = None
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the depth estimation model.
        Must be implemented by each specific estimator.
        """
        pass

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map from image.

        Args:
            image: Input image in BGR format (OpenCV convention)

        Returns:
            Depth map as numpy array where higher values typically mean closer objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata (name, version, parameters, etc.)
        """
        pass

    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess image for the model.
        Can be overridden by subclasses if needed.

        Args:
            image: Input image

        Returns:
            Preprocessed input ready for the model
        """
        return image

    def postprocess(self, output: Any) -> np.ndarray:
        """
        Postprocess model output.
        Can be overridden by subclasses if needed.

        Args:
            output: Raw model output

        Returns:
            Processed depth map
        """
        return output
