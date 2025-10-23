"""
Factory for creating depth estimators.
Provides a simple interface to instantiate different depth estimation models.
"""

from typing import Dict, Type, Optional
from .base import BaseDepthEstimator
from .estimators.midas import MiDaSDepthEstimator


class DepthEstimatorFactory:
    """
    Factory class for creating depth estimators.

    Example:
        # Create a MiDaS estimator
        estimator = DepthEstimatorFactory.create("midas", model_type="DPT_Hybrid")

        # Use it
        depth = estimator.estimate(image)
    """

    _estimators: Dict[str, Type[BaseDepthEstimator]] = {
        "midas": MiDaSDepthEstimator,
        # Add more estimators here as they are implemented:
        # "zoe": ZoeDepthEstimator,
        # "depth_anything": DepthAnythingEstimator,
        # "marigold": MarigoldEstimator,
    }

    @classmethod
    def create(cls, estimator_type: str, **kwargs) -> BaseDepthEstimator:
        """
        Create a depth estimator instance.

        Args:
            estimator_type: Type of estimator ("midas", "zoe", etc.)
            **kwargs: Configuration parameters for the specific estimator

        Returns:
            Initialized depth estimator instance

        Raises:
            ValueError: If estimator_type is not supported

        Example:
            estimator = DepthEstimatorFactory.create(
                "midas",
                model_type="DPT_Large"
            )
        """
        estimator_type_lower = estimator_type.lower()

        if estimator_type_lower not in cls._estimators:
            available = ", ".join(cls._estimators.keys())
            raise ValueError(
                f"Unsupported estimator type: '{estimator_type}'. "
                f"Available types: {available}"
            )

        estimator_class = cls._estimators[estimator_type_lower]
        return estimator_class(**kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, Type[BaseDepthEstimator]]:
        """
        Get dictionary of all available estimators.

        Returns:
            Dictionary mapping estimator names to their classes
        """
        return cls._estimators.copy()

    @classmethod
    def register(cls, name: str, estimator_class: Type[BaseDepthEstimator]) -> None:
        """
        Register a new depth estimator type.

        Args:
            name: Name to register the estimator under
            estimator_class: The estimator class to register

        Example:
            DepthEstimatorFactory.register("custom", CustomDepthEstimator)
        """
        if not issubclass(estimator_class, BaseDepthEstimator):
            raise TypeError(
                f"estimator_class must be a subclass of BaseDepthEstimator, "
                f"got {estimator_class}"
            )
        cls._estimators[name.lower()] = estimator_class


# Convenience function for backward compatibility
def create_depth_estimator(
    estimator_type: str = "midas", **kwargs
) -> BaseDepthEstimator:
    """
    Convenience function to create a depth estimator.

    Args:
        estimator_type: Type of estimator to create
        **kwargs: Configuration parameters

    Returns:
        Initialized depth estimator

    Example:
        estimator = create_depth_estimator("midas", model_type="DPT_Hybrid")
    """
    return DepthEstimatorFactory.create(estimator_type, **kwargs)
