"""
Depth Anything estimator implementation.
Uses the Depth Anything V2 model from HuggingFace.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ..base import BaseDepthEstimator


class DepthAnythingEstimator(BaseDepthEstimator):
    """
    Depth Anything V2 depth estimator.

    Supports multiple model variants:
    - small: Fastest, good for real-time
    - base: Balanced accuracy and speed
    - large: Highest accuracy

    Example usage:
        estimator = DepthEstimatorFactory.create(
            "depth_anything",
            model_size="small"  # small, base, large
        )
    """

    SUPPORTED_MODELS = {
        "small": {
            "repo": "depth-anything/Depth-Anything-V2-Small-hf",
            "size": "~100MB",
            "accuracy": "good",
            "speed": "fastest",
        },
        "base": {
            "repo": "depth-anything/Depth-Anything-V2-Base-hf",
            "size": "~400MB",
            "accuracy": "high",
            "speed": "medium",
        },
        "large": {
            "repo": "depth-anything/Depth-Anything-V2-Large-hf",
            "size": "~1.3GB",
            "accuracy": "highest",
            "speed": "slowest",
        },
    }

    def __init__(self, model_size: str = "small", **kwargs):
        """
        Initialize Depth Anything estimator.

        Args:
            model_size: Model size variant (small, base, large)
            **kwargs: Additional configuration
        """
        super().__init__(model_size=model_size, **kwargs)
        self.model_size = model_size

        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model size: {model_size}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.load_model()

    def load_model(self) -> None:
        """Load Depth Anything model from HuggingFace."""
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model repository
        model_repo = self.SUPPORTED_MODELS[self.model_size]["repo"]

        # Load model and processor
        print(
            f"Loading Depth Anything V2 ({self.model_size}) model on {self.device}..."
        )
        self.processor = AutoImageProcessor.from_pretrained(model_repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_repo)
        self.model.to(self.device)
        self.model.eval()

        print("✓ Depth Anything model loaded successfully!")

    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for Depth Anything.

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed inputs ready for the model
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the processor to prepare inputs
        inputs = self.processor(images=img_rgb, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using Depth Anything.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Depth map as numpy array (higher values = closer objects)
        """
        # Preprocess
        inputs = self.preprocess(image)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess - extract predicted depth and resize to original size
        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()

        return depth_map

    def get_model_info(self) -> Dict[str, Any]:
        """Get Depth Anything model information."""
        return {
            "name": "Depth Anything V2",
            "variant": self.model_size,
            "framework": "PyTorch + HuggingFace",
            "device": str(self.device),
            "properties": self.SUPPORTED_MODELS[self.model_size],
            "repository": self.SUPPORTED_MODELS[self.model_size]["repo"],
        }
