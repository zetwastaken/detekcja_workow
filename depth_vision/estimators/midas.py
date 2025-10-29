"""
MiDaS depth estimator implementation.
"""

import torch
import cv2
import numpy as np
import ssl
from typing import Dict, Any

from ..base import BaseDepthEstimator

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context


class MiDaSDepthEstimator(BaseDepthEstimator):
    """
    MiDaS-based depth estimator.

    Supports multiple MiDaS model variants:
    - DPT_Large: Highest accuracy
    - DPT_Hybrid: Best balance (recommended)
    - MiDaS_small: Fastest, lower accuracy
    """

    SUPPORTED_MODELS = {
        "DPT_Large": {"size": "~1.3GB", "accuracy": "highest", "speed": "slowest"},
        "DPT_Hybrid": {"size": "~800MB", "accuracy": "high", "speed": "medium"},
        "MiDaS_small": {"size": "~80MB", "accuracy": "medium", "speed": "fastest"},
    }

    def __init__(self, model_type: str = "DPT_Hybrid", **kwargs):
        """
        Initialize MiDaS depth estimator.

        Args:
            model_type: Model variant to use
            **kwargs: Additional configuration
        """
        super().__init__(model_type=model_type, **kwargs)
        self.model_type = model_type

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.load_model()

    def load_model(self) -> None:
        """Load MiDaS model and transforms."""
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"Loading MiDaS {self.model_type} model on {self.device}...")
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load transform
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print("✓ MiDaS model loaded successfully!")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for MiDaS.

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed tensor ready for MiDaS
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform and move to device
        input_batch = self.transform(img_rgb).to(self.device)

        return input_batch

    def postprocess(self, output: torch.Tensor, target_shape: tuple) -> np.ndarray:
        """
        Postprocess MiDaS output.

        Args:
            output: Raw model output
            target_shape: Target shape (H, W) to resize to

        Returns:
            Depth map as numpy array
        """
        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            output.unsqueeze(1),
            size=target_shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()

        return depth_map

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map from image using MiDaS.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Depth map as numpy array (higher values = closer objects)
        """
        # Preprocess
        input_batch = self.preprocess(image)

        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)

        # Postprocess
        depth_map = self.postprocess(prediction, image.shape[:2])

        return depth_map

    def get_model_info(self) -> Dict[str, Any]:
        """Get MiDaS model information."""
        return {
            "name": "MiDaS",
            "variant": self.model_type,
            "framework": "PyTorch",
            "device": str(self.device),
            "properties": self.SUPPORTED_MODELS[self.model_type],
            "repository": "intel-isl/MiDaS",
        }
