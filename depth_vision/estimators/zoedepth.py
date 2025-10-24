"""
ZoeDepth estimator implementation.
ZoeDepth is a high-quality metric depth estimation model.
ZoeDepth is inverted by default to match other estimators (closer = higher values).
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any
import sys
import os

from ..base import BaseDepthEstimator


class ZoeDepthEstimator(BaseDepthEstimator):
    """
    ZoeDepth depth estimator.

    Supports multiple model variants:
    - NK (NYU-KITTI): Indoor/outdoor general purpose
    - N (NYU): Indoor scenes
    - K (KITTI): Outdoor/driving scenes

    Example usage:
        estimator = DepthEstimatorFactory.create(
            "zoedepth",
            model_type="NK"  # NK, N, K
        )
    """

    SUPPORTED_MODELS = {
        "NK": {
            "repo": "isl-org/ZoeDepth",
            "model_name": "ZoeD_NK",
            "size": "~350MB",
            "accuracy": "highest",
            "speed": "medium",
            "description": "General purpose (indoor + outdoor)",
        },
        "N": {
            "repo": "isl-org/ZoeDepth",
            "model_name": "ZoeD_N",
            "size": "~350MB",
            "accuracy": "high",
            "speed": "medium",
            "description": "Indoor scenes (NYU dataset)",
        },
        "K": {
            "repo": "isl-org/ZoeDepth",
            "model_name": "ZoeD_K",
            "size": "~350MB",
            "accuracy": "high",
            "speed": "medium",
            "description": "Outdoor/driving scenes (KITTI dataset)",
        },
    }

    def __init__(self, model_type: str = "NK", **kwargs):
        """
        Initialize ZoeDepth estimator.

        Args:
            model_type: Model variant to use (NK, N, K)
            **kwargs: Additional configuration
        """
        super().__init__(model_type=model_type, **kwargs)
        self.model_type = model_type
        self.model: Any = None  # Type annotation for the model

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.load_model()

    def load_model(self) -> None:
        """Load ZoeDepth model from torch hub."""
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model name
        model_name = self.SUPPORTED_MODELS[self.model_type]["model_name"]
        repo = self.SUPPORTED_MODELS[self.model_type]["repo"]

        # Load model
        print(f"Loading ZoeDepth {self.model_type} model on {self.device}...")

        # Monkey-patch the load_state_dict method to handle PyTorch compatibility
        # This fixes the relative_position_index buffer issue with newer PyTorch versions
        import types

        def patched_load_state_dict(self, state_dict, assign=False):
            # Filter out relative_position_index keys that are now buffers
            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if "relative_position_index" not in k
            }
            # Call the original method with filtered state dict and strict=False
            return self._original_load_state_dict(
                filtered_state_dict, strict=False, assign=assign
            )

        # Store original and apply patch
        torch.nn.Module._original_load_state_dict = torch.nn.Module.load_state_dict
        torch.nn.Module.load_state_dict = patched_load_state_dict

        try:
            self.model = torch.hub.load(
                repo, model_name, pretrained=True, trust_repo=True
            )
        finally:
            # Restore original method
            torch.nn.Module.load_state_dict = torch.nn.Module._original_load_state_dict
            delattr(torch.nn.Module, "_original_load_state_dict")

        # Fix BEiT backbone compatibility issue with newer timm versions
        # The attribute 'drop_path' was renamed to 'drop_path1'
        self._fix_beit_compatibility()

        self.model.to(self.device)
        self.model.eval()

        print("✓ ZoeDepth model loaded successfully!")

    def _fix_beit_compatibility(self) -> None:
        """Fix compatibility issues with BEiT backbone in newer timm versions."""
        # Recursively fix all Block modules in the model
        for _name, module in self.model.named_modules():
            if module.__class__.__name__ == "Block":
                # If the module has drop_path1 but not drop_path, create an alias
                if hasattr(module, "drop_path1") and not hasattr(module, "drop_path"):
                    module.drop_path = module.drop_path1
                # Handle drop_path2 as well if it exists
                if hasattr(module, "drop_path2") and not hasattr(module, "drop_path"):
                    module.drop_path = module.drop_path2

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ZoeDepth.

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed image in RGB format
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return img_rgb

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using ZoeDepth.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Depth map as numpy array (inverted metric depth - higher values = closer)
        """
        # Preprocess
        img_rgb = self.preprocess(image)

        # Predict
        with torch.no_grad():
            # ZoeDepth expects RGB numpy array and returns metric depth
            depth_map = self.model.infer_pil(img_rgb)

        # Invert depth map: ZoeDepth returns metric depth where higher = farther
        # We invert it so higher values = closer (consistent with other estimators)
        # Use max value to avoid division by zero
        max_depth = np.max(depth_map)
        if max_depth > 0:
            depth_map = max_depth - depth_map

        return depth_map

    def get_model_info(self) -> Dict[str, Any]:
        """Get ZoeDepth model information."""
        return {
            "name": "ZoeDepth",
            "variant": self.model_type,
            "framework": "PyTorch",
            "device": str(self.device),
            "properties": self.SUPPORTED_MODELS[self.model_type],
            "repository": self.SUPPORTED_MODELS[self.model_type]["repo"],
            "output_type": "metric depth (meters)",
        }
