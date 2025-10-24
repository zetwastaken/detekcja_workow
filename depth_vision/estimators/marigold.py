"""
Marigold depth estimator implementation.
Uses the Marigold diffusion-based depth estimation model from HuggingFace.
"""

import os
import torch
import cv2
import numpy as np
from typing import Dict, Any
from diffusers import MarigoldDepthPipeline
from ..base import BaseDepthEstimator


class MarigoldDepthEstimator(BaseDepthEstimator):
    """
    Marigold depth estimator using diffusion models.

    Marigold is a diffusion-based depth estimation model that provides
    high-quality depth predictions with fine details.

    Supports multiple model variants:
    - lcm: Fastest, uses Latent Consistency Model for fast inference
    - base: Standard diffusion model (slower but potentially more accurate)

    Example usage:
        estimator = DepthEstimatorFactory.create(
            "marigold",
            model_variant="lcm",  # lcm or base
            num_inference_steps=4  # fewer steps = faster
        )
    """

    SUPPORTED_MODELS = {
        "lcm": {
            "repo": "prs-eth/marigold-lcm-v1-0",
            "size": "~2GB",
            "accuracy": "very high",
            "speed": "fast",
            "default_steps": 4,
        },
        "base": {
            "repo": "prs-eth/marigold-v1-0",
            "size": "~2GB",
            "accuracy": "highest",
            "speed": "slow",
            "default_steps": 50,
        },
    }

    def __init__(
        self,
        model_variant: str = "lcm",
        num_inference_steps: int = None,
        ensemble_size: int = 1,
        **kwargs,
    ):
        """
        Initialize Marigold depth estimator.

        Args:
            model_variant: Model variant (lcm, base)
            num_inference_steps: Number of denoising steps (default depends on variant)
            ensemble_size: Number of predictions to ensemble (1-10, default=1)
            **kwargs: Additional configuration
        """
        super().__init__(
            model_variant=model_variant,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
            **kwargs,
        )
        self.model_variant = model_variant
        self.ensemble_size = max(1, min(10, ensemble_size))

        if model_variant not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model variant: {model_variant}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Set inference steps
        if num_inference_steps is None:
            self.num_inference_steps = self.SUPPORTED_MODELS[model_variant][
                "default_steps"
            ]
        else:
            self.num_inference_steps = num_inference_steps

        self.load_model()

    def load_model(self) -> None:
        """Load Marigold model from HuggingFace."""
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model repository
        model_repo = self.SUPPORTED_MODELS[self.model_variant]["repo"]

        # Load pipeline
        print(f"Loading Marigold ({self.model_variant}) model on {self.device}...")
        print(f"This may take a moment on first run (downloading ~2GB)...")

        # Try to load from cache first, fallback to download if not available
        try:
            self.pipe = MarigoldDepthPipeline.from_pretrained(
                model_repo,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                local_files_only=True,  # Use cached files only
            )
            print("✓ Loaded from cache")
        except (OSError, ValueError) as e:
            print(f"Model not in cache ({e.__class__.__name__}), downloading...")
            self.pipe = MarigoldDepthPipeline.from_pretrained(
                model_repo,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
            )

        self.pipe.to(self.device)

        print("✓ Marigold model loaded successfully!")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Marigold.

        Args:
            image: BGR image from OpenCV

        Returns:
            RGB image as numpy array
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return img_rgb

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using Marigold.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Depth map as numpy array (higher values = closer objects)
        """
        # Preprocess
        img_rgb = self.preprocess(image)

        # Predict using the pipeline
        with torch.no_grad():
            depth_output = self.pipe(
                img_rgb,
                num_inference_steps=self.num_inference_steps,
                ensemble_size=self.ensemble_size,
            )

        # Extract depth prediction
        # Marigold returns a MarigoldDepthOutput object with 'prediction' attribute
        # or it might be a direct tensor/array depending on the version
        if hasattr(depth_output, "prediction"):
            depth_tensor = depth_output.prediction
        elif isinstance(depth_output, torch.Tensor):
            depth_tensor = depth_output
        else:
            # Fallback: assume it's the first element if it's a tuple/list
            depth_tensor = (
                depth_output[0]
                if isinstance(depth_output, (tuple, list))
                else depth_output
            )

        # Convert to numpy and squeeze to remove batch dimension
        if isinstance(depth_tensor, torch.Tensor):
            depth_map = depth_tensor.squeeze().cpu().numpy()
        else:
            depth_map = np.array(depth_tensor).squeeze()

        # Marigold outputs metric depth (smaller values = closer objects)
        # Invert so that closer objects have higher values (consistent with other models)
        # Use max - depth instead of 1/depth to avoid extreme values
        depth_map = depth_map.max() - depth_map

        # Normalize to [0, 1] range
        epsilon = 1e-6
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + epsilon)

        return depth_map

    def get_model_info(self) -> Dict[str, Any]:
        """Get Marigold model information."""
        return {
            "name": "Marigold",
            "variant": self.model_variant,
            "framework": "PyTorch + Diffusers",
            "device": str(self.device),
            "inference_steps": self.num_inference_steps,
            "ensemble_size": self.ensemble_size,
            "properties": self.SUPPORTED_MODELS[self.model_variant],
            "repository": self.SUPPORTED_MODELS[self.model_variant]["repo"],
        }
