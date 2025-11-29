"""
Depth Pro estimator implementation.
Uses Apple's ML Research Depth Pro model for high-quality depth estimation.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path
from ..base import BaseDepthEstimator


class DepthProEstimator(BaseDepthEstimator):
    """
    Apple Depth Pro depth estimator.

    Depth Pro is a foundation model for zero-shot metric monocular depth estimation.
    It produces high-resolution depth maps with sharp boundaries and fine-grained details.

    Features:
    - Zero-shot metric depth estimation
    - High resolution output (up to 2.25 megapixels)
    - Sharp boundary delineation
    - Fast inference (~0.3s on GPU)

    Example usage:
        estimator = DepthEstimatorFactory.create("depth_pro")
        depth = estimator.estimate(image)
    """

    def __init__(self, **kwargs):
        """
        Initialize Depth Pro estimator.

        Args:
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.load_model()

    def _get_checkpoint_path(self) -> str:
        """
        Get the path to the Depth Pro checkpoint, downloading if necessary.

        Returns:
            Path to the checkpoint file
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download Depth Pro. Install with:\n"
                "pip install huggingface_hub"
            ) from exc

        print(
            "Downloading Depth Pro checkpoint from HuggingFace (this may take a while on first run)..."
        )

        # Download from HuggingFace and cache it
        checkpoint_path = hf_hub_download(
            repo_id="apple/DepthPro", filename="depth_pro.pt", repo_type="model"
        )

        print(f"✓ Checkpoint cached at: {checkpoint_path}")
        return checkpoint_path

    def load_model(self) -> None:
        """Load Depth Pro model from HuggingFace."""
        try:
            from depth_pro import create_model_and_transforms
            from depth_pro.depth_pro import DepthProConfig
        except ImportError as exc:
            raise ImportError(
                "Depth Pro is not installed. Please install it with:\n"
                "pip install git+https://github.com/apple/ml-depth-pro.git"
            ) from exc

        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Depth Pro model on {self.device}...")

        # Get checkpoint path (downloads and caches if needed)
        checkpoint_path = self._get_checkpoint_path()

        # Create config with the checkpoint path
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=checkpoint_path,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        # Load model and preprocessing transform
        # Use half precision if CUDA is available for faster inference
        precision = torch.half if torch.cuda.is_available() else torch.float32
        self.model, self.transform = create_model_and_transforms(
            config=config, device=self.device, precision=precision
        )
        self.model.eval()

        print("✓ Depth Pro model loaded successfully!")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for Depth Pro.

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed tensor ready for Depth Pro
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Depth Pro's transform (already includes ToTensor, normalization, etc.)
        input_tensor = self.transform(img_rgb)

        return input_tensor

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using Depth Pro.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Depth map as numpy array (higher values = closer objects, matching other estimators)
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Predict
        with torch.no_grad():
            prediction = self.model.infer(input_tensor)

        # Extract depth from prediction
        # Depth Pro returns a dict with 'depth' and optionally 'focallength_px'
        if isinstance(prediction, dict):
            depth = prediction["depth"]
        else:
            depth = prediction

        # Convert to numpy and squeeze batch dimension
        depth_map = depth.squeeze().cpu().numpy()

        # Resize to original image size if needed
        if depth_map.shape != image.shape[:2]:
            depth_map = cv2.resize(
                depth_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Invert to match convention: higher values = closer objects
        # Depth Pro outputs metric depth (higher = farther), we invert it
        depth_map = np.max(depth_map) - depth_map

        return depth_map

    def get_model_info(self) -> Dict[str, Any]:
        """Get Depth Pro model information."""
        return {
            "name": "Depth Pro",
            "variant": "foundation",
            "framework": "PyTorch",
            "device": str(self.device),
            "properties": {
                "size": "~2GB",
                "accuracy": "highest",
                "speed": "fast",
                "metric_depth": True,
                "max_resolution": "2.25MP",
            },
            "repository": "apple/ml-depth-pro",
            "paper": "https://arxiv.org/abs/2410.02073",
        }
