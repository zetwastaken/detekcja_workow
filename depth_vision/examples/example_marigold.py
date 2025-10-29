"""
Example script demonstrating Marigold depth estimation.

This script shows how to use the Marigold diffusion-based depth estimator.
Marigold provides high-quality depth maps with fine details using a diffusion model.
"""

import cv2
import numpy as np
from pathlib import Path

from depth_vision import create_depth_estimator
from depth_vision.utils import visualize_depth


def main():
    """Run Marigold depth estimation example."""
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Example image path (you'll need to provide your own image)
    # For testing, you can use any image in your data folder
    image_path = Path(__file__).parent.parent.parent / "data" / "test_image.jpg"

    # Check if image exists
    if not image_path.exists():
        print(f"❌ Image not found at {image_path}")
        print("Please provide a test image in the data/ folder")
        return

    # Load image
    print(f"Loading image from {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print("❌ Failed to load image")
        return

    print(f"Image shape: {image.shape}")

    # Create Marigold estimator
    print("\n" + "=" * 60)
    print("Creating Marigold depth estimator...")
    print("=" * 60)

    # Use LCM variant for faster inference (4 steps)
    estimator = create_depth_estimator(
        "marigold",
        model_variant="lcm",  # or "base" for higher quality but slower
        num_inference_steps=4,  # 4 for LCM, 50 for base
        ensemble_size=1,  # increase to 3-5 for better quality
    )

    # Get model info
    info = estimator.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Variant: {info['variant']}")
    print(f"Device: {info['device']}")
    print(f"Inference steps: {info['inference_steps']}")
    print(f"Ensemble size: {info['ensemble_size']}")

    # Estimate depth
    print("\n" + "=" * 60)
    print("Estimating depth... (this may take a moment)")
    print("=" * 60)

    depth_map = estimator.estimate(image)

    print(f"\n✓ Depth estimation complete!")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

    # Visualize depth with different colormaps
    colormaps = ["INFERNO", "VIRIDIS", "PLASMA", "TURBO"]

    for colormap in colormaps:
        colored_depth = visualize_depth(depth_map, colormap=colormap)
        output_path = output_dir / f"marigold_depth_{colormap.lower()}.png"
        cv2.imwrite(str(output_path), colored_depth)
        print(f"Saved: {output_path}")

    # Save raw depth map
    raw_output = output_dir / "marigold_depth_raw.npy"
    np.save(str(raw_output), depth_map)
    print(f"Saved raw depth: {raw_output}")

    print("\n✓ All outputs saved successfully!")


if __name__ == "__main__":
    main()
