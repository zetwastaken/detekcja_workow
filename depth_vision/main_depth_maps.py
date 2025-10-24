"""
Main application - Bag Detection with Depth Estimation
Demonstrates the flexible multi-model depth estimation framework.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
from depth_vision.factory import create_depth_estimator, DepthEstimatorFactory
from depth_vision.base import BaseDepthEstimator
from depth_vision.utils import visualize_depth


def initialize_estimator(estimator_type: str = "midas", **config) -> BaseDepthEstimator:
    """
    Initialize a depth estimator.

    Args:
        estimator_type: Type of estimator ("midas", "zoe", etc.)
        **config: Model-specific configuration (e.g., model_type="DPT_Large")

    Returns:
        Initialized depth estimator instance

    Example:
        # MiDaS with specific model
        estimator = initialize_estimator("midas", model_type="DPT_Hybrid")

        # Future: Different estimator
        # estimator = initialize_estimator("zoe", model_size="large")
    """
    print(f"Initializing {estimator_type} depth estimator")
    estimator = create_depth_estimator(estimator_type, **config)
    print(f"✓ Model: {estimator.get_model_info()['name']}")
    return estimator


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array, or None if loading failed
    """
    print("Loading image")
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None

    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    return image


def generate_depth_map(estimator: BaseDepthEstimator, image: np.ndarray) -> np.ndarray:
    """
    Generate depth map from image.

    Args:
        estimator: Initialized depth estimator instance
        image: Input image

    Returns:
        Depth map as numpy array
    """
    print("Generating depth map")
    depth_map = estimator.estimate(image)

    print(f"✓ Depth map generated: {depth_map.shape}")
    print(f"  - Min depth: {depth_map.min():.2f}")
    print(f"  - Max depth: {depth_map.max():.2f}")
    print(f"  - Mean depth: {depth_map.mean():.2f}")

    return depth_map


def find_extreme_points(depth_map: np.ndarray) -> Dict[str, Tuple[Any, Any, Any]]:
    """
    Find the closest and farthest points in the depth map.

    Args:
        depth_map: Depth map array

    Returns:
        Dictionary with 'closest' and 'farthest' keys, each containing (x, y, depth_value)
    """
    # Find closest point (highest depth value)
    closest_y, closest_x = np.unravel_index(np.argmax(depth_map), depth_map.shape)
    closest_depth = depth_map[closest_y, closest_x]

    # Find farthest point (lowest depth value)
    farthest_y, farthest_x = np.unravel_index(np.argmin(depth_map), depth_map.shape)
    farthest_depth = depth_map[farthest_y, farthest_x]

    return {
        "closest": (int(closest_x), int(closest_y), float(closest_depth)),
        "farthest": (int(farthest_x), int(farthest_y), float(farthest_depth)),
    }


def analyze_depth_map(depth_map: np.ndarray) -> Dict[str, Tuple[Any, Any, Any]]:
    """
    Analyze depth map and print information about extreme points.

    Args:
        depth_map: Depth map array

    Returns:
        Dictionary with analysis results
    """
    print("Analyzing depth information")

    extreme_points = find_extreme_points(depth_map)

    closest_x, closest_y, closest_depth = extreme_points["closest"]
    print(f"✓ Closest point detected at: ({closest_x}, {closest_y})")
    print(f"  Depth value: {closest_depth:.2f}")

    farthest_x, farthest_y, farthest_depth = extreme_points["farthest"]
    print(f"✓ Farthest point detected at: ({farthest_x}, {farthest_y})")
    print(f"  Depth value: {farthest_depth:.2f}")

    return extreme_points


def save_depth_visualization(
    depth_map: np.ndarray,
    output_path: str = "output_depth_colored.png",
    colormap: int = cv2.COLORMAP_INFERNO,
) -> str:
    """
    Save colored depth visualization.

    Args:
        depth_map: Depth map array
        output_path: Path to save the visualization
        colormap: OpenCV colormap to use

    Returns:
        Path where the file was saved
    """
    colored_depth = visualize_depth(depth_map, colormap=colormap)
    cv2.imwrite(output_path, colored_depth)
    print(f"✓ Saved: {output_path}")
    return output_path


def save_comparison_image(
    original_image: np.ndarray,
    depth_map: np.ndarray,
    output_path: str = "output_comparison.jpg",
    colormap: int = cv2.COLORMAP_INFERNO,
) -> str:
    """
    Save side-by-side comparison of original and depth map.

    Args:
        original_image: Original input image
        depth_map: Depth map array
        output_path: Path to save the comparison
        colormap: OpenCV colormap to use

    Returns:
        Path where the file was saved
    """
    h, w = original_image.shape[:2]
    colored_depth = visualize_depth(depth_map, colormap=colormap)
    colored_depth_resized = cv2.resize(colored_depth, (w, h))
    comparison = np.hstack([original_image, colored_depth_resized])
    cv2.imwrite(output_path, comparison)
    print(f"✓ Saved: {output_path}")
    return output_path


def save_results(
    original_image: np.ndarray,
    depth_map: np.ndarray,
    original_image_path: str,
    model_info: Dict[str, str],
    output_dir: str = "output",
) -> None:
    """
    Save all output visualizations in organized format.

    Args:
        original_image: Original input image
        depth_map: Depth map array
        original_image_path: Path to original image file
        model_info: Dictionary containing model information
        output_dir: Directory to save outputs
    """
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Extract base filename without extension and model name
    base_name = Path(original_image_path).stem
    model_name = model_info["name"].replace(" ", "_")

    # Construct output paths
    depth_output = f"{output_dir}/{base_name}_{model_name}_depth.png"
    comparison_output = f"{output_dir}/{base_name}_{model_name}_comparison.jpg"

    save_depth_visualization(depth_map, depth_output)
    save_comparison_image(original_image, depth_map, comparison_output)


def main(estimator_type: str = "midas", **estimator_config):
    """
    Main function orchestrating the depth estimation pipeline.

    Args:
        estimator_type: Type of depth estimator to use ("midas", "zoe", etc.)
        **estimator_config: Configuration for the specific estimator

    Example:
        # Use MiDaS with DPT_Large
        main("midas", model_type="DPT_Large")

        # Future: Use a different estimator
        # main("zoe", model_size="large")
    """
    title = "Bag Detection with Depth Vision"
    print("=" * 70)
    print(" " * ((70 - len(title)) // 2) + title)
    print("=" * 70)

    # Configuration
    image_path = "data/worki_1.jpg"

    # Execute pipeline with specified estimator
    estimator = initialize_estimator(estimator_type, **estimator_config)
    image = load_image(image_path)

    if image is None:
        return None

    depth_map = generate_depth_map(estimator, image)
    analyze_depth_map(depth_map)
    model_info = estimator.get_model_info()
    save_results(image, depth_map, image_path, model_info)

    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)

    return depth_map


def process_single_image(
    image_path: str,
    estimator: Optional[BaseDepthEstimator] = None,
    estimator_type: str = "midas",
    **estimator_config,
) -> np.ndarray:
    """
    Process a single image and return its depth map.

    Args:
        image_path: Path to the image file
        estimator: Optional pre-initialized estimator (creates new if None)
        estimator_type: Type of estimator to create if None
        **estimator_config: Configuration for new estimator (e.g., model_type="DPT_Large")

    Returns:
        Depth map as numpy array

    Raises:
        ValueError: If image cannot be loaded
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Initialize estimator if not provided
    if estimator is None:
        estimator = create_depth_estimator(estimator_type, **estimator_config)

    # Get depth map
    depth_map = estimator.estimate(image)

    return depth_map


def process_multiple_images(
    image_paths: list[str], estimator_type: str = "midas", **estimator_config
) -> list[np.ndarray]:
    """
    Process multiple images efficiently (loads model only once).

    Args:
        image_paths: List of image file paths
        estimator_type: Type of estimator to use
        **estimator_config: Configuration for estimator (e.g., model_type="DPT_Large")

    Returns:
        List of depth maps
    """
    # Initialize estimator once for efficiency
    estimator = create_depth_estimator(estimator_type, **estimator_config)

    depth_maps = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            depth = process_single_image(img_path, estimator)
            depth_maps.append(depth)
            print(f"  ✓ Depth map: {depth.shape}")
        except ValueError as e:
            print(f"  ❌ Error: {e}")

    return depth_maps


if __name__ == "__main__":
    # Example 1: Run with default MiDaS DPT_Large
    # depth_map = main("midas", model_type="DPT_Large")

    # depth_map_2 = main("depth_anything", model_size="large")

    # depth_map_3 = main("zoedepth", model_type="NK")

    # Marigold with LCM variant - uses default 4 steps automatically
    depth_map_4 = main("marigold", model_variant="base", num_inference_steps=1)

    # Example 3: Process multiple images
    # images = ["data/worki_1.jpg", "data/worki_2.jpg"]
    # depth_maps = process_multiple_images(images, "midas", model_type="DPT_Hybrid")

    # Example 4 (future): Use different estimator

    # depth_maps = process_multiple_images(images)
