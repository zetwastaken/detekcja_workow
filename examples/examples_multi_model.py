"""
Examples of using the multi-model depth estimation framework.

This file demonstrates:
1. How to use different depth estimators
2. How to compare results from multiple models
3. How to add your own custom estimator
"""

import cv2
import numpy as np
from pathlib import Path

from depth_vision import (
    DepthEstimatorFactory,
    create_depth_estimator,
    visualize_depth,
    BaseDepthEstimator,
)


def example_1_basic_usage():
    """Example 1: Basic usage with MiDaS"""
    print("\n" + "=" * 70)
    print("Example 1: Basic MiDaS Usage")
    print("=" * 70)

    # Create estimator using factory
    estimator = create_depth_estimator("midas", model_type="DPT_Hybrid")

    # Load and process image
    image = cv2.imread("data/worki_1.jpg")
    depth_map = estimator.estimate(image)

    # Get model info
    info = estimator.get_model_info()
    print(f"Model: {info['name']} {info['variant']}")
    print(f"Device: {info['device']}")
    print(f"Depth map shape: {depth_map.shape}")


def example_2_compare_models():
    """Example 2: Compare different MiDaS model variants"""
    print("\n" + "=" * 70)
    print("Example 2: Compare MiDaS Model Variants")
    print("=" * 70)

    image = cv2.imread("data/worki_1.jpg")
    models = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

    for model_type in models:
        print(f"\nTesting {model_type}...")
        estimator = create_depth_estimator("midas", model_type=model_type)

        depth_map = estimator.estimate(image)

        # Save visualization
        colored = visualize_depth(depth_map)
        output_path = f"output_{model_type.lower()}.png"
        cv2.imwrite(output_path, colored)
        print(f"  ✓ Saved: {output_path}")
        print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")


def example_3_list_available():
    """Example 3: List all available estimators"""
    print("\n" + "=" * 70)
    print("Example 3: Available Depth Estimators")
    print("=" * 70)

    available = DepthEstimatorFactory.list_available()

    print(f"\nFound {len(available)} estimator(s):")
    for name, estimator_class in available.items():
        print(f"  - {name}: {estimator_class.__name__}")

    print("\nSupported MiDaS models:")
    from depth_vision.estimators.midas import MiDaSDepthEstimator

    for model, specs in MiDaSDepthEstimator.SUPPORTED_MODELS.items():
        print(f"  - {model}: {specs}")


def example_4_custom_estimator():
    """Example 4: How to create a custom depth estimator"""
    print("\n" + "=" * 70)
    print("Example 4: Custom Depth Estimator Template")
    print("=" * 70)

    # This is how you would create a custom estimator
    class CustomDepthEstimator(BaseDepthEstimator):
        """Custom depth estimator implementation"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.load_model()

        def load_model(self):
            print("Loading custom model...")
            # Load your model here
            # self.model = load_your_model()
            pass

        def estimate(self, image: np.ndarray) -> np.ndarray:
            # Implement your depth estimation logic
            # depth_map = self.model(image)
            # return depth_map
            raise NotImplementedError("Implement your estimation logic")

        def get_model_info(self):
            return {"name": "CustomDepthModel", "version": "1.0", "custom": True}

    # Register your custom estimator
    DepthEstimatorFactory.register("custom", CustomDepthEstimator)

    print("✓ Custom estimator registered as 'custom'")
    print("  Usage: estimator = create_depth_estimator('custom')")

    # Now it's available in the factory
    available = DepthEstimatorFactory.list_available()
    print(f"\nAvailable estimators: {list(available.keys())}")


def example_5_batch_processing():
    """Example 5: Efficient batch processing"""
    print("\n" + "=" * 70)
    print("Example 5: Batch Processing Multiple Images")
    print("=" * 70)

    # List of images to process
    image_paths = ["data/worki_1.jpg"]  # Add more paths here

    # Create estimator once (efficient for multiple images)
    estimator = create_depth_estimator("midas", model_type="DPT_Hybrid")

    depth_maps = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {i}/{len(image_paths)}: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Could not load image")
            continue

        depth_map = estimator.estimate(image)
        depth_maps.append(depth_map)

        # Save result
        output_path = f"output_batch_{i}.png"
        colored = visualize_depth(depth_map)
        cv2.imwrite(output_path, colored)
        print(f"  ✓ Saved: {output_path}")

    print(f"\n✓ Processed {len(depth_maps)} images")


def example_6_future_models():
    """Example 6: How to use future depth estimators"""
    print("\n" + "=" * 70)
    print("Example 6: Using Future Depth Estimators")
    print("=" * 70)

    print("\nOnce implemented, you'll be able to use them like this:")

    print("\n# Depth Anything:")
    print('estimator = create_depth_estimator("depth_anything", model_size="large")')

    print("\n# ZoeDepth:")
    print('estimator = create_depth_estimator("zoe", model_type="NK")')

    print("\n# Marigold:")
    print('estimator = create_depth_estimator("marigold", num_steps=10)')

    print("\nAll estimators share the same interface:")
    print("depth_map = estimator.estimate(image)")
    print("info = estimator.get_model_info()")


if __name__ == "__main__":
    """Run all examples"""

    print("\n" + "=" * 70)
    print("Multi-Model Depth Estimation Framework - Examples")
    print("=" * 70)

    # Run examples (comment out the ones you don't want to run)

    # example_1_basic_usage()
    # example_2_compare_models()
    example_3_list_available()
    example_4_custom_estimator()
    # example_5_batch_processing()
    example_6_future_models()

    print("\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)
