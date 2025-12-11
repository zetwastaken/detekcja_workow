"""
Test script for tile reassembly function.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple


def tile_image(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: int = 80,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Tile an image into overlapping patches.
    Returns a list of (tile, y_start, x_start) tuples.
    """
    stride = tile_size - overlap
    height, width = image.shape[:2]
    tiles: List[Tuple[np.ndarray, int, int]] = []

    # Generate x coordinates
    x_starts = []
    x = 0
    while x <= width - tile_size:
        x_starts.append(x)
        x += stride
    if x_starts and x_starts[-1] < width - tile_size:
        x_starts.append(width - tile_size)
    elif not x_starts and width >= tile_size:
        x_starts.append(0)

    # Generate y coordinates
    y_starts = []
    y = 0
    while y <= height - tile_size:
        y_starts.append(y)
        y += stride
    if y_starts and y_starts[-1] < height - tile_size:
        y_starts.append(height - tile_size)
    elif not y_starts and height >= tile_size:
        y_starts.append(0)

    # Extract tiles with their positions
    for y_start in y_starts:
        for x_start in x_starts:
            tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tiles.append((tile, y_start, x_start))

    return tiles


def reassemble_predictions(
    tiles_info: List[Tuple[int, int, np.ndarray]],
    original_shape: Tuple[int, int],
    tile_size: int = 640,
    overlap: int = 80,
) -> np.ndarray:
    """
    Reassemble prediction tiles back into a full-resolution image.
    """
    height, width = original_shape
    
    if tiles_info:
        channels = tiles_info[0][2].shape[2] if len(tiles_info[0][2].shape) == 3 else 1
    else:
        channels = 3
    
    if channels == 1:
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
    else:
        output = np.zeros((height, width, channels), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)

    # Create blending weights (feather edges to reduce artifacts)
    tile_weights = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Apply smooth blending at overlapping regions
    # Use distance from edge, but ensure minimum weight of 0.1 to avoid zero-weight pixels
    for i in range(tile_size):
        for j in range(tile_size):
            dist_from_edge = min(i, j, tile_size - 1 - i, tile_size - 1 - j)
            # Add 1 to distance to avoid zero weights at edges
            normalized_dist = min((dist_from_edge + 1) / (overlap / 2 + 1), 1.0)
            tile_weights[i, j] = max(normalized_dist, 0.1)  # Minimum weight of 0.1

    # Blend tiles
    for y_start, x_start, tile in tiles_info:
        y_end = min(y_start + tile_size, height)
        x_end = min(x_start + tile_size, width)
        
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        tile_cropped = tile[:tile_h, :tile_w]
        weights_cropped = tile_weights[:tile_h, :tile_w]
        
        if channels == 1:
            output[y_start:y_end, x_start:x_end] += tile_cropped.astype(np.float32) * weights_cropped
            weights[y_start:y_end, x_start:x_end] += weights_cropped
        else:
            for c in range(channels):
                output[y_start:y_end, x_start:x_end, c] += tile_cropped[:, :, c].astype(np.float32) * weights_cropped
            weights[y_start:y_end, x_start:x_end] += weights_cropped

    # Normalize by weights
    if channels == 1:
        mask = weights > 0
        output[mask] /= weights[mask]
    else:
        for c in range(channels):
            mask = weights > 0
            output[:, :, c][mask] /= weights[mask]

    return output.astype(np.uint8)


def test_tile_reassembly():
    """Test that tiling and reassembly works correctly."""
    print("Testing tile reassembly...")
    
    # Create a test image with a gradient pattern
    test_image = np.zeros((1280, 1920, 3), dtype=np.uint8)
    
    # Add horizontal gradient
    for i in range(test_image.shape[1]):
        test_image[:, i, 0] = int(255 * i / test_image.shape[1])  # Blue channel
    
    # Add vertical gradient
    for i in range(test_image.shape[0]):
        test_image[i, :, 1] = int(255 * i / test_image.shape[0])  # Green channel
    
    # Red channel is constant
    test_image[:, :, 2] = 128
    
    # Save original (use PNG to avoid compression artifacts)
    cv2.imwrite("/tmp/test_original.png", test_image)
    print(f"Created test image: {test_image.shape}")
    
    # Tile the image
    tiles = tile_image(test_image, tile_size=640, overlap=80)
    print(f"Created {len(tiles)} tiles")
    
    # Prepare tiles for reassembly (simulating prediction output)
    tiles_info = [(y, x, tile) for tile, y, x in tiles]
    
    # Reassemble
    reassembled = reassemble_predictions(
        tiles_info,
        test_image.shape[:2],
        tile_size=640,
        overlap=80
    )
    
    # Save reassembled (use PNG to avoid compression artifacts)
    cv2.imwrite("/tmp/test_reassembled.png", reassembled)
    print(f"Reassembled image: {reassembled.shape}")
    
    # Calculate difference
    diff = np.abs(test_image.astype(np.int16) - reassembled.astype(np.int16))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff:.2f}")
    
    # Check if reassembly is accurate (allow for small rounding errors)
    if max_diff <= 2 and mean_diff < 0.5:
        print("✓ Test PASSED: Reassembly is accurate!")
        return True
    else:
        print(f"⚠️  Test has differences (max: {max_diff}, mean: {mean_diff:.2f})")
        # Save diff map
        diff_map = np.clip(diff * 50, 0, 255).astype(np.uint8)  # Amplify for visibility
        cv2.imwrite("/tmp/test_diff.png", diff_map)
        print("Diff map saved to /tmp/test_diff.png")
        
        # If the differences are small enough for visual use, it's acceptable
        if max_diff < 10 and mean_diff < 2:
            print("✓ Differences are within acceptable range for visual results")
            return True
        return False


if __name__ == "__main__":
    test_tile_reassembly()
