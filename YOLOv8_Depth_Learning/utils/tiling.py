"""Reusable tiling helpers for dataset generation and inference."""

from typing import Dict, List, Tuple

import numpy as np


def _start_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    """Compute start indices for tiling along one dimension."""
    stride = tile_size - overlap
    starts: List[int] = []
    pos = 0

    while pos <= length - tile_size:
        starts.append(pos)
        pos += stride

    if starts and starts[-1] < length - tile_size:
        starts.append(length - tile_size)
    elif not starts and length >= tile_size:
        starts.append(0)

    return starts


def tile_coordinates(
    image_shape: Tuple[int, int], tile_size: int, overlap: int
) -> Tuple[List[int], List[int]]:
    """Return y and x start coordinates for tiling an image."""
    height, width = image_shape[:2]
    y_starts = _start_positions(height, tile_size, overlap)
    x_starts = _start_positions(width, tile_size, overlap)
    return y_starts, x_starts


def tile_image(
    image: np.ndarray, tile_size: int, overlap: int
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Tile an image into overlapping patches.

    Returns a list of (tile, y_start, x_start) tuples.
    """
    y_starts, x_starts = tile_coordinates(image.shape, tile_size, overlap)
    tiles: List[Tuple[np.ndarray, int, int]] = []

    for y_start in y_starts:
        for x_start in x_starts:
            tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tiles.append((tile, y_start, x_start))

    return tiles


def tile_image_with_names(
    image: np.ndarray,
    base_name: str,
    tile_size: int,
    overlap: int,
    extension: str = ".jpg",
) -> Dict[str, np.ndarray]:
    """
    Tile an image and return a dict mapping tile filenames to tile arrays.

    Filenames follow the pattern <base_name>_R###_C###.<ext>
    to match existing dataset naming.
    """
    y_starts, x_starts = tile_coordinates(image.shape, tile_size, overlap)
    tiles: Dict[str, np.ndarray] = {}

    for row_idx, y_start in enumerate(y_starts):
        for col_idx, x_start in enumerate(x_starts):
            tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tile_filename = f"{base_name}_R{row_idx:03d}_C{col_idx:03d}{extension}"
            tiles[tile_filename] = tile

    return tiles
