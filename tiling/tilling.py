"""
Python script for automatic tiling all images within a specified
directory into smaller, overlapping tiles.
"""

import argparse
import os
from pathlib import Path

import cv2 as cv

# Add project root so we can reuse shared tiling helpers
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from YOLOv8_Depth_Learning.utils.tiling import tile_image_with_names


def tilling(path: str, tile_size: int = 640, overlap: int = 80):
    """
    Tiles all images found in specified directory into smaller patches with a defined overlap.
    Results are saved in 'img_tiles' folder.

    Function ensures that the last tiles in each row/column are correctly cropped to
    stay inside the original image boundaries.

    :param path: Path to the directory containing source images.
    "type path: str
    :param tile_size: Target size of each tile, tile_size x tile_size. Defaults to 640.
    "type tile_size: int
    :param overlap: The size of the overlap between neighbouring tiles in px. Defaults to 80.
    "type overlap: int
    :return: none
    """
    stride = tile_size - overlap
    if stride <= 0:
        print("Error: overlap must be less than tile_size.")
        return

    output_dir = "img_tiles"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    img_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(path):
        if not filename.lower().endswith(img_formats):
            continue

        img_path = os.path.join(path, filename)
        img = cv.imread(img_path)
        if img is None:
            print(f"Error reading image: {filename}. Skipping.")
            continue

        height, width = img.shape[:2]
        print(f"Processing image: {filename} (width: {width}, height: {height})")

        base_name, _ = os.path.splitext(filename)
        tiles = tile_image_with_names(img, base_name, tile_size, overlap)

        for tile_name, tile_img in tiles.items():
            output_filepath = os.path.join(output_dir, tile_name)
            cv.imwrite(output_filepath, tile_img)

        print(f"Made {len(tiles)} tiles for image: {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile images in a folder into smaller overlapping patches."
    )
    parser.add_argument(
        "path", type=str, help="Path to the folder containing images to be tiled."
    )
    parser.add_argument(
        "--tile_size", type=int, default=640, help="Size of each tile. (Default: 640)"
    )
    parser.add_argument(
        "--overlap", type=int, default=128, help="Tiles overlap size. (Default: 128)"
    )
    args = parser.parse_args()
    tilling(args.path, args.tile_size, args.overlap)
    print("Tiling completed.")
