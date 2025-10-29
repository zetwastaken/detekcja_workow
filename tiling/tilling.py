"""
Python script for automatic tiling all images within a specified
directory into smaller, overlapping tiles.
"""

import os
import argparse
import cv2 as cv


def tilling(path: str, tile_size: int = 640, overlap: int = 128):
    """
    Tiles all images found in specified directory into smaller patches with a defined overlap.
    Results are saved in 'img_tiles' folder.

    Function ensures that the last tiles in each row/column are correctly cropped to
    stay inside the original image boundaries.

    :param path: Path to the directory containing source images.
    "type path: str
    :param tile_size: Target size of each tile, tile_size x tile_size. Defaults to 640.
    "type tile_size: int
    :param overlap: The size of the overlap between neighbouring tiles in px. Defaults to 128.
    "type overlap: int
    :return: none
    """
    # stride = step size for tiling
    stride = tile_size - overlap
    if stride <= 0:
        print("Error: overlap must be less than tile_size:")
        return

    # Output directory
    output_dir = "img_tiles"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # supported image formats
    img_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # iterate through images in folder
    for filename in os.listdir(path):
        if filename.lower().endswith(img_formats):
            img_path = os.path.join(path, filename)
            # read image
            img = cv.imread(img_path)
            if img is None:
                print(f"Error reading image: {filename}. Skipping.")
                continue
        height, width = img.shape[:2]
        print(f"Processing image: {filename} (width: {width}, height {height}")

        # genereate x coordinates
        x_starts = []
        x = 0
        while x <= width - tile_size:
            x_starts.append(x)
            x += stride

        # add last tile if needed
        if x_starts and x_starts[-1] < width - tile_size:
            x_starts.append(width - tile_size)
        elif not x_starts and width >= tile_size:
            x_starts.append(0)

        # generate y coordinates
        y_starts = []
        y = 0
        while y <= height - tile_size:
            y_starts.append(y)
            y += stride

        if y_starts and y_starts[-1] < height - tile_size:
            y_starts.append(height - tile_size)
        elif not y_starts and height >= tile_size:
            y_starts.append(0)

        # extract and save
        tile_count = 0
        for i, y_start in enumerate(y_starts):
            for j, x_start in enumerate(x_starts):
                tile = img[y_start : y_start + tile_size, x_start : x_start + tile_size]

                base_name, _ = os.path.splitext(filename)
                output_filename = f"{base_name}_R{i:03d}_C{j:03d}.jpg"
                output_filepath = os.path.join(output_dir, output_filename)

                cv.imwrite(output_filepath, tile)
                tile_count += 1
        print(f"Made {tile_count} tiles for image: {filename}.")


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
