"""
Wrapper to run yolo_depth_inference without tiling.
Defaults to running on original photos from detekcja_workow/data,
generating RGBD inputs on the fly (since raw photos are RGB).
"""

from copy import deepcopy
from pathlib import Path

from yolo_depth_inference import PRESET_SETTINGS, main


def main_no_tile():
    settings = deepcopy(PRESET_SETTINGS)
    settings["tile"] = False
    settings["from_rgb"] = True  # photos in data are RGB; generate depth/RGBD
    settings["run_type"] = "rgbd"
    settings["source"] = [Path(__file__).resolve().parents[1] / "data"]
    main(settings=settings)


if __name__ == "__main__":
    main_no_tile()
