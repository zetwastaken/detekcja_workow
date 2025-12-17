"""Shared helpers for creating YOLO dataset folders and metadata."""

from pathlib import Path
from typing import Iterable, Sequence
import shutil


def create_dataset_structure(
    base_dir: Path, splits: Iterable[str] = ("train", "valid")
) -> None:
    """Create images/labels subfolders for each split."""
    for split in splits:
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_labels(
    src_dataset: Path, dst_dataset: Path, splits: Iterable[str] = ("train", "valid")
) -> None:
    """Copy label txt files from source dataset to destination dataset structure."""
    for split in splits:
        src = src_dataset / split / "labels"
        if not src.exists():
            continue
        dst = dst_dataset / split / "labels"
        dst.mkdir(parents=True, exist_ok=True)
        for label_file in src.glob("*.txt"):
            shutil.copy2(label_file, dst / label_file.name)


def write_data_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    channels: int | None = None,
    header: str | None = None,
    generator: str | None = None,
) -> None:
    """Write a YOLO data.yaml with optional header comment and channels entry."""
    names_block = "\n".join(
        [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
    )

    header_line = f"# {header}" if header else "# YOLOv8 Dataset Configuration"
    generator_line = f"# Generated automatically by {generator}" if generator else ""
    generator_line = generator_line and f"{generator_line}\n"

    channels_line = f"channels: {channels}\n" if channels is not None else ""

    content = (
        f"{header_line}\n"
        f"{generator_line}"
        "\n"
        "# Dataset path (absolute)\n"
        f"path: {dataset_dir}\n"
        "\n"
        "# Train and validation image paths (relative to 'path')\n"
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        "# Number of classes\n"
        f"nc: {len(class_names)}\n"
        "\n"
        "# Class names\n"
        "names:\n"
        f"{names_block}\n"
    )

    if channels_line:
        content += "\n# Input channels\n" + channels_line

    with open(dataset_dir / "data.yaml", "w") as f:
        f.write(content)
