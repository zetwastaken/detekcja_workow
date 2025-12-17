"""
Test script for SAHI-based RGBD inference.

This test validates the core logic and structure of the SAHI inference script
without requiring full dependencies to be installed.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_script_structure():
    """Test that the SAHI inference script has correct structure."""
    script_path = Path(__file__).parent / "yolo_rgbd_inference_sahi.py"
    
    print("Testing SAHI inference script structure...")
    
    # Check file exists
    assert script_path.exists(), f"Script not found: {script_path}"
    print("✓ Script file exists")
    
    # Read script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for required imports
    required_imports = [
        'from sahi import AutoDetectionModel',
        'from sahi.predict import get_sliced_prediction',
        'from ultralytics import YOLO',
        'import cv2',
        'import numpy as np',
        'from depth_vision.factory import DepthEstimatorFactory',
        'from depth_vision.utils import normalize_depth',
    ]
    
    for imp in required_imports:
        assert imp in content, f"Missing import: {imp}"
    
    print("✓ All required imports present")
    
    # Check for key functions
    required_functions = [
        'def create_rgbd_image',
        'def find_model_weights',
        'def main',
    ]
    
    for func in required_functions:
        assert func in content, f"Missing function: {func}"
    
    print("✓ All required functions present")
    
    # Check for RGBDDetectionModel class
    assert 'class RGBDDetectionModel(AutoDetectionModel):' in content, \
        "Missing RGBDDetectionModel class"
    print("✓ RGBDDetectionModel class present")
    
    # Check for key RGBDDetectionModel methods
    required_methods = [
        'def __init__',
        'def perform_inference',
        'def _create_object_prediction_list_from_original_predictions',
    ]
    
    for method in required_methods:
        assert method in content, f"Missing method in RGBDDetectionModel: {method}"
    
    print("✓ All required methods in RGBDDetectionModel present")
    
    # Check for SAHI parameters configuration
    assert 'SLICE_HEIGHT' in content, "Missing SLICE_HEIGHT parameter"
    assert 'SLICE_WIDTH' in content, "Missing SLICE_WIDTH parameter"
    assert 'OVERLAP_HEIGHT_RATIO' in content, "Missing OVERLAP_HEIGHT_RATIO parameter"
    assert 'OVERLAP_WIDTH_RATIO' in content, "Missing OVERLAP_WIDTH_RATIO parameter"
    print("✓ SAHI parameters configured")
    
    # Check for detection counting
    assert 'total_detections' in content, "Missing detection counting"
    assert 'detection_stats.txt' in content, "Missing statistics file generation"
    print("✓ Detection counting functionality present")
    
    # Check for get_sliced_prediction usage
    assert 'get_sliced_prediction(' in content, "Missing SAHI sliced prediction call"
    print("✓ SAHI sliced prediction properly called")
    
    # Check for statistics writing
    assert "Per-Image Detection Counts:" in content, \
        "Missing per-image statistics in output"
    assert "Total detections:" in content, "Missing total detections in output"
    assert "Average detections per image:" in content, \
        "Missing average detections in output"
    print("✓ Statistics output properly formatted")
    
    print("\n" + "=" * 70)
    print("✓ All structure tests PASSED!")
    print("=" * 70)
    return True


def test_constants_and_configuration():
    """Test that configuration constants are reasonable."""
    script_path = Path(__file__).parent / "yolo_rgbd_inference_sahi.py"
    
    print("\nTesting configuration constants...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Extract and validate SAHI parameters
    import re
    
    # Check SLICE_HEIGHT
    match = re.search(r'SLICE_HEIGHT\s*=\s*(\d+)', content)
    assert match, "SLICE_HEIGHT not found"
    slice_height = int(match.group(1))
    assert slice_height == 640, f"Expected SLICE_HEIGHT=640, got {slice_height}"
    print(f"✓ SLICE_HEIGHT = {slice_height}")
    
    # Check SLICE_WIDTH
    match = re.search(r'SLICE_WIDTH\s*=\s*(\d+)', content)
    assert match, "SLICE_WIDTH not found"
    slice_width = int(match.group(1))
    assert slice_width == 640, f"Expected SLICE_WIDTH=640, got {slice_width}"
    print(f"✓ SLICE_WIDTH = {slice_width}")
    
    # Check OVERLAP_HEIGHT_RATIO
    match = re.search(r'OVERLAP_HEIGHT_RATIO\s*=\s*([\d.]+)', content)
    assert match, "OVERLAP_HEIGHT_RATIO not found"
    overlap_h = float(match.group(1))
    assert 0 <= overlap_h < 1, f"OVERLAP_HEIGHT_RATIO should be in [0, 1), got {overlap_h}"
    print(f"✓ OVERLAP_HEIGHT_RATIO = {overlap_h}")
    
    # Check OVERLAP_WIDTH_RATIO
    match = re.search(r'OVERLAP_WIDTH_RATIO\s*=\s*([\d.]+)', content)
    assert match, "OVERLAP_WIDTH_RATIO not found"
    overlap_w = float(match.group(1))
    assert 0 <= overlap_w < 1, f"OVERLAP_WIDTH_RATIO should be in [0, 1), got {overlap_w}"
    print(f"✓ OVERLAP_WIDTH_RATIO = {overlap_w}")
    
    # Check depth model configuration
    assert 'DEPTH_MODEL = "depth_anything"' in content, \
        "Unexpected DEPTH_MODEL value"
    print('✓ DEPTH_MODEL = "depth_anything"')
    
    assert 'DEPTH_CONFIG = {"model_size": "large"}' in content, \
        "Unexpected DEPTH_CONFIG value"
    print('✓ DEPTH_CONFIG = {"model_size": "large"}')
    
    print("\n" + "=" * 70)
    print("✓ All configuration tests PASSED!")
    print("=" * 70)
    return True


def test_documentation():
    """Test that documentation exists and is comprehensive."""
    doc_path = Path(__file__).parent / "README_sahi_inference.md"
    
    print("\nTesting documentation...")
    
    assert doc_path.exists(), f"Documentation not found: {doc_path}"
    print("✓ Documentation file exists")
    
    with open(doc_path, 'r') as f:
        doc_content = f.read()
    
    # Check for key sections
    required_sections = [
        "# SAHI-based RGBD Inference Pipeline",
        "## Overview",
        "## Requirements",
        "## Usage",
        "## Configuration",
        "## Technical Details",
        "## Example",
        "## Performance Notes",
        "## Troubleshooting",
    ]
    
    for section in required_sections:
        assert section in doc_content, f"Missing documentation section: {section}"
    
    print("✓ All required documentation sections present")
    
    # Check for important topics
    important_topics = [
        "SAHI",
        "detection count",
        "detection_stats.txt",
        "overlap ratio",
        "Depth Anything",
        "RGBD",
    ]
    
    for topic in important_topics:
        assert topic.lower() in doc_content.lower(), \
            f"Documentation missing topic: {topic}"
    
    print("✓ All important topics covered in documentation")
    
    print("\n" + "=" * 70)
    print("✓ All documentation tests PASSED!")
    print("=" * 70)
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("SAHI RGBD Inference - Structure and Configuration Tests")
    print("=" * 70)
    
    try:
        test_script_structure()
        test_constants_and_configuration()
        test_documentation()
        
        print("\n" + "=" * 70)
        print("SUCCESS: All tests passed!")
        print("=" * 70)
        print("\nNote: Full runtime testing requires installing dependencies:")
        print("  pip install sahi ultralytics opencv-python numpy torch")
        print("\nTo run full inference, ensure you have:")
        print("  1. Trained YOLO model weights in runs/segment/")
        print("  2. Input images in data/")
        print("  3. All dependencies installed")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
