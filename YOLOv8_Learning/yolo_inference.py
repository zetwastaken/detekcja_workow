"""
YOLOv8 Inference Script - Testing trained model on new images
"""
from pathlib import Path

from ultralytics import YOLO

root_dir = Path(__file__).resolve().parents[2]
project_dir = Path(__file__).resolve().parents[1]
runs_dir = root_dir / 'runs' / 'segment'

# Load your trained model (prefer v2 as requested)
preferred_run = runs_dir / 'yolov8_sandbag_seg_v2' / 'weights' / 'best.pt'

if preferred_run.exists():
    model_path = preferred_run
else:
    # Fallback: pick the newest available run/weights
    best_weights = sorted(runs_dir.glob('*/weights/best.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not best_weights:
        raise FileNotFoundError(f"No weights found under {runs_dir}")
    model_path = best_weights[0]

model = YOLO(model_path)

# Path to image(s) you want to test
# Can be:
# - Single image: 'path/to/image.jpg'
# - Multiple images: ['image1.jpg', 'image2.jpg']
# - Folder: 'path/to/folder/'
# - URL: 'https://example.com/image.jpg'
# test_image = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/tiling/all_V1_640_80'  # CHANGE THIS

# test_image = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/worki_ok/dobre_do_wykorzystania/DJI_0398_2pD.JPG'  # Whole image

base_path = project_dir / 'tiling' / 'all_V1_640_80'
test_images = [
    'DJI_0398_2pD_R002_C004.jpg',
    'DJI_0480_3pD_R003_C002.jpg',
    'IMG_2015_1pK_R003_C007.jpg',
    'IMG_2015_1pK_R003_C005.jpg',
    'IMG_2015_1pK_R004_C008.jpg',
    'IMG_2027_1pK_R003_C003.jpg',
    'IMG_2072_3pK_R001_C006.jpg',
    'IMG_2075_3pK_R003_C010.jpg',
    'IMG_2077_3pK_R003_C004.jpg',
    'IMG_2085_4pK_R004_C008.jpg',
    'worki_1_R002_C004.jpg',
]
test_image = [str(base_path / img) for img in test_images]

# Run inference
results = model.predict(
    source=test_image,
    save=True,              # Save results to runs/segment/predict/
    save_txt=True,          # Save labels as .txt files
    save_conf=True,         # Save confidence scores
    conf=0.25,              # Confidence threshold (0-1)
    iou=0.7,                # IoU threshold for NMS
    show_labels=True,       # Show class labels
    show_conf=True,         # Show confidence scores
    show_boxes=True,        # Show bounding boxes
    line_width=2,           # Line width for boxes and masks
)

# Print results
for i, result in enumerate(results):
    print(f"\n=== Image {i+1} ===")
    print(f"Image path: {result.path}")
    print(f"Image shape: {result.orig_shape}")
    
    if result.masks is not None:
        print(f"Number of detected sandbags: {len(result.masks)}")
        
        # Print details for each detection
        for j, (box, mask, conf, cls) in enumerate(zip(
            result.boxes.xyxy, 
            result.masks.data,
            result.boxes.conf,
            result.boxes.cls
        )):
            print(f"\nSandbag {j+1}:")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Bounding box: {box.tolist()}")
            print(f"  Mask shape: {mask.shape}")
    else:
        print("No sandbags detected")
    
    print(f"\nResults saved to: {result.save_dir}")

print("\n✅ Inference completed!")
