from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

project_root = Path(__file__).resolve().parent.parent
data_yaml = project_root / "datasets" / "dataset_yolov8_V1" / "data.yaml"

results = model.train(
    data=str(data_yaml),
    epochs=1500,
    patience=150,
    imgsz=640,
    batch=32,
    lr0=0.01,
    lrf=0.01,
    task="segment",
    cache=False,
    # device='cpu',
    device=0,
    cos_lr=True,
    optimizer="auto",
    name=f'yolov8_sandbag_seg_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}',
    exist_ok=True,
    save=True,
    save_period=10,
    # ? augmentaion parameters
    degrees=15.0,
    mosaic=1.0,
    fliplr=0.5,
    flipud=0.2,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    copy_paste=0.0,
    close_mosaic=10,
    amp=True,
    mixup=0.1,
    dropout=0.0,  # Regularizacja
    weight_decay=0.0005,  # Regularizacja
    warmup_epochs=5.0,  # Zwiększone - łagodniejszy start
    warmup_momentum=0.8,
)
