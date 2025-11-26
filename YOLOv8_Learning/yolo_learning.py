from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data='datasets/dataset_yolov8_V1/data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    task='segment',
    # cache=True,
    device=0,
    patience=50,
    cos_lr=True,
    optimizer='auto',
    name='yolov8_sandbag_seg_v1',
    exist_ok=True,
    #? augmentaion parameters
    degrees=15.0,
    mosaic=1.0,
    fliplr=0.5,
    scale=0.5,
    # hsv_h=0.015,
    # hsv_s=0.7,
    # hsv_v=0.4,
    copy_paste=0.3,
    mixup=0.1,
)