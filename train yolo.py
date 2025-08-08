from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='vegetables.yaml',   
    epochs=30,            # fewer epochs since 1 class
    imgsz=416,            # smaller size for speed
    batch=8,              # smaller batch for low RAM
    device='cpu',         # CPU mode
    name='yolov8n-carrot'
)
