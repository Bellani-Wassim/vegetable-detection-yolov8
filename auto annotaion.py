from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")  # Pretrained model

splits = ['train', 'validation', 'test']

for split in splits:
    image_dir = Path(f"data/{split}/images")
    label_dir = Path(f"data/{split}/labels")
    label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_dir.glob("*.[jp][pn]g"):
        results = model(img_path)

        for r in results:
            label_path = label_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    xywh = box.xywh[0].tolist()
                    xywh_norm = [
                        xywh[0] / r.orig_shape[1],  # x_center / width
                        xywh[1] / r.orig_shape[0],  # y_center / height
                        xywh[2] / r.orig_shape[1],  # width / width
                        xywh[3] / r.orig_shape[0],  # height / height
                    ]
                    f.write(f"{cls} {' '.join(map(str, xywh_norm))}\n")
