from ultralytics import YOLO

#we gonna test the model on one image

model = YOLO('runs/detect/yolov8n-carrot/weights/best.pt')
results = model('data/test/images/carrots (21).jpg')
results[0].show()
results[0].save(filename="predict carrots (21).jpg")
