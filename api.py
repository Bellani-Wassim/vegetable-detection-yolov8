from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os

# Create API object
app = FastAPI()

# Load your trained model (fixing the path format)
model = YOLO(r"runs\detect\yolov8n-carrot\weights\best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO on the uploaded image
    results = model(temp_file)

    # Remove temp file to save space
    os.remove(temp_file)

    # Return YOLO's output in JSON format
    return results[0].tojson()
