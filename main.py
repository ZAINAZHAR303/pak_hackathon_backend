from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive Model URL
model_url = "https://drive.google.com/uc?export=download&id=1EcG2J2-xeCLbOfCdOju5JSqvSy-NNYc2"
model_path = "yolov8x.pt"

# Download model if not already available
def download_model():
    print("🔄 Downloading YOLOv8x model from Google Drive...")
    response = requests.get(model_url, stream=True)
    with open(model_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("✅ Model downloaded successfully!")

if not os.path.exists(model_path):
    download_model()

# Load Models
import time

print("🔄 Loading models... Please wait.")
start_time = time.time()
species_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
species_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").eval()
yolo_model = YOLO(model_path)
threat_model = pipeline("image-classification", model="nateraw/vit-base-beans")
end_time = time.time()
print(f"✅ Models loaded in {end_time - start_time:.2f} seconds!")

@app.post("/detect_species/")
async def detect_species(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results = species_model(image)
    return {"species_detected": results}

@app.post("/count_population/")
async def count_population(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results = yolo_model(image)
    return {"count": len(results[0].boxes)}

@app.post("/assess_health/")
async def assess_health(file: UploadFile = File(...)):
    image = Image.open(file.file)
    # Placeholder function: Implement actual health assessment logic
    return {"status": "Good", "score": 85, "indicators": {"brightness": 75, "saturation": 80}}

@app.post("/detect_threat/")
async def detect_threat(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results = threat_model(image)
    return {"threats_detected": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
