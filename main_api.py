from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
from io import BytesIO
import joblib
import onnxruntime as ort
from utils import make_inference_onnx
import os

app = FastAPI()

# Load necessary files
onnx_model_path = './model/resnet50_feature_extractor.onnx'
ort_session = ort.InferenceSession(onnx_model_path)
gmm = joblib.load('./model/gmm_cars_v1.pkl')
scaler = joblib.load('./scaler/scaler_cars_v1.pkl')


@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    image_paths = []
    for file in files:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        image.save(f"./temp_images/{file.filename.split('/')[-1]}")
        image_paths.append(f"./temp_images/{file.filename.split('/')[-1]}")

    labels = make_inference_onnx(image_paths, ort_session, gmm, scaler)

    for image_path in image_paths:
        os.remove(image_path)

    # Return JSON with the labels of each image
    return JSONResponse(content={"labels": labels.tolist()})
