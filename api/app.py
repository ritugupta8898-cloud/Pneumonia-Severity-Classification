import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model

MODEL_PATH = "/Users/pratyushgupta/Documents/Pneumonia-Severity-Classification/cnn_pneumonia.h5"

app = FastAPI()          # <-- THIS MUST EXIST
model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    model = load_model(MODEL_PATH)

def preprocess_image_bytes(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image_bytes(image_bytes)

    prob = float(model.predict(x)[0][0])

    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "p_pneumonia": round(prob, 4)
    }
