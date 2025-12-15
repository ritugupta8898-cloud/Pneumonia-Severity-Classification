from tensorflow.keras.models import load_model

import sys
import cv2
import numpy as np
model = load_model(
    "/Users/pratyushgupta/Documents/Pneumonia-Severity-Classification/cnn_pneumonia.h5"
)

def preprocess_image(path):
    img = cv2.imread(path)          # reads as BGR
    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
    return img
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    x = preprocess_image(sys.argv[1])
    pred = model.predict(x)

    print("Raw model output:", pred)