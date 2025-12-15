# Pneumonia X-Ray Classification (CNN)

This project classifies chest X-rays into:

- **0 → NORMAL**
- **1 → PNEUMONIA**

The final CNN model was trained from scratch using TensorFlow/Keras and improved through
architecture tuning, class balancing, and data augmentation, which significantly improved
recall balance and overall performance.

---

## Tech Stack
- Python 3.11
- TensorFlow (tensorflow-macos, tensorflow-metal)
- NumPy (< 2.0)
- OpenCV
- FastAPI
- Uvicorn

---

## Project Structure

```
Pneumonia-Severity-Classification/
├── api/
│   ├── app.py          # FastAPI inference service
│   └── main.py         # CLI inference script
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── train_cnn.py
│   └── evaluate.py
├── models/
│   └── cnn_pneumonia.h5
└── README.md
```

---

## Setup Instructions

### 1. Create and activate virtual environment
```
python3.11 -m venv tf_env
source tf_env/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

---

## Run Inference (CLI)
```
python api/main.py /path/to/xray_image.jpeg
```

---

## Run FastAPI Server
```
uvicorn api.app:app --reload
```

Open in browser:
```
http://127.0.0.1:8000/docs
```

---

## Notes
- Inference preprocessing matches training preprocessing
- Model is loaded once at API startup
- Focus is on end-to-end ML deployment

---

## Model Architecture

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.3),
])

model = Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='tanh'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```

**Loss Function**: Binary Cross-Entropy  
**Optimizer**: Adam  
**Metrics**: Accuracy, Precision, Recall  

---

## Results (Best Run)

- **Accuracy**: 0.90  
- **Normal Recall**: 0.86  
- **Pneumonia Recall**: 0.92  

**Confusion Matrix**
```
[[202  32]
 [ 32 358]]
```

Balanced performance with high pneumonia sensitivity while still detecting normals reliably.

---

## Key Learnings
- Augmentation was the primary performance booster
- Balanced recall is more important than raw accuracy in medical tasks
- Simpler CNNs generalized better than deeper variants
- Reducing class dominance improved fairness

---

## Future Improvements
- Early stopping and learning-rate scheduling
- Transfer learning (ResNet50, EfficientNet)
- Augmentation and MixUp tuning
- Class-weight optimization for recall parity

