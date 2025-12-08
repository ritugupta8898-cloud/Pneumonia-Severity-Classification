# Pneumonia X-Ray Classification (CNN)

This project classifies Chest X-Rays into:

0 → NORMAL
1 → PNEUMONIA

The final CNN model was trained from scratch using TensorFlow/Keras, and improved through architecture tuning, class balancing, and most importantly data augmentation, which significantly boosted recall balance and overall performance.

------------------------------------------------------------
📂 Project Structure
------------------------------------------------------------

Pneumonia-Severity-Classification/
│── data/
│   ├── raw/
│   ├── processed/
│
│── src/
│   ├── preprocess.py
│   ├── train_cnn.py
│   ├── evaluate.py
│
│── models/
│   ├── cnn_pneumonia.h5     ← final trained model
│
└── README.md

------------------------------------------------------------
🧠 Model Architecture
------------------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.3),
])

model = Sequential([
    data_augmentation,
    layers.Conv2D(32,3,activation='relu',padding='same',input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='tanh'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

Loss Function  : Binary Cross-Entropy
Optimizer      : Adam
Metrics        : Accuracy, Precision, Recall

------------------------------------------------------------
📊 Results (Best Run)
------------------------------------------------------------

Accuracy       : 0.90
Normal Recall  : 0.86
Pneumonia Recall : 0.92

Confusion Matrix:
[[202  32]
 [ 32 358]]

Balanced performance — high pneumonia sensitivity while still detecting normals reliably.

------------------------------------------------------------
🔥 Key Learnings
------------------------------------------------------------

• Augmentation was the main performance booster  
• Balanced recall > raw accuracy for medical use  
• Simple models generalize better than deeper ones  
• Reduced pneumonia dominance → restored fairness

------------------------------------------------------------
🚀 Future Improvements
------------------------------------------------------------

• Add early stopping + LR scheduler  
• Transfer Learning (ResNet50, EfficientNet) to push accuracy further  
• Mixup/augmentation tuning for even more robustness  
• Class-weight tuning to match recalls perfectly

------------------------------------------------------------

