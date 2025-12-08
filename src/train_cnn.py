import tensorflow as tf
from tensorflow.keras import layers, Sequential

from preprocess import train_ds, val_ds, test_ds


from tensorflow.keras import layers, Sequential
import tensorflow as tf
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.15),
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
    layers.Dense(128, activation='tanh'),   # gentle, balanced
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight={0:3.0, 1:1.0},  
)

model.save("cnn_pneumonia.h5")
print("Model saved as cnn_pneumonia.h5")
