import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from preprocess import test_ds

# Load trained model
model = tf.keras.models.load_model("cnn_pneumonia.h5")

# Collect predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend((preds > 0.5).astype("int32").flatten())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal","Pneumonia"]))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)


fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

