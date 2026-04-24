import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 128
MODEL_PATH = "./tensorboard/cifar10_resnet_aug_tf2/best_model.keras"

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Same preprocessing used during training
x_test = x_test.astype("float32") / 255.0

cifar10_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
cifar10_std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

x_test = (x_test - cifar10_mean) / cifar10_std

true_classes = y_test.flatten()

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Predict
predictions = model.predict(x_test, batch_size=BATCH_SIZE)
pred_classes = np.argmax(predictions, axis=1)

# Accuracy
correct = pred_classes == true_classes
accuracy = correct.mean() * 100

print(f"\nAccuracy on Test Set: {accuracy:.2f}%")
print(f"Correct Predictions: {correct.sum()} / {len(x_test)}")

# Per-class metrics
print("\nClassification Report:")
print(classification_report(
    true_classes,
    pred_classes,
    target_names=class_names
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))

# Verify all classes are being predicted
unique, counts = np.unique(pred_classes, return_counts=True)

print("\nPredicted Class Distribution:")
for cls, count in zip(unique, counts):
    print(f"{class_names[cls]}: {count}")