import os
import logging
# import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import keras as ks
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

tf.random.set_seed(21)
np.random.seed(21)

# =========================
# PARAMETERS
# =========================
BATCH_SIZE = 128
MODEL_PATH = "./tensorboard/cifar10_deeper_tf2/best_model.keras"

# =========================
# LOAD CIFAR-10 DATA
# =========================
(_, _), (x_test, y_test) = ks.datasets.cifar10.load_data()

# Normalize (same as training)
x_test = x_test / 255.0

# Convert labels to categorical (only for consistency)
y_test_cat = ks.utils.to_categorical(y_test, 10)

# =========================
# LOAD TRAINED MODEL
# =========================
print("\nLoading trained model...")
model = ks.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# =========================
# PREDICTION
# =========================
print("\nRunning predictions...")

predictions = model.predict(x_test, batch_size=BATCH_SIZE)

# Convert softmax outputs → class indices
predicted_class = np.argmax(predictions, axis=1)

# Flatten y_test for comparison
true_class = y_test.flatten()

# =========================
# ACCURACY CALCULATION
# =========================
correct = (true_class == predicted_class)
acc = correct.mean() * 100
correct_numbers = correct.sum()

print("\nAccuracy on Test-Set: {0:.2f}% ({1} / {2})".format(
    acc, correct_numbers, len(x_test)
))

# =========================
# CIFAR-10 CLASS NAMES
# =========================
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# =========================
# PREDICT ON TEST SET
# =========================
predictions = model.predict(x_test, batch_size=128)

pred_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)
true_classes = y_test.flatten()

# =========================
# SELECT TOP 8 PER CLASS
# =========================
top_images_per_class = {}

for class_idx in range(10):
    idxs = np.where(pred_classes == class_idx)[0]
    sorted_idxs = idxs[np.argsort(confidences[idxs])[::-1]]
    top_images_per_class[class_idx] = sorted_idxs[:8]

# =========================
# PLOT (8 rows x 10 columns)
# =========================
rows = 8
cols = 10

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))

for col in range(cols):  # each column = class
    for row in range(rows):  # each row = image
        ax = axes[row, col]

        if row < len(top_images_per_class[col]):
            img_idx = top_images_per_class[col][row]
            ax.imshow(x_test[img_idx])
        else:
            ax.imshow(np.zeros((32,32,3)))

        ax.axis('off')

    # Set column titles (top row only)
    axes[0, col].set_title(class_names[col], fontsize=12)

plt.suptitle("Top 8 Predictions per Class (Organized by Category)", fontsize=18)
plt.tight_layout()
plt.show()

# predictions = model.predict(x_test)
# pred_classes = np.argmax(predictions, axis=1)
# confidences = np.max(predictions, axis=1)
# true_classes = y_test.flatten()
# =========================
# FIND MISCLASSIFICATIONS
# =========================
wrong_idxs = np.where(pred_classes != true_classes)[0]

# Sort mistakes by confidence (descending → worst mistakes first)
sorted_wrong = wrong_idxs[np.argsort(confidences[wrong_idxs])[::-1]]

# Number of images to display
N = 40

# =========================
# CIFAR-10 CLASS NAMES
# =========================
# class_names = [
#     'airplane', 'automobile', 'bird', 'cat', 'deer',
#     'dog', 'frog', 'horse', 'ship', 'truck'
# ]

# =========================
# PLOT GRID
# =========================
rows = 5
cols = 8

fig, axes = plt.subplots(rows, cols, figsize=(16, 10))

for i in range(N):
    ax = axes[i // cols, i % cols]
    
    idx = sorted_wrong[i]
    img = x_test[idx]
    
    true_label = class_names[true_classes[idx]]
    pred_label = class_names[pred_classes[idx]]
    conf = confidences[idx]
    
    ax.imshow(img)
    ax.set_title(f"T:{true_label}\nP:{pred_label} ({conf:.2f})", fontsize=8)
    ax.axis('off')

plt.suptitle("Most Confident WRONG Predictions (Model Mistakes)", fontsize=16)
plt.tight_layout()
plt.show()

# Macro (treats all classes equally)
precision_macro = precision_score(true_classes, pred_classes, average='macro')
recall_macro = recall_score(true_classes, pred_classes, average='macro')
f1_macro = f1_score(true_classes, pred_classes, average='macro')

# Weighted (accounts for class frequency)
precision_weighted = precision_score(true_classes, pred_classes, average='weighted')
recall_weighted = recall_score(true_classes, pred_classes, average='weighted')
f1_weighted = f1_score(true_classes, pred_classes, average='weighted')

print("\n=== Overall Metrics ===")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro):    {recall_macro:.4f}")
print(f"F1 Score (Macro):  {f1_macro:.4f}")

print(f"\nPrecision (Weighted): {precision_weighted:.4f}")
print(f"Recall (Weighted):    {recall_weighted:.4f}")
print(f"F1 Score (Weighted):  {f1_weighted:.4f}")

# Per-class breakdown
print("\n=== Per-Class Report ===")
print(classification_report(true_classes, pred_classes))

def compute_ece(predictions, true_labels, n_bins=15):
    confidences = np.max(predictions, axis=1)
    predictions_cls = np.argmax(predictions, axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions_cls[in_bin] == true_labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

ece = compute_ece(predictions, true_classes)

print("\n=== Calibration ===")
print(f"Expected Calibration Error (ECE): {ece:.4f}")