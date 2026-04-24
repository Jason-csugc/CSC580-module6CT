import os
import logging
# import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import keras as ks
import numpy as np
from time import time
import matplotlib.pyplot as plt

# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Set seed for reproducibility
tf.random.set_seed(21)

# =========================
# LOAD CIFAR-10 DATA
# =========================
(x_train, y_train), (x_test, y_test) = ks.datasets.cifar10.load_data()

# Normalize images (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = ks.utils.to_categorical(y_train, 10)
y_test = ks.utils.to_categorical(y_test, 10)

# =========================
# PARAMETERS
# =========================
BATCH_SIZE = 128
EPOCHS = 20
LOG_DIR = "./tensorboard/cifar10_tf2/"

# =========================
# BUILD CNN MODEL
# =========================
model = ks.models.Sequential([

    # Block 1
    ks.Input(shape=(32, 32, 3)),
    ks.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(32, (3,3), activation='relu'),
    ks.layers.MaxPooling2D((2,2)),
    ks.layers.Dropout(0.25),

    # Block 2
    ks.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(64, (3,3), activation='relu'),
    ks.layers.MaxPooling2D((2,2)),
    ks.layers.Dropout(0.25),

    # Block 3
    ks.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    ks.layers.BatchNormalization(),
    ks.layers.MaxPooling2D((2,2)),
    ks.layers.Dropout(0.25),

    # Fully Connected
    ks.layers.Flatten(),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dropout(0.5),
    ks.layers.Dense(10, activation='softmax')
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=ks.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.summary()

# =========================
# CALLBACKS (TensorBoard + Checkpoint)
# =========================
tensorboard = ks.callbacks.TensorBoard(log_dir=LOG_DIR)

checkpoint = ks.callbacks.ModelCheckpoint(
    filepath=LOG_DIR + "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN MODEL
# =========================
start_time = time()

history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard, checkpoint],
    verbose=1
)

# =========================
# EVALUATE MODEL
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\nFinal Test Accuracy: {:.2f}%".format(test_acc * 100))

# =========================
# TRAINING TIME
# =========================
end_time = time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)

print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

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

# Predicted class + confidence
pred_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

true_classes = y_test.flatten()

# =========================
# SELECT TOP 8 PER CLASS
# =========================
top_images_per_class = {}

for class_idx in range(10):
    # Get indices where prediction == class
    idxs = np.where(pred_classes == class_idx)[0]

    # Sort those by confidence (descending)
    sorted_idxs = idxs[np.argsort(confidences[idxs])[::-1]]

    # Take top 8
    top_images_per_class[class_idx] = sorted_idxs[:8]

# =========================
# PLOT GRID (10 rows x 8 cols)
# =========================
fig, axes = plt.subplots(10, 8, figsize=(16, 20))

for class_idx in range(10):
    for i in range(8):
        ax = axes[class_idx, i]

        if i < len(top_images_per_class[class_idx]):
            img_idx = top_images_per_class[class_idx][i]
            ax.imshow(x_test[img_idx])

            # Show confidence
            conf = confidences[img_idx]
            ax.set_title(f"{conf:.2f}", fontsize=8)
        else:
            ax.imshow(np.zeros((32,32,3)))

        ax.axis('off')

    # Label each row with class name
    axes[class_idx, 0].set_ylabel(class_names[class_idx], fontsize=12)

plt.suptitle("Top 8 Highest-Confidence Predictions Per Class", fontsize=18)
plt.tight_layout()
plt.show()

# =========================
# FIND MISCLASSIFICATIONS
# =========================
# wrong_idxs = np.where(pred_classes != true_classes)[0]

# Sort mistakes by confidence (descending → worst mistakes first)
# sorted_wrong = wrong_idxs[np.argsort(confidences[wrong_idxs])[::-1]]

# Number of images to display
# N = 40

# =========================
# PLOT GRID
# =========================
# rows = 5
# cols = 8

# fig, axes = plt.subplots(rows, cols, figsize=(16, 10))

# for i in range(N):
#     ax = axes[i // cols, i % cols]
    
#     idx = sorted_wrong[i]
#     img = x_test[idx]
    
#     true_label = class_names[true_classes[idx]]
#     pred_label = class_names[pred_classes[idx]]
#     conf = confidences[idx]
    
#     ax.imshow(img)
#     ax.set_title(f"T:{true_label}\nP:{pred_label} ({conf:.2f})", fontsize=8)
#     ax.axis('off')

# plt.suptitle("Most Confident WRONG Predictions (Model Mistakes)", fontsize=16)
# plt.tight_layout()
# plt.show()
