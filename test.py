import os
import logging
# import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import keras as ks

# =========================
# PARAMETERS
# =========================
BATCH_SIZE = 128
MODEL_PATH = "./tensorboard/cifar10_tf2/best_model.keras"

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
