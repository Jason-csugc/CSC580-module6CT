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

# CIFAR-10 normalization
cifar10_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
cifar10_std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

x_train = (x_train - cifar10_mean) / cifar10_std
x_test = (x_test - cifar10_mean) / cifar10_std

y_train_cat = ks.utils.to_categorical(y_train, 10)
y_test_cat = ks.utils.to_categorical(y_test, 10)

# =========================
# PARAMETERS
# =========================
BATCH_SIZE = 128
EPOCHS = 60
LOG_DIR = "./tensorboard/cifar10_deeper_tf2/"
MODEL_PATH = LOG_DIR + "best_model.keras"

# =========================
# DEEPER CNN MODEL
# =========================
model = ks.Sequential([

    ks.layers.Input(shape=(32, 32, 3)),

    # Block 1
    ks.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.MaxPooling2D((2, 2)),
    ks.layers.Dropout(0.25),

    # Block 2
    ks.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.MaxPooling2D((2, 2)),
    ks.layers.Dropout(0.30),

    # Block 3
    ks.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.MaxPooling2D((2, 2)),
    ks.layers.Dropout(0.40),

    # Block 4
    ks.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.GlobalAveragePooling2D(),

    # Classifier
    ks.layers.Dense(256, activation="relu"),
    ks.layers.BatchNormalization(),
    ks.layers.Dropout(0.50),
    ks.layers.Dense(10, activation="softmax")
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=ks.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    ks.callbacks.TensorBoard(log_dir=LOG_DIR),

    ks.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    ks.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),

    ks.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# =========================
# TRAIN MODEL
# =========================
start_time = time()

history = model.fit(
    x_train,
    y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# =========================
# EVALUATE MODEL
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

hours, rem = divmod(time() - start_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

# =========================
# TRAINING TIME
# =========================
end_time = time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
