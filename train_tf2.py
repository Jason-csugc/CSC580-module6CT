"""Train and evaluate a CNN on CIFAR-10 with TensorFlow/Keras.

This module loads and preprocesses CIFAR-10, builds a convolutional neural
network, trains with TensorBoard logging and best-checkpoint saving, then
evaluates final test accuracy and prints total training time.
"""

import os
import logging
import warnings
from time import time
import numpy as np

# Disable oneDNN optimizations for consistent performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Ignore NumPy 2.4 deprecation warning triggered by CIFAR-10 pickle loading.
warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as Python or NumPy boolean.*",
    category=Warning,
)

# pylint: disable=wrong-import-position
import tensorflow as tf
import keras as ks

tf.random.set_seed(21)

# =========================
# PARAMETERS
# =========================
BATCH_SIZE = 128
EPOCHS = 20
LOG_DIR = "./tensorboard/cifar10_tf2/"
MODEL_PATH = LOG_DIR + "best_model.keras"


# =========================
# SUPPORTING FUNCTIONS
# =========================

def load_data():
    """Load and preprocess the CIFAR-10 train/test splits.

    Steps:
        1. Load CIFAR-10 training and test data.
        2. Convert image dtype to float32 and scale pixel values to [0, 1].
        3. Standardize each RGB channel with CIFAR-10 mean/std statistics.
        4. One-hot encode labels into 10 classes.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_train: Standardized training images, shape (N_train, 32, 32, 3).
            y_train: One-hot training labels, shape (N_train, 10).
            x_test: Standardized test images, shape (N_test, 32, 32, 3).
            y_test: One-hot test labels, shape (N_test, 10).
    """
    (x_train, y_train), (x_test, y_test) = ks.datasets.cifar10.load_data()

    # Convert to float
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CIFAR-10 channel statistics

    print("Mean:", np.mean(x_train, axis=(0,1,2)))
    print("Std:", np.std(x_train, axis=(0,1,2)))

    cifar10_mean = np.mean(x_train, axis=(0,1,2))
    cifar10_std  = np.std(x_train, axis=(0,1,2))

    # Normalize (broadcasts across image)
    x_train = (x_train - cifar10_mean) / cifar10_std
    x_test  = (x_test  - cifar10_mean) / cifar10_std

    # One-hot encode labels
    y_train = ks.utils.to_categorical(y_train, 10)
    y_test  = ks.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model():
    """Build and compile the CNN model.

    Architecture: three convolutional blocks with batch normalization and
    dropout, followed by a fully connected classifier head.

    Returns:
        keras.Sequential: Compiled model ready for training.
    """
    model = ks.models.Sequential([

        # Block 1
        ks.Input(shape=(32, 32, 3)),
        ks.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2D(32, (3, 3), activation='relu'),
        ks.layers.MaxPooling2D((2, 2)),
        ks.layers.Dropout(0.25),

        # Block 2
        ks.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2D(64, (3, 3), activation='relu'),
        ks.layers.MaxPooling2D((2, 2)),
        ks.layers.Dropout(0.25),

        # Block 3
        ks.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        ks.layers.BatchNormalization(),
        ks.layers.MaxPooling2D((2, 2)),
        ks.layers.Dropout(0.25),

        # Fully Connected
        ks.layers.Flatten(),
        ks.layers.Dense(128, activation='relu'),
        ks.layers.Dropout(0.5),
        ks.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_callbacks(log_dir, model_path):
    """Create training callbacks for TensorBoard logging and model checkpointing.

    Args:
        log_dir (str): Directory for TensorBoard event files.
        model_path (str): File path to save the best model checkpoint.

    Returns:
        list[keras.callbacks.Callback]: List of configured callbacks.
    """
    tensorboard = ks.callbacks.TensorBoard(log_dir=log_dir)

    checkpoint = ks.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    return [tensorboard, checkpoint]


def train_model(model, x_train, y_train, x_test, y_test, callbacks, batch_size, epochs):
    """Fit the model on training data and return the training history.

    Args:
        model (keras.Model): Compiled model to train.
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): One-hot encoded training labels.
        x_test (np.ndarray): Validation images.
        y_test (np.ndarray): One-hot encoded validation labels.
        callbacks (list): Keras callbacks to use during training.
        batch_size (int): Number of samples per gradient update.
        epochs (int): Maximum number of training epochs.

    Returns:
        keras.callbacks.History: Training history object.
    """
    return model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )


def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test set and print accuracy.

    Args:
        model (keras.Model): Trained model.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): One-hot encoded test labels.

    Returns:
        float: Test accuracy in percent.
    """
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    return test_acc * 100


def main():
    """Run the end-to-end training workflow.

    Loads data, builds and summarizes the model, configures callbacks, trains,
    evaluates on the test set, and reports elapsed training time.

    Returns:
        None
    """
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()
    model.summary()

    callbacks = build_callbacks(LOG_DIR, MODEL_PATH)

    start_time = time()
    train_model(model, x_train, y_train, x_test, y_test, callbacks, BATCH_SIZE, EPOCHS)

    evaluate_model(model, x_test, y_test)

    hours, rem = divmod(time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
