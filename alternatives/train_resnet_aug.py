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
EPOCHS = 80
LOG_DIR = "./tensorboard/cifar10_resnet_aug_tf2/"
MODEL_PATH = LOG_DIR + "best_model.keras"

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = ks.Sequential([
    ks.layers.RandomFlip("horizontal"),
    ks.layers.RandomRotation(0.08),
    ks.layers.RandomZoom(0.10),
    ks.layers.RandomTranslation(0.08, 0.08),
], name="data_augmentation")

# =========================
# RESIDUAL BLOCK
# =========================
def residual_block(x, filters, stride=1, dropout_rate=0.0):
    shortcut = x

    x = ks.layers.Conv2D(
        filters, 
        kernel_size=(3, 3), 
        strides=stride, 
        padding="same",
        use_bias=False
    )(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation("relu")(x)

    x = ks.layers.Conv2D(
        filters, 
        kernel_size=(3, 3), 
        strides=1, 
        padding="same",
        use_bias=False
    )(x)
    x = ks.layers.BatchNormalization()(x)

    # Match shortcut dimensions when filters or spatial size changes
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = ks.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)
        shortcut = ks.layers.BatchNormalization()(shortcut)

    x = ks.layers.Add()([x, shortcut])
    x = ks.layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = ks.layers.Dropout(dropout_rate)(x)

    return x

# =========================
# BUILD RESIDUAL CNN
# =========================
inputs = ks.layers.Input(shape=(32, 32, 3))

x = data_augmentation(inputs)

# Initial convolution
x = ks.layers.Conv2D(
    64,
    kernel_size=(3, 3),
    padding="same",
    use_bias=False
)(x)
x = ks.layers.BatchNormalization()(x)
x = ks.layers.Activation("relu")(x)

# Residual stages
x = residual_block(x, 64, stride=1, dropout_rate=0.10)
x = residual_block(x, 64, stride=1, dropout_rate=0.10)

x = residual_block(x, 128, stride=2, dropout_rate=0.20)
x = residual_block(x, 128, stride=1, dropout_rate=0.20)

x = residual_block(x, 256, stride=2, dropout_rate=0.30)
x = residual_block(x, 256, stride=1, dropout_rate=0.30)

x = residual_block(x, 512, stride=2, dropout_rate=0.40)
x = residual_block(x, 512, stride=1, dropout_rate=0.40)

# Classifier
x = ks.layers.GlobalAveragePooling2D()(x)
x = ks.layers.Dense(256, activation="relu")(x)
x = ks.layers.BatchNormalization()(x)
x = ks.layers.Dropout(0.50)(x)

outputs = ks.layers.Dense(10, activation="softmax")(x)

model = ks.Model(inputs=inputs, outputs=outputs)

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
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    ks.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
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

print("\nFinal Test Accuracy: {:.2f}%".format(test_acc * 100))

hours, rem = divmod(time() - start_time, 3600)
minutes, seconds = divmod(rem, 60)

print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours), int(minutes), seconds
))