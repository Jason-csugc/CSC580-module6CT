"""Evaluate a trained CIFAR-10 CNN model.

Loads the saved best model, runs predictions on the test set, and reports
accuracy, precision, recall, F1 score, per-class metrics, calibration
error (ECE), and visual plots of top predictions and misclassifications.
"""

import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

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
np.random.seed(21)


BATCH_SIZE = 128
MODEL_PATH = "./tensorboard/cifar10_tf2/best_model.keras"

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]



def load_data():
    """Load and preprocess the CIFAR-10 test split.

    Steps:
        1. Load only the CIFAR-10 test set.
        2. Convert image dtype to float32 and scale pixel values to [0, 1].
        3. Standardize each RGB channel using CIFAR-10 train-set mean/std.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            x_test: Standardized test images with shape (N, 32, 32, 3).
            y_test: Integer class labels with shape (N, 1).
    """
    (_, _), (x_test, y_test) = ks.datasets.cifar10.load_data()

    # Convert to float
    x_test = x_test.astype("float32") / 255.0

    # SAME normalization as training
    # calculated from the training set in train_tf2.py
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std  = [0.2470, 0.2435, 0.2616]

    x_test = (x_test - cifar10_mean) / cifar10_std

    return x_test, y_test


def load_model(model_path):
    """Load a saved Keras model from disk.

    Args:
        model_path (str): Path to the saved .keras model file.

    Returns:
        keras.Model: The loaded model ready for inference.
    """
    print("\nLoading trained model...")
    model = ks.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def run_predictions(model, x_test, batch_size):
    """Run model inference on the test set.

    Args:
        model (keras.Model): Trained model.
        x_test (np.ndarray): Normalized test images.
        batch_size (int): Batch size for inference.

    Returns:
        predictions (np.ndarray): Raw softmax outputs of shape (n, 10).
        pred_classes (np.ndarray): Predicted class indices.
        confidences (np.ndarray): Maximum softmax probability per sample.
    """
    print("\nRunning predictions...")
    predictions = model.predict(x_test, batch_size=batch_size)
    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    return predictions, pred_classes, confidences


def print_accuracy(pred_classes, true_classes, x_test):
    """Compute and print overall test-set accuracy.

    Args:
        pred_classes (np.ndarray): Predicted class indices.
        true_classes (np.ndarray): Ground-truth class indices.
        x_test (np.ndarray): Test images (used only for total count).
    """
    correct = true_classes == pred_classes
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print(f"\nAccuracy on Test-Set: {acc:.2f}% ({correct_numbers} / {len(x_test)})")


def unnormalize(img):
    """Convert a standardized CIFAR-10 image back to displayable RGB values.

    Args:
        img (np.ndarray): Image normalized with CIFAR-10 channel mean/std.

    Returns:
        np.ndarray: Image in [approximately 0, 1] RGB space for plotting.
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])
    return (img * std) + mean


def plot_top_predictions(x_test, pred_classes, confidences, class_names):
    """Plot the top-8 highest-confidence predictions for each class.

    Displays an 8-row × 10-column grid where each column corresponds to one
    CIFAR-10 class, ordered by descending prediction confidence.

    Args:
        x_test (np.ndarray): Normalized test images.
        pred_classes (np.ndarray): Predicted class indices.
        confidences (np.ndarray): Maximum softmax probability per sample.
        class_names (list[str]): Human-readable class labels.
    """
    # Select top 8 highest-confidence images per class
    top_images_per_class = {}
    for class_idx in range(10):
        idxs = np.where(pred_classes == class_idx)[0]
        sorted_idxs = idxs[np.argsort(confidences[idxs])[::-1]]
        top_images_per_class[class_idx] = sorted_idxs[:8]

    rows, cols = 8, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    fig.canvas.manager.set_window_title("Top 8 Predictions per Class (Organized by Category)")

    for col in range(cols):
        for row in range(rows):
            ax = axes[row, col]
            if row < len(top_images_per_class[col]):
                ax.imshow(unnormalize(x_test[top_images_per_class[col][row]]))
            else:
                ax.imshow(np.zeros((32, 32, 3)))
            ax.axis('off')
        axes[0, col].set_title(class_names[col], fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_misclassifications(x_test, pred_classes, true_classes, confidences, class_names, n=40):
    """Plot the most confidently wrong predictions made by the model.

    Displays a grid of misclassified images sorted by descending confidence,
    annotated with the true and predicted labels.

    Args:
        x_test (np.ndarray): Normalized test images.
        pred_classes (np.ndarray): Predicted class indices.
        true_classes (np.ndarray): Ground-truth class indices.
        confidences (np.ndarray): Maximum softmax probability per sample.
        class_names (list[str]): Human-readable class labels.
        n (int): Number of misclassifications to display. Defaults to 40.
    """
    wrong_idxs = np.where(pred_classes != true_classes)[0]
    sorted_wrong = wrong_idxs[np.argsort(confidences[wrong_idxs])[::-1]]

    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.canvas.manager.set_window_title("Most Confident WRONG Predictions (Model Mistakes)")

    for i in range(n):
        ax = axes[i // cols, i % cols]
        idx = sorted_wrong[i]
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[pred_classes[idx]]
        conf = confidences[idx]
        ax.imshow(unnormalize(x_test[idx]))
        ax.set_title(f"T:{true_label}\nP:{pred_label} ({conf:.2f})", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def print_metrics(pred_classes, true_classes):
    """Print macro and weighted precision, recall, F1, and per-class report.

    Args:
        pred_classes (np.ndarray): Predicted class indices.
        true_classes (np.ndarray): Ground-truth class indices.
    """
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

    print("\n=== Per-Class Report ===")
    print(classification_report(true_classes, pred_classes))


def compute_ece(predictions, true_labels, n_bins=15):
    """Compute the Expected Calibration Error (ECE).

    Measures how well the model's predicted confidences align with its actual
    accuracy by binning predictions and comparing average confidence to accuracy
    within each bin.

    Args:
        predictions (np.ndarray): Raw softmax outputs of shape (n, num_classes).
        true_labels (np.ndarray): Ground-truth class indices.
        n_bins (int): Number of confidence bins. Defaults to 15.

    Returns:
        float: The ECE score (lower is better).
    """
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


def main():
    """Orchestrate model evaluation: load data and model, run predictions,
    print metrics, plot visualizations, and report calibration error.
    """
    x_test, y_test = load_data()
    true_classes = y_test.flatten()

    model = load_model(MODEL_PATH)

    predictions, pred_classes, confidences = run_predictions(model, x_test, BATCH_SIZE)

    unique, counts = np.unique(pred_classes, return_counts=True)

    print("\nPredicted Class Distribution:")
    for cls, count in zip(unique, counts):
        print(f"{CLASS_NAMES[cls]}: {count}")

    print_accuracy(pred_classes, true_classes, x_test)
    plot_top_predictions(x_test, pred_classes, confidences, CLASS_NAMES)
    plot_misclassifications(x_test, pred_classes, true_classes, confidences, CLASS_NAMES)
    print_metrics(pred_classes, true_classes)

    ece = compute_ece(predictions, true_classes)
    print("\n=== Calibration ===")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")


if __name__ == "__main__":
    main()
