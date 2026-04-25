# CSC580-module6CT

CSC580 Module 6CT Project

## Product

### Overview

This repository contains a CIFAR-10 image classification workflow built with
TensorFlow and Keras. The project trains a convolutional neural network (CNN)
on CIFAR-10, saves the best-performing checkpoint, and evaluates the trained
model with accuracy, precision, recall, F1 score, calibration error, and
prediction visualizations.

## Features

- `load_data()` in `train_tf2.py`: loads CIFAR-10 train/test splits, scales images, standardizes channels, and one-hot encodes labels
- `build_model()`: defines and compiles the CNN architecture used for training
- `build_callbacks()`: creates TensorBoard and model-checkpoint callbacks
- `train_model()`: fits the CNN on CIFAR-10 with validation tracking
- `evaluate_model()`: computes and prints final test accuracy after training
- `main()` in `train_tf2.py`: orchestrates training, evaluation, and training-time reporting
- `load_data()` in `test_tf2.py`: loads and preprocesses the CIFAR-10 test split for inference
- `load_model()`: restores the saved `.keras` checkpoint for evaluation
- `run_predictions()`: runs batched inference and extracts predicted classes and confidences
- `print_accuracy()`: computes and prints test-set accuracy
- `unnormalize()`: converts standardized images back to displayable RGB values
- `plot_top_predictions()`: shows the highest-confidence predictions per class
- `plot_misclassifications()`: visualizes the most confident incorrect predictions
- `print_metrics()`: reports macro/weighted precision, recall, F1, and a per-class classification report
- `compute_ece()`: computes Expected Calibration Error (ECE)
- `main()` in `test_tf2.py`: orchestrates model loading, inference, metrics, plots, and calibration reporting

## Getting Started

1. Create and activate a virtual environment:

```bash
python3 -m venv mod6ct
source mod6ct/bin/activate
```

2. If you want GPU acceleration, complete the NVIDIA/WSL CUDA setup steps from the [nvidia-gpu repository](https://github.com/Jason-csugc/nvidia-gpu) first. That repo currently includes commands for installing the WSL CUDA 13.2 toolkit and exporting `LD_LIBRARY_PATH` for NVIDIA libraries.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python train_tf2.py
```

5. Evaluate the saved model:

```bash
python test_tf2.py
```

6. (Optional) Launch TensorBoard for training logs:

```bash
tensorboard --logdir tensorboard/cifar10_tf2
```

## Notes

- Designed for coursework use in CSC580 Module 6CT.
- Training uses the CIFAR-10 dataset bundled through `keras.datasets.cifar10`.
- Randomness is controlled with a fixed TensorFlow seed (`21`).
- Image preprocessing includes scaling to `[0, 1]` and per-channel standardization.
- The best model checkpoint is saved to `tensorboard/cifar10_tf2/best_model.keras`.
- TensorFlow logging is reduced and selected NumPy deprecation warnings are suppressed for cleaner console output.
- The evaluation script includes both quantitative metrics and qualitative visual inspection of predictions and failures.

## Outputs

1. Console output:
- Training progress across epochs with validation accuracy and loss
- Final test accuracy after training
- Total training time for the training run
- Test-set accuracy during evaluation
- Predicted class distribution across the test set
- Macro and weighted precision, recall, and F1 scores
- Per-class classification report
- Expected Calibration Error (ECE)

2. Artifacts:
- Best trained model saved at `tensorboard/cifar10_tf2/best_model.keras`
- TensorBoard event files in `tensorboard/cifar10_tf2/`
- Matplotlib figure showing top-confidence predictions for each CIFAR-10 class

<img width="2012" height="1270" alt="image" src="https://github.com/user-attachments/assets/48e72ddb-94bc-4033-9891-066fb3bea497" />

  
- Matplotlib figure showing the most confident misclassifications

<img width="1610" height="1073" alt="image" src="https://github.com/user-attachments/assets/5370b6cb-5667-468d-b39a-c4bc6714e312" />


## Additional Links

- [Code](https://github.com/Jason-csugc/CSC580-module6CT)
- [Issues](https://github.com/Jason-csugc/CSC580-module6CT/issues)
- [Pull requests](https://github.com/Jason-csugc/CSC580-module6CT/pulls)
- [Actions](https://github.com/Jason-csugc/CSC580-module6CT/actions)
- [Projects](https://github.com/Jason-csugc/CSC580-module6CT/projects)
- [Security and quality](https://github.com/Jason-csugc/CSC580-module6CT/security)
