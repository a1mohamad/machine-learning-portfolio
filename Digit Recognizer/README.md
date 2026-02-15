# Digit Recognizer: A Comprehensive Walkthrough
### 🖋️ Handwritten Digit Classification using Convolutional Neural Networks (CNN)

This repository details the process of building and training a deep learning model to recognize handwritten digits from the famous [MNIST dataset](https://www.kaggle.com/c/digit-recognizer). The project utilizes **TensorFlow** and **Keras** to implement a robust classification pipeline.

## 🚀 Project Overview
The goal is to accurately identify digits (0-9) based on 28x28 grayscale images. This implementation focuses on high accuracy through rigorous preprocessing and a robust cross-validation strategy.

### 📝 Workflow
1.  **Data Loading & Exploration**: Inspecting the MNIST pixel data and checking for distribution balance.
2.  **Data Visualization**: Reshaping 1D pixel arrays into 2D images to verify data integrity.
3.  **Preprocessing Pipeline**: Normalizing and reshaping data for CNN compatibility.
4.  **CNN Architecture**: Designing a multi-layered convolutional network.
5.  **Cross-Validation**: Training across multiple folds to ensure generalization.
6.  **Submission**: Generating predictions for the Kaggle test set.

## 🔬 Technical Implementation Details

### 1. Data Preprocessing Pipeline
To streamline the input for the model, a sequential preprocessing pipeline was created:
* **Reshaping**: Flat vectors of 784 pixels are reshaped back into their native (28, 28, 1) image format.
* **Normalization**: Pixel values are rescaled from [0, 255] to [0, 1] using `Rescaling(1./255)` to help the model converge faster.

### 2. Model Architecture (CNN)
The model is built using the Keras Functional API, featuring:
* **Convolutional Layer (`Conv2D`)**: 32 filters of size 3x3 with `ReLU` activation to detect edges and patterns.
* **Max Pooling (`MaxPooling2D`)**: Downsamples feature maps to reduce spatial dimensions and prevent overfitting.
* **Dense Layers**: A fully connected layer with 128 neurons followed by a 10-neuron output layer (one for each digit).
* **Numerical Stability**: The output layer uses a linear activation combined with `SparseCategoricalCrossentropy(from_logits=True)` for better performance.

### 3. Training & Validation Strategy
* **Stratified K-Fold CV**: Used 10-fold cross-validation to ensure that each digit class is proportionally represented in every training fold.
* **Reproducibility**: Global random seeds (`SEED = 28`) were set for both NumPy and TensorFlow to ensure consistent results across different runs.
* **Performance Optimization**: Utilized `tf.data` with `AUTOTUNE` prefetching and shuffling for an efficient training data pipeline.

## 📊 Evaluation & Results
The notebook tracks accuracy and loss history for each fold, providing a clear view of the model's learning curve and stability. Final predictions are exported to a `submission.csv` file formatted for Kaggle.

---
**Keywords**: TensorFlow, Keras, CNN, MNIST, Computer Vision, Cross-Validation.