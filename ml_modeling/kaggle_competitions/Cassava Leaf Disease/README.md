# 🌿 Cassava Leaf Disease Classification 🌿

This project focuses on identifying diseases in cassava plants using deep learning. The repository implements a robust pipeline designed specifically to navigate Kaggle's computational constraints while maximizing model performance.

---

## 🚀 The Two-Notebook Strategy

To optimize the workflow, I utilized a decoupled approach between training and inference:

### 1. Training Notebook (TPU Acceleration)
The **Training Notebook** is the "heavy lifter" of the project.
* **Purpose**: Handles data ingestion, model architecture definition, and high-speed training.
* **Hardware**: Leverages **TPUs (Tensor Processing Units)** to handle large-scale image data and deep architectures efficiently.
* **Output**: Once training is complete, the model is saved in the `.keras` format. This saved model becomes a permanent asset on Kaggle, allowing it to be used as an input for other notebooks without retraining.

### 2. Inference Notebook (GPU & TTA)
The **Inference Notebook** is designed to be lightweight, fast, and reliable for submissions.
* **Purpose**: Loads the pre-trained model and generates the final `submission.csv`.
* **Hardware**: Uses **GPU acceleration** for rapid batch inference.
* **Optimization**: It does **not** perform any training. Instead, it uses **Test-Time Augmentation (TTA)** to boost accuracy by averaging predictions across multiple augmented views of the test images.

---

## 🛠️ Technical Details & Approach

### 🧠 Model Architecture
* **Base Model**: **EfficientNetB3**, known for its excellent balance between parameter efficiency and accuracy.
* **Input Resolution**: Images are processed at a high resolution of **512 x 512** pixels to capture fine-grained leaf details.
* **Preprocessing**: Implements the specific preprocessing required for EfficientNet architectures.

### 🔍 Test-Time Augmentation (TTA) Strategy
To reduce the impact of random variations and improve robustness, the inference pipeline applies 5 augmentation runs per image:
* **Geometric Shifts**: Random rotation (up to 40°), translation, and zooming.
* **Flips**: Both horizontal and vertical flips.
* **Aggregation**: Predictions are averaged across these runs to determine the final class label.

### 📊 Validation & Verification
The inference pipeline includes a custom `check_model_on_train_images` function. This tool:
1.  Samples random images from the training set.
2.  Runs the full TTA inference process.
3.  Visualizes the results with **confidence scores** and compares them to ground truth labels to ensure the model is performing as expected before final submission.

---

## 🔗 Kaggle Resources
* **Dataset**: [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)
* **Fine-Tuned Model**: [Cassava Leaf Model (EfficientNetB3)](https://www.kaggle.com/models/amirmohamadaskari/cassava-leaf-model)
* **Training Notebook**: [EfficientNetB3 Fine-Tuning](https://www.kaggle.com/code/amirmohamadaskari/cassava-leaf-train-efficientnetb3-fine-tuning/edit)