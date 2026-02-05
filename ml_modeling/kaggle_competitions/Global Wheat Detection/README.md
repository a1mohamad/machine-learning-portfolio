# 🌾 Global Wheat Detection: KerasCV YOLOv8 Stratified Workflow

This project presents a high-performance solution for the **Global Wheat Detection Kaggle Competition**, designed to localize wheat heads across diverse agricultural environments. The architecture leverages the **YOLOv8** model via the **KerasCV** API, featuring a multi-phase training regime and a robust inference pipeline with Test-Time Augmentation (TTA).

---

## 🏗️ Project Architecture & Methodology

The solution is divided into two distinct technical phases to ensure both model convergence and maximum predictive accuracy.

### Phase I: Stratified Training Optimization
The training process (`final_wheat_detection-YOLOV8_Fine-Tuning.ipynb`) utilizes a **three-phase optimization strategy** to manage the complexity of dense object detection:
* **Warmup Phase:** A short initial training period with a controlled learning rate increase to stabilize model weights and prevent early-stage divergence.
* **Mid-Tune Phase:** The primary training period utilizing an optimized **Cosine Decay** learning rate schedule for comprehensive feature extraction.
* **Fine-Tune Phase:** A final, low-learning-rate refinement stage dedicated to maximizing predictive accuracy and fine-tuning the detector's sensitivity.

### Phase II: Robust Inference Pipeline
The inference process (`wheat-detection-loading-yolov8-inference-with-tta.ipynb`) focuses on squeezing every bit of performance from the trained artifacts:
* **Test-Time Augmentation (TTA):** The pipeline applies **8 unique transformations** (including horizontal/vertical flips and color jittering) to each test image, allowing the model to detect wheat heads from multiple perspectives.
* **Weighted Boxes Fusion (WBF):** Instead of standard Non-Maximum Suppression (NMS), this project implements **WBF** via the `ensemble_boxes` library. This algorithm merges overlapping predictions from the TTA passes by calculating a weighted average of coordinates, significantly reducing localization errors in dense clusters.

---

## 🛠️ Key Technical Features

* **Focal Loss Implementation:** The model utilizes **Focal Loss** to address the class imbalance between the background and the dense foreground objects (wheat heads). This ensures the model focuses on "hard" examples during training, improving detection accuracy in cluttered field conditions.
* **Hardware-Agnostic Distribution:** The system automatically detects and initializes the best available **TensorFlow Strategy**, supporting **TPUStrategy** (for TPUs), **MirroredStrategy** (for multi-GPU), and standard CPU fallback.
* **Negative Sample Strategy:** The training set explicitly integrates images with no annotations (true negatives) to train the model to ignore background clutter like soil and leaves, which constitutes approximately 1% of the total dataset.
* **Dense Detection Tuning:** Optimized for images containing an average of **43 wheat heads**, with many containing over 100, where objects are typically small (~60x60 pixels in a 1024x1024 frame).
* **Metric Compliance:** Performance is evaluated using **COCO mean Average Precision (mAP)** calculated through the `cocometrics` library.

---

## 📦 System Requirements

### Core Frameworks
* **TensorFlow 2.15+**
* **KerasCV** (YOLOv8 backbone and detector)
* **Keras** (.keras artifact management)

### Specialized Packages
* **ensemble-boxes:** Essential for the Weighted Boxes Fusion (WBF) post-processing.
* **cocometrics:** For standardized competition metric evaluation.
* **scikit-learn:** For stratified data splitting.

### Utilities
* **OpenCV (cv2):** For image processing and coordinate transformations.
* **Pandas & NumPy:** For data management and metadata handling.
* **Matplotlib & Seaborn:** For exploratory data analysis and visual verification.

---

## 📍 Resources
* **Competition Link:** [Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection)
* **Trained Model Weights:** [Wheat Detection Models on Kaggle](https://www.kaggle.com/models/amirmohamadaskari/wheat-detection)