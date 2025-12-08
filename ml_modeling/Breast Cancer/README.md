# 🎗️ Breast Cancer Diagnosis Prediction

---

## Project Overview
It is given by Kaggle from the UCI Machine Learning Repository, in one of its challenges. It is a dataset of Breast Cancer patients with Malignant and Benign tumors. Logistic Regression is used to predict whether the given patient has Malignant
or a Benign tumor based on the attributes in the given dataset. But in this project, we use various models like Random Forest, SVM, K-means, XGBoost, and ... to compare all models, evaluate their results, and finally combine their results to achieve the best prediction. our dataset is downloaded from Kaggle.


This project is a comprehensive analysis aimed at developing a **robust machine learning model** to accurately predict **breast cancer diagnosis** (either **benign** or **malignant**). It is a **binary classification** problem utilizing the Wisconsin Breast Cancer (Diagnostic) dataset. The approach involves training and evaluating a wide array of classification models, culminating in the creation of an **ensemble model** to maximize predictive accuracy.

---

## Dataset

The analysis is performed on the **Wisconsin Breast Cancer (Diagnostic) dataset**, loaded from the file `breast_cancer_combined.csv`.

### Data Characteristics

* **Total Records**: 1284 rows
* **Features**: 33 columns, including 30 real-valued features computed from digitized images of a fine needle aspirate (FNA) of a breast mass.

### Key Features

The features are categorized into mean, standard error (se), and "worst" (largest) values for ten core characteristics of the cell nuclei:

* **Target Variable**: **`diagnosis`** (M = Malignant, B = Benign)
* **Core Features**: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, and `fractal_dimension`

---

## Dependencies

The project relies on a comprehensive set of Python libraries for its full machine learning workflow:

* **Pandas & NumPy**: For efficient data manipulation and numerical operations.
* **Matplotlib & Seaborn**: For visualization, including plotting **Confusion Matrices**.
* **Scikit-learn (`sklearn`)**: Provides all essential machine learning tools:
    * `StandardScaler` (Feature scaling)
    * `train_test_split` (Data splitting)
    * `LogisticRegression`, `DecisionTreeClassifier`, `SVC`, `LinearSVC`, `RandomForestClassifier`, `GaussianNB` (Individual models)
    * `accuracy_score`, `f1_score`, `confusion_matrix` (Evaluation metrics)
* **XGBoost**: For implementing the **eXtreme Gradient Boosting** algorithms (`XGBClassifier` and `XGBRFClassifier`).

---

## Methodology and Approaches Used

The project was executed through a structured pipeline to ensure a thorough analysis and robust model building:

### 1. Exploratory Data Analysis (EDA)

* **Data Loading**: Loaded the dataset from `breast_cancer_combined.csv`.
* **Data Inspection**: Verified the dimensions (`df.shape` resulting in (1284, 33)) and displayed the first few rows (`df.head()`) to understand the structure and data types.

### 2. Data Preprocessing

* **Feature Removal**: Columns that were non-predictive or redundant, such as **`id`** and the empty **`Unnamed: 32`** column, were dropped.
* **Target Encoding**: The categorical `diagnosis` column (M/B) was converted to a numerical representation (binary) for model training.
* **Data Splitting**: The dataset was divided into training and testing sets using **`train_test_split`**.
* **Feature Scaling**: **`StandardScaler`** was applied to standardize the numerical features, preventing features with large magnitudes from dominating the learning process.

### 3. Model Training and Comparison

A wide range of classification algorithms was trained and evaluated to identify the best performer for this diagnostic task:

| Model Type | Algorithm |
| :--- | :--- |
| **Support Vector Machines** | `SVC()`, `LinearSVC()` |
| **Ensemble/Boosting** | `RandomForestClassifier()`, `XGBClassifier()`, `XGBRFClassifier()` |
| **Linear Models** | `LogisticRegression()` |
| **Decision Tree** | `DecisionTreeClassifier()` |

### 4. Ensemble Modeling (Voting Classifier)

* An **Ensemble Model** was constructed using a **Voting Classifier** mechanism.
* This approach combines the predictions of multiple individual high-performing models (like Logistic Regression and Linear SVC) to make a final, more robust prediction. The final decision is based on a vote (mode) of the individual model predictions.

---

## Results and Conclusion

The model comparison demonstrated exceptionally high performance across several advanced classifiers:

| Model | Train Accuracy | Test Accuracy | Test F1 Score (Malignant) |
| :--- | :--- | :--- | :--- |
| **XGBClassifier** | 1.0000 | **1.0000** | **1.0000** |
| **RandomForestClassifier** | 1.0000 | **0.9922** | 0.9900 |
| **LogisticRegression** | 0.9581 | **0.9689** | 0.9596 |
| **LinearSVC** | 0.9523 | **0.9611** | 0.9515 |
| **Ensemble Model** | N/A | **0.9969** | 0.9965 |

The **XGBoost Classifier** achieved perfect accuracy (1.0000) on the test set. Furthermore, the **Ensemble Model** (Voting Classifier) showed strong generalized performance with a Test Accuracy of **99.69%**.

This project confirms the effectiveness of machine learning in medical diagnostics, with ensemble and gradient boosting techniques proving particularly strong in accurately differentiating between benign and malignant tumors. Future work could include hyperparameter tuning and exploring deep learning models.

I gave this project to my mom, who is dealing right now with this disease...:))
hope the healing...
2/27/2025
