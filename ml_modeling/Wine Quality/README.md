# 🍷 Wine Quality Prediction

---

## Project Overview

This project focuses on building a machine learning classification model to predict the quality of wine based on its physicochemical properties. The process involves **Exploratory Data Analysis (EDA)**, **comprehensive data preprocessing**, and **training several state-of-the-art machine learning models** to achieve high predictive accuracy.

---

## Dataset

The analysis uses the **`wine_quality.csv`** dataset, which contains 6497 entries and 13 columns.

### Key Features (Physicochemical Properties):

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `type` | Object (Categorical) | White or Red wine |
| `fixed acidity` | Float | Most fixed acids are non-volatile |
| `volatile acidity` | Float | The amount of acetic acid, which can lead to an unpleasant, vinegar taste |
| `citric acid` | Float | Adds 'freshness' and flavor to wines |
| `residual sugar` | Float | The amount of sugar remaining after fermentation |
| `chlorides` | Float | The amount of salt in the wine |
| `free sulfur dioxide` | Float | The free form of SO2 exists in equilibrium between molecular SO2 and bisulfite ion |
| `total sulfur dioxide` | Float | Total free and bound forms of SO2 |
| `density` | Float | A measure of mass per volume |
| `pH` | Float | Acidity/alkalinity level |
| `sulphates` | Float | A wine additive that contributes to SO2 levels |
| `alcohol` | Float | The alcohol content of the wine |
| `quality` | Integer | Wine quality rating (3-9) |

---

## Dependencies

The project relies on the following Python libraries:

* **Pandas**: For efficient data handling and manipulation.
* **NumPy**: For working with arrays and numerical operations.
* **Matplotlib & Seaborn**: Used for data visualization and Exploratory Data Analysis (EDA).
* **Scikit-learn (Sklearn)**: For data preprocessing, model development, and evaluation.
* **XGBoost**: Contains the eXtreme Gradient Boosting machine learning algorithm, which is known for achieving high prediction accuracy.

---

## Methodology and Approaches Used

This notebook followed a structured machine learning pipeline:

### 1. Exploratory Data Analysis (EDA) and Cleaning

* **Data Inspection**: Inspected the first five rows (`df.head()`) and checked data types and non-null counts (`df.info()`).
* **Descriptive Statistics**: Examined descriptive statistical measures (`df.describe().T`) to understand the distribution and spread of the data.
* **Missing Value Handling**: The approach to handle missing values was to impute them with the **mean** of the respective column, as the columns contain continuous values.
* **Data Visualization**: Histograms were used to visualize the distribution of continuous-value columns.

### 2. Feature Engineering and Preprocessing

* **Target Binarization**: The continuous `quality` feature was converted into a binary classification target (`best quality`) to simplify the prediction problem.
* **Categorical Encoding**: The categorical `type` column ('white' and 'red') was converted into numerical values (1 and 0, respectively) to be compatible with machine learning models.
* **Data Splitting**: The segregated features and target variables were split into a **80:20 ratio** for training (`xtrain`, `ytrain`) and validation (`xtest`, `ytest`).
* **Feature Scaling**: The data was normalized using **`MinMaxScaler`** after the train-test split to ensure stable and fast model training.

### 3. Model Training and Evaluation

Three state-of-the-art machine learning models were trained on the prepared data for comparison:

* **Logistic Regression**
* **XGBoost Classifier (`XGBClassifier`)**
* **Support Vector Classifier (`SVC`)**

The performance of each model was evaluated using:
* **Training Accuracy**
* **Validation Accuracy** (Testing Accuracy)
* **Confusion Matrix**: Plotted for the validation data of all three models to visualize classification results.

---

## Results

Model evaluation indicated that the **Logistic Regression** and **SVC()** classifiers performed better on the validation data, showing a smaller difference between training and validation accuracy.