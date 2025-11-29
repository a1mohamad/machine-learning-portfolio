# 📈 Tesla Stock Price Prediction using Machine Learning

---

## Project Overview

Machine learning proves immensely helpful in many industries in automating  tasks that earlier required human labor one such application of ML is predicting whether a particular trade will be profitable or not.

In this article, we will learn how to predict a signal that indicates whether buying a particular stock will be helpful or not by using ML. For this purpose, we use Tesla stock price.

This project implements a comprehensive machine learning solution to predict the **Tesla (TSLA) stock price** movement. It utilizes **historical OHLC data** in a **time-series classification** approach. The objective is to evaluate multiple state-of-the-art classifiers to find the best model for forecasting, with a focus on **eXtreme Gradient Boosting (XGBoost)** for its known predictive strength in complex datasets.

---

## Dataset

The analysis is based on the **Tesla Stock Price data** (contained in `tesla_stock.csv`), covering approximately **10 years** of trading data from **June 29, 2010, to February 3, 2020**.

### Data Statistics

* **Total Entries**: 2416 rows
* **Features**: 7 columns (OHLC data plus volume)

| Feature Name | Description |
| :--- | :--- |
| **Date** | The trading day (used for time-series analysis) |
| **Open** | Price at the start of the trading day |
| **High** | Highest price during the day |
| **Low** | Lowest price during the day |
| **Close** | Price at the end of the trading day |
| **Adj Close** | Closing price adjusted for corporate actions |
| **Volume** | Number of shares traded |

---

## Dependencies

This project requires the following Python libraries for data processing, model building, and evaluation:

* **Pandas**: For loading the data frame and data analysis.
* **NumPy**: For fast numerical computations.
* **Matplotlib/Seaborn**: For data visualization and graphical representations.
* **Scikit-learn (Sklearn)**: For data preprocessing, splitting, and model development.
* **XGBoost**: For the eXtreme Gradient Boosting algorithm, targeting high-accuracy predictions.

---

## Methodology and Approaches Used

The analysis followed a detailed machine learning workflow incorporating specialized time-series techniques:

### 1. Exploratory Data Analysis (EDA)

* **Data Inspection**: Performed checks on the dataset shape, descriptive statistics, and data types (`df.head()`, `df.shape`, `df.describe()`, `df.info()`).
* **Trend Analysis**: Analyzed how the stock prices have moved over the period of time.
* **Quarterly Impact Study**: Specifically analyzed how the **end of the quarters affects the stock prices**, as quarterly results heavily influence stock movements.

### 2. Feature Engineering

* **Date Decomposition**: The `Date` column was broken down to create three new features: **`year`**, **`month`**, and **`day`**.
* **Quarter-End Indicator**: A new binary feature, **`is_quarter_end`**, was created to flag trading days that fall on a month that ends a quarter (Month 3, 6, 9, 12). This was calculated using `df['is_quarter_end'] = np.where((df['month'] % 3 == 0), 1, 0)`.

### 3. Data Preprocessing and Splitting

* **Normalization**: The data was normalized using `StandardScaler` (inferred from import) to ensure **stable and fast training** of the model.
* **Train-Test Split**: The entire dataset was split into training and testing sets with an **80/20 ratio** using `train_test_split`.

### 4. Model Training and Selection

A battery of **six machine learning models** was trained and compared to determine the optimal classifier for the prediction task:

* **Logistic Regression**
* **Support Vector Classifier (SVC)** (with `kernel='poly'` and `probability=True`)
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **XGBoost Classifier (`XGBClassifier`)**
* **XGBoost Random Forest Classifier (`XGBRFClassifier`)**

### 5. Evaluation Metrics

The models were evaluated using a comprehensive set of metrics, chosen to assess performance accurately, especially given the classification of stock price movements:

* **ROC-AUC Curve**: Used as a primary metric because the project aims to predict **soft probabilities** (continuous values between 0 and 1), which the ROC-AUC curve is generally used to measure.
* **Accuracy Score**
* **F1 Score** (for both Class 1 and Class 0)
* **Confusion Matrix**

---

## Results and Conclusion

The **XGBoost Classifier (`XGBClassifier`)** demonstrated the strongest generalization performance on the validation set:

| Model | Test Accuracy | Roc-Auc Curve Test Accuracy | Test F1 Score (Class 1) |
| :--- | :--- | :--- | :--- |
| **XGBClassifier** | **0.5351** | **0.5339** | **0.5580** |
| *LogisticRegression* | 0.5227 | 0.5141 | 0.6118 |
| *SVC* | 0.5103 | 0.4904 | 0.6740 |

*Note: The high training accuracies (1.0000) for the Decision Tree and Random Forest models suggest significant overfitting, making their test results less reliable despite the strong training performance*.