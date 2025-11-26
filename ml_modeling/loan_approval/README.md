# 📑 Loan Approval Prediction using Machine Learning

---

## Project Overview

`Loans` are the major requirement of the modern world. By this only, Banks get a major part of the total profit. It is beneficial for students to manage their education and living expenses, and for people to buy any luxury like houses, cars, etc. But when it comes to deciding whether the applicant’s profile is relevant for granting a loan. Banks have to look after many aspects.

So, here we will be using machine learning algorithms to ease their work and predict whether the candidate’s profile is relevant or not using key features like Marital Status, Education, Applicant Income, Credit History, etc.

This project implements a machine learning solution to predict whether a loan application will be approved based on various applicant features. The analysis involves **data preprocessing**, **exploratory data analysis (EDA)**, and **training multiple classification models** to determine the best predictor for the `Loan_Status` target variable. The goal is to build an accurate model to assist in loan eligibility screening.

---

## Dataset

The model utilizes the **`LoanApprovalPrediction.csv`** dataset. The dataset contains 13 features, including demographic and financial information about the applicants, with `Loan_ID` being dropped during preprocessing as it is not predictive.

### Key Features:

| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| **Gender** | Gender of the applicant. | Categorical |
| **Married** | Marital Status of the applicant. | Categorical |
| **Dependents** | Number of dependents (0, 1, 2, 3+). | Categorical |
| **Education** | Applicant's education level. | Categorical |
| **Self_Employed** | Indicates if the applicant is self-employed. | Categorical |
| **ApplicantIncome** | Applicant's income. | Numerical |
| **CoapplicantIncome** | Co-applicant's income. | Numerical |
| **LoanAmount** | Loan amount (in thousands). | Numerical |
| **Loan_Amount_Term** | Terms of the loan (in months). | Numerical |
| **Credit_History** | Credit history of debt repayment (0.0 or 1.0). | Numerical/Categorical |
| **Property_Area** | Area of property (Rural/Urban/Semi-urban). | Categorical |
| **Loan_Status (Target)** | Status of Loan Approval (Y for Yes, N for No). | Target Variable |

---

## Dependencies

The project is written in Python and requires the following libraries:

* **Pandas & NumPy**: For efficient data loading, cleaning, and numerical manipulation.
* **Matplotlib & Seaborn**: For data visualization, including EDA plots (e.g., bar plots, heatmaps) to understand feature distributions and correlations.
* **Scikit-learn (`sklearn`)**: For machine learning tasks, including:
    * `LabelEncoder` for categorical variable encoding.
    * `train_test_split` for data partitioning.
    * Classification models (listed below).
    * `metrics` for model evaluation.

---

## Methodology and Models

1.  **Data Preprocessing**: Handled missing values (imputation) and encoded categorical features (e.g., using `LabelEncoder`).
2.  **Model Training**: The data was split into training and testing sets, and the following classification models were trained:
    * **Logistic Regression**
    * **Support Vector Classifier (SVC)**
    * **K-Nearest Neighbors Classifier (KNeighborsClassifier)**
    * **Random Forest Classifier**

---

## Results and Conclusion

The **Random Forest Classifier** was identified as the best-performing model on the testing dataset, achieving an accuracy score of **82%**.

For potential future work and to achieve higher predictive accuracy, it is recommended to explore **ensemble learning techniques** such as **Bagging** and **Boosting** algorithms.