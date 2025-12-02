# 🏠 California House Price Prediction with Linear Regression

---

## Project Overview

This project focuses on predicting house prices in California using a **Linear Regression** model. It utilizes the well-known **California Housing Dataset** to perform a classic **regression analysis**. The objective is to build a basic predictive model and evaluate its performance using standard regression metrics.

---

## Dataset

The analysis uses the **California Housing Dataset**, which is conveniently accessed via `sklearn.datasets.fetch_california_housing`.

### Data Characteristics

* **Total Records**: 20,640 entries
* **Features**: 8 independent variables

### Key Features

| Feature Name | Description |
| :--- | :--- |
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| **Target Variable** | Median house value for California districts (The price to be predicted) |

---

## Dependencies

The project relies on the following Python libraries for data handling, modeling, and visualization:

* **Scikit-learn (`sklearn`)**: Key for fetching the dataset, splitting data, and implementing the Linear Regression model.
    * `fetch_california_housing` (Dataset loading)
    * `train_test_split` (Data splitting)
    * `LinearRegression` (Model implementation)
    * `mean_absolute_error`, `mean_squared_error` (Evaluation metrics)
* **Pandas & NumPy**: For efficient data manipulation.
* **Matplotlib & Seaborn**: For data visualization (e.g., plotting actual vs. predicted values).

---

## Methodology and Approaches Used

The project followed a straightforward regression modeling pipeline:

### 1. Data Preparation and Splitting

* **Dataset Loading**: The California housing data was loaded directly from the `sklearn.datasets` module.
* **DataFrame Creation**: The NumPy data was converted into a **Pandas DataFrame** for easier manipulation and analysis.
* **Train-Test Split**: The data was split into a **70% training set** and a **30% testing set** to evaluate the model's generalization capability.

### 2. Model Training

* **Model Selection**: A **Linear Regression** model was chosen as the algorithm for this prediction task.
* **Training**: The model was trained using the features (`x_train`) and target values (`y_train`).

### 3. Prediction and Evaluation

* **Prediction**: The trained model was used to predict the house prices (`y_pred`) on the test set (`x_test`).
* **Evaluation Metrics**: The performance of the Linear Regression model was assessed using two key regression metrics:
    * **Mean Squared Error (MSE)**: The average of the squared differences between the predicted and actual values.
    * **Mean Absolute Error (MAE)**: The average of the absolute differences between the predicted and actual values.
* **Visualization**: A scatter plot was created to visually compare the **Actual Test Values** against the **Predicted Values**.

---

## Results and Next Steps

### Model Performance

The Linear Regression model achieved the following results on the test data:

* **Mean Squared Error (MSE)**: **0.548**
* **Mean Absolute Error (MAE)**: **0.538**

### Suggested Improvements

The notebook suggests further steps to improve the model's performance, which may include:

1.  **Exploring other regression algorithms** (e.g., Ridge, Lasso, or tree-based models).
2.  **Hyperparameter Tuning** to optimize the Linear Regression model.
3.  **Feature Engineering** to create more predictive variables.