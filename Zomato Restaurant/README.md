# Zomato Data Analysis Using Python 🍽️

## Project Description
This project conducts a **detailed data analysis** of Zomato restaurant data using Python and its core data science libraries. The primary goal is to extract meaningful insights by exploring restaurant features, delivery options, customer preferences, and costs.

---

## Analysis Objectives
The analysis in this notebook seeks to answer the following key questions based on the dataset:

1.  Do a greater number of restaurants provide **online delivery** as opposed to offline services?
2.  Which **types of restaurants** are the most favored by the general public?
3.  What **price range** is preferred by couples for their dinner at restaurants?

---

## Data Source
The analysis is performed on a CSV file titled `"Zomato data .csv"`.

---

## Technologies and Libraries

The project utilizes the following Python libraries for data manipulation, analysis, and visualization:

* **Pandas:** For loading data frames and performing multiple data analysis tasks.
* **Numpy:** For efficient handling of large calculations and complex computations.
* **Matplotlib:** Used for creating high-quality **plots, charts, and histograms**.
* **Seaborn:** Offers a high-level interface for creating visually appealing and **informative statistical graphics**.

---

## Methodology and Steps
The following key steps were performed in the notebook:

1.  **Import Libraries:** Import `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.
2.  **Data Loading & Preprocessing:** Load the data. The `rate` column data type is cleaned and converted to a **float** by removing the `/5` denominator.
3.  **Data Inspection:** Obtain a summary using `dataframe.info()` and check for null values.
4.  **Distribution Analysis:** Analyze and visualize the distribution of restaurant types (`listed_in(type)`) and customer ratings (`rate`).
5.  **Cost Analysis:** Explore the distribution of approximate costs for two people (`approx_cost(for two people)`).
6.  **Online vs. Offline Orders:** Create a pivot table and a **heatmap** to visualize the relationship between the type of restaurant and its online ordering availability.

---

## Key Findings
The analysis yielded the following main conclusions:

* **Ratings Distribution:** The majority of restaurants received ratings ranging from **3.8 to 4.2**.
* **Online Order Preference:** **Dining restaurants** primarily accept **offline orders**, whereas **cafes** primarily receive **online orders**. This suggests a preference for placing orders in person at full-service restaurants but favoring online ordering at cafes.

---

## References
This project was developed with reference to the following resource:
* [Zomato Data Analysis Using Python](https://www.geeksforgeeks.org/zomato-data-analysis-using-python/)