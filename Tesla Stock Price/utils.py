

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score


results = {}
# Function to compute model evaluation metrics
def model_metrics(y_true, y_pred):
    global results
    acc_s = accuracy_score(y_true, y_pred)
    roc_s = roc_auc_score(y_true, y_pred)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc_s, roc_s, f1_class_1, f1_class_0, cm

# Function to plot confusion matrix
def plot_cm(cm):
    global results
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return
def model_description(model, x_train, y_train, x_test, y_test):
    global results
    trained_model = model.fit(x_train, y_train)
    
    # Predictions
    train_preds = trained_model.predict(x_train)
    test_preds = trained_model.predict(x_test)
    
    # Compute evaluation metrics
    acc_train, roc_train, f1_1_train, f1_0_train, cm_train = model_metrics(y_train, train_preds)
    acc_test, roc_test, f1_1_test, f1_0_test, cm_test = model_metrics(y_test, test_preds)

    # Store results in dictionary
    results[model.__class__.__name__] = {
        "Train Accuracy": acc_train,
        "Roc-Auc Train Accuracy": roc_train,
        "Train F1 Class 1": f1_1_train,
        "Train F1 Class 0": f1_0_train,
        "Test Accuracy": acc_test,
        "Roc-Auc Test Accuracy": roc_test,
        "Test F1 Class 1": f1_1_test,
        "Test F1 Class 0": f1_0_test,
        "Confusion Matrix Train": cm_train,
        "Confusion Matrix Test": cm_test
    }
# Function to visualize stored confusion matrices
def plot_stored_cm(model_name):
    global results
    if model_name in results:
        print(f"\nConfusion Matrix for {model_name} (Train):")
        plot_cm(results[model_name]["Confusion Matrix Train"])
        
        print(f"Confusion Matrix for {model_name} (Test):")
        plot_cm(results[model_name]["Confusion Matrix Test"])
    else:
        print(f"Model '{model_name}' not found in results.")
    return
# Define a function to display a summary of model performance metrics
def display_results():
    global results
    print("\n=== Model Performance Summary ===")
    for model_name, metrics in results.items():
        print(f"\n📌 Model: {model_name}")
        print(f"Train Accuracy: {metrics['Train Accuracy']:.4f}")
        print(f"Roc-Auc Curve Train Accuracy: {metrics['Roc-Auc Train Accuracy']:.4f}")
        print(f"Train F1 Score (Class 1): {metrics['Train F1 Class 1']:.4f}")
        print(f"Train F1 Score (Class 0): {metrics['Train F1 Class 0']:.4f}")
        print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"Roc-Auc Curve Test Accuracy: {metrics['Roc-Auc Test Accuracy']:.4f}")
        print(f"Test F1 Score (Class 1): {metrics['Test F1 Class 1']:.4f}")
        print(f"Test F1 Score (Class 0): {metrics['Test F1 Class 0']:.4f}")

    print("\n=== Confusion Matrices ===")
    for model_name, metrics in results.items():
        print(f"\n🔹 Confusion Matrix for {model_name} (Train):")
        plot_cm(metrics["Confusion Matrix Train"])
        
        print(f"🔹 Confusion Matrix for {model_name} (Test):")
        plot_cm(metrics["Confusion Matrix Test"])

    return