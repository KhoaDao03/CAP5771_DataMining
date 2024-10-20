# src/logistic_regression.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(project_dir):
    outputs_dir = os.path.join(project_dir, 'outputs')
    X_train = pd.read_csv(os.path.join(outputs_dir, 'X_train_clf.csv'))
    X_test = pd.read_csv(os.path.join(outputs_dir, 'X_test_clf.csv'))
    y_train = pd.read_csv(os.path.join(outputs_dir, 'y_train_clf.csv'))
    y_test = pd.read_csv(os.path.join(outputs_dir, 'y_test_clf.csv'))
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train.values.ravel())
    return clf

def evaluate_model(model, X_test, y_test, project_dir):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("Logistic Regression Evaluation:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    plt.savefig(os.path.join(figures_dir, 'logistic_regression_confusion_matrix.png'))
    plt.close()

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figures_dir, 'logistic_regression_roc_curve.png'))
    plt.close()

def save_model(model, project_dir, filename='logistic_regression_model.pkl'):
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Model saved to outputs/models/{filename}")

def main():
    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Ensure output directories exist
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_data(project_dir)

    # Train model
    lr_model = train_logistic_regression(X_train, y_train)

    # Evaluate model
    evaluate_model(lr_model, X_test, y_test, project_dir)

    # Save model
    save_model(lr_model, project_dir)

if __name__ == "__main__":
    main()
