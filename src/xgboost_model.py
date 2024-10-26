# src/xgboost_model.py

import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

def load_data(project_dir, task='regression'):
    """
    Load the preprocessed data for regression or classification.

    Parameters:
    - project_dir (str): Absolute path to the project root directory.
    - task (str): 'regression' or 'classification' to specify the task.

    Returns:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Testing features.
    - y_train (Series): Training target.
    - y_test (Series): Testing target.
    """
    outputs_dir = os.path.join(project_dir, 'outputs')
    if task == 'regression':
        X_train = pd.read_csv(os.path.join(outputs_dir, 'X_train_reg.csv'))
        X_test = pd.read_csv(os.path.join(outputs_dir, 'X_test_reg.csv'))
        y_train = pd.read_csv(os.path.join(outputs_dir, 'y_train_reg.csv'))
        y_test = pd.read_csv(os.path.join(outputs_dir, 'y_test_reg.csv'))
    elif task == 'classification':
        X_train = pd.read_csv(os.path.join(outputs_dir, 'X_train_clf.csv'))
        X_test = pd.read_csv(os.path.join(outputs_dir, 'X_test_clf.csv'))
        y_train = pd.read_csv(os.path.join(outputs_dir, 'y_train_clf.csv'))
        y_test = pd.read_csv(os.path.join(outputs_dir, 'y_test_clf.csv'))
    else:
        raise ValueError("Task must be 'regression' or 'classification'")
    return X_train, X_test, y_train.squeeze(), y_test.squeeze()

def xgboost_regression(X_train, y_train):
    """
    Train an XGBoost Regressor.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (XGBRegressor): Trained XGBoost Regressor model.
    """
    xgb_reg = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train)
    return xgb_reg

def evaluate_regression_model(model, X_test, y_test, project_dir, model_name='xgboost'):
    """
    Evaluate the regression model and generate evaluation metrics and plots.

    Parameters:
    - model: Trained regression model.
    - X_test (DataFrame): Testing features.
    - y_test (Series): Testing target.
    - project_dir (str): Absolute path to the project root directory.
    - model_name (str): Name of the model (used for saving plots).
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name.upper()} Regression Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted Final Grades ({model_name.upper()} Regression)")
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_regression_actual_vs_predicted.png'))
    plt.close()

def xgboost_classification(X_train, y_train):
    """
    Train an XGBoost Classifier.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (XGBClassifier): Trained XGBoost Classifier model.
    """
    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def evaluate_classification_model(model, X_test, y_test, project_dir, model_name='xgboost'):
    """
    Evaluate the classification model and generate evaluation metrics and plots.

    Parameters:
    - model: Trained classification model.
    - X_test (DataFrame): Testing features.
    - y_test (Series): Testing target.
    - project_dir (str): Absolute path to the project root directory.
    - model_name (str): Name of the model (used for saving plots).
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"{model_name.upper()} Classification Evaluation:")
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
    plt.title(f"Confusion Matrix ({model_name.upper()} Classification)")
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_classification_confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name.upper()} Classification)")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figures_dir, f'{model_name}_classification_roc_curve.png'))
    plt.close()

def save_model(model, project_dir, filename):
    """
    Save the trained model to the models directory.

    Parameters:
    - model: Trained model.
    - project_dir (str): Absolute path to the project root directory.
    - filename (str): Desired filename for the saved model.
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Model saved to outputs/models/{filename}")

def main():
    """
    Main function to execute XGBoost Regression and Classification.
    """
    # Suppress any unnecessary warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Ensure output directories exist
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------
    # XGBoost Regression
    # -------------------------
    print("\n--- XGBoost Regression ---\n")
    # Load data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = load_data(project_dir, task='regression')

    # Train XGBoost Regressor
    xgb_reg_model = xgboost_regression(X_train_reg, y_train_reg)

    # Evaluate Regression Model
    evaluate_regression_model(xgb_reg_model, X_test_reg, y_test_reg, project_dir, model_name='xgboost')

    # Save Regression Model
    save_model(xgb_reg_model, project_dir, filename='xgboost_regression_model.pkl')

    # ----------------------------
    # XGBoost Classification
    # ----------------------------
    print("\n--- XGBoost Classification ---\n")
    # Load data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = load_data(project_dir, task='classification')

    # Train XGBoost Classifier
    xgb_clf_model = xgboost_classification(X_train_clf, y_train_clf)

    # Evaluate Classification Model
    evaluate_classification_model(xgb_clf_model, X_test_clf, y_test_clf, project_dir, model_name='xgboost')

    # Save Classification Model
    save_model(xgb_clf_model, project_dir, filename='xgboost_classification_model.pkl')

if __name__ == "__main__":
    main()
