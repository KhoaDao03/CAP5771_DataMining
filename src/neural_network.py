# src/neural_network.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_data(project_dir, task='regression'):
    """
    Load and scale data for neural network training.
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

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.squeeze(), y_test.squeeze(), scaler

def ann_regression(X_train, y_train):
    """
    Train an Artificial Neural Network for regression.

    Returns:
    - model: Trained Keras model.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

    return model

def evaluate_regression_model(model, X_test, y_test, project_dir, model_name='ann'):
    """
    Evaluate the regression model and generate evaluation metrics and plots.
    """
    predictions = model.predict(X_test).flatten()
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

def ann_classification(X_train, y_train):
    """
    Train an Artificial Neural Network for classification.

    Returns:
    - model: Trained Keras model.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

    return model

def evaluate_classification_model(model, X_test, y_test, project_dir, model_name='ann'):
    """
    Evaluate the classification model and generate evaluation metrics and plots.
    """
    probabilities = model.predict(X_test).flatten()
    predictions = (probabilities >= 0.5).astype(int)

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

def save_model(model, scaler, project_dir, filename):
    """
    Save the trained model and scaler to the models directory.
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Save model
    model.save(os.path.join(models_dir, filename))
    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, f"{filename}_scaler.pkl"))
    print(f"Model and scaler saved to outputs/models/{filename}")

def main():
    """
    Main function to execute ANN Regression and Classification.
    """
    # Suppress any unnecessary warnings for cleaner output
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')

    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Ensure output directories exist
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------
    # ANN Regression
    # -------------------------
    print("\n--- ANN Regression ---\n")
    # Load data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg = load_data(project_dir, task='regression')

    # Train ANN Regressor
    ann_reg_model = ann_regression(X_train_reg, y_train_reg)

    # Evaluate Regression Model
    evaluate_regression_model(ann_reg_model, X_test_reg, y_test_reg, project_dir, model_name='ann')

    # Save Regression Model and Scaler
    save_model(ann_reg_model, scaler_reg, project_dir, filename='ann_regression_model.h5')

    # ----------------------------
    # ANN Classification
    # ----------------------------
    print("\n--- ANN Classification ---\n")
    # Load data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, scaler_clf = load_data(project_dir, task='classification')

    # Train ANN Classifier
    ann_clf_model = ann_classification(X_train_clf, y_train_clf)

    # Evaluate Classification Model
    evaluate_classification_model(ann_clf_model, X_test_clf, y_test_clf, project_dir, model_name='ann')

    # Save Classification Model and Scaler
    save_model(ann_clf_model, scaler_clf, project_dir, filename='ann_classification_model.h5')

if __name__ == "__main__":
    main()
