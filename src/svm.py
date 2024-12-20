# src/svm.py

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import SVR, SVC
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
from sklearn.preprocessing import StandardScaler

def load_data(project_dir, task='regression'):
    """
    Load the preprocessed data for regression or classification.

    Returns scaled data suitable for SVM.
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

    # Scale features for SVM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.squeeze(), y_test.squeeze(), scaler

def svm_regression(X_train, y_train):
    """
    Train a Support Vector Regressor.

    Parameters:
    - X_train (array): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (SVR): Trained Support Vector Regressor model.
    """
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    svr.fit(X_train, y_train)
    return svr

def evaluate_regression_model(model, X_test, y_test, project_dir, model_name='svm'):
    """
    Evaluate the regression model and generate evaluation metrics and plots.

    Parameters:
    - model: Trained regression model.
    - X_test (array): Testing features.
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

def svm_classification(X_train, y_train):
    """
    Train a Support Vector Classifier.

    Parameters:
    - X_train (array): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (SVC): Trained Support Vector Classifier model.
    """
    svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svc.fit(X_train, y_train)

    
    return svc

def evaluate_classification_model(model, X_test, y_test, project_dir, model_name='svm'):
    """
    Evaluate the classification model and generate evaluation metrics and plots.

    Parameters:
    - model: Trained classification model.
    - X_test (array): Testing features.
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
    

def save_model(model, scaler, project_dir, filename):
    """
    Save the trained model and scaler to the models directory.

    Parameters:
    - model: Trained model.
    - scaler: Scaler used to preprocess data.
    - project_dir (str): Absolute path to the project root directory.
    - filename (str): Desired filename for the saved model.
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Save model and scaler together
    joblib.dump({'model': model, 'scaler': scaler}, os.path.join(models_dir, filename))
    print(f"Model saved to outputs/models/{filename}")


def plot_svm_decision_boundary(model, X, y, project_dir, title='SVM Decision Boundary', model_name='svm'):
    """
    Plots the decision boundary of an SVM classifier after reducing dimensions to 2D using PCA.
    Saves the plot to the figures folder.
    
    Parameters:
    - model: Trained SVM model.
    - X (array): Feature data.
    - y (array): Target labels.
    - project_dir (str): Path to the project directory.
    - title (str): Title of the plot.
    - model_name (str): Name of the model (used for saving files).
    """
    # Reduce data to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Create a mesh to plot in
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    # Flatten the grid to pass into the classifier
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid)
    Z = model.predict(grid_original).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k'
    )
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Save the plot to the figures folder
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_decision_boundary.png'))
    plt.close()

def plot_regression_residuals(model, X_test, y_test, project_dir, title='Residuals Plot', model_name='svm'):
    """
    Plots the residuals of the regression model predictions and saves to the figures folder.
    
    Parameters:
    - model: Trained regression model.
    - X_test (array): Testing features.
    - y_test (array): Actual target values.
    - project_dir (str): Path to the project directory.
    - title (str): Title of the plot.
    - model_name (str): Name of the model (used for saving files).
    """
    predictions = model.predict(X_test)
    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))
    sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)

    # Save the plot to the figures folder
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_regression_residuals.png'))
    plt.close()


def plot_support_vectors(model, X, y, project_dir, title='Support Vectors', model_name='svm'):
    """
    Plots the support vectors of an SVM classifier after reducing dimensions to 2D using PCA.
    Saves the plot to the figures folder.
    
    Parameters:
    - model: Trained SVM model.
    - X (array): Feature data.
    - y (array): Target labels.
    - project_dir (str): Path to the project directory.
    - title (str): Title of the plot.
    - model_name (str): Name of the model (used for saving files).
    """
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    support_vectors = model.support_vectors_
    support_vectors_reduced = pca.transform(support_vectors)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k'
    )
    plt.scatter(
        support_vectors_reduced[:, 0],
        support_vectors_reduced[:, 1],
        facecolors='none',
        edgecolors='yellow',
        s=100,
        label='Support Vectors'
    )
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Save the plot to the figures folder
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_support_vectors.png'))
    plt.close()



def main():
    """
    Main function to execute SVM Regression and Classification.
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
    # SVM Regression
    # -------------------------
    print("\n--- SVM Regression ---\n")
    # Load data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg = load_data(project_dir, task='regression')

    # Train SVM Regressor
    svm_reg_model = svm_regression(X_train_reg, y_train_reg)

    # Evaluate Regression Model
    evaluate_regression_model(svm_reg_model, X_test_reg, y_test_reg, project_dir, model_name='svm')
    
    # Save Regression Model and Scaler
    save_model(svm_reg_model, scaler_reg, project_dir, filename='svm_regression_model.pkl')

    # ----------------------------
    # SVM Classification
    # ----------------------------
    print("\n--- SVM Classification ---\n")
    # Load data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, scaler_clf = load_data(project_dir, task='classification')

    # Train SVM Classifier
    svm_clf_model = svm_classification(X_train_clf, y_train_clf)

  # Visualize SVM Classification
    print("\nVisualizing SVM Classification Decision Boundary...")
    X_combined = np.vstack((X_train_clf, X_test_clf))
    y_combined = np.hstack((y_train_clf, y_test_clf))
    plot_svm_decision_boundary(svm_clf_model, X_combined, y_combined, project_dir, title='SVM Classification Decision Boundary')

    print("\nVisualizing SVM Support Vectors...")
    plot_support_vectors(svm_clf_model, X_combined, y_combined, project_dir, title='SVM Support Vectors')

    # Save Classification Model and Scaler
    save_model(svm_clf_model, scaler_clf, project_dir, filename='svm_classification_model.pkl')

if __name__ == "__main__":
    main()
