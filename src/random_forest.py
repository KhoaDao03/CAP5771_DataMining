# src/random_forest.py
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

def random_forest_regression(X_train, y_train):
    """
    Train a Random Forest Regressor.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (RandomForestRegressor): Trained Random Forest Regressor model.
    """
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    return rf_reg

def evaluate_regression_model(model, X_test, y_test, project_dir, model_name='random_forest'):
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

    print(f"{model_name.replace('_', ' ').title()} Regression Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted Final Grades ({model_name.replace('_', ' ').title()} Regression)")
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_regression_actual_vs_predicted.png'))
    plt.close()

def random_forest_classification(X_train, y_train):
    """
    Train a Random Forest Classifier.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.

    Returns:
    - model (RandomForestClassifier): Trained Random Forest Classifier model.
    """
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    return rf_clf

def evaluate_classification_model(model, X_test, y_test, project_dir, model_name='random_forest'):
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

    print(f"{model_name.replace('_', ' ').title()} Classification Evaluation:")
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
    plt.title(f"Confusion Matrix ({model_name.replace('_', ' ').title()} Classification)")
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
    plt.title(f"ROC Curve ({model_name.replace('_', ' ').title()} Classification)")
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


def visualize_single_tree(model, X_train, project_dir, tree_index=0, model_name='random_forest'):
    """
    Visualize a single tree from the Random Forest model.

    Parameters:
    - model: Trained Random Forest model.
    - X_train (DataFrame): Training features.
    - project_dir (str): Path to save the plot.
    - tree_index (int): Index of the tree to visualize.
    - model_name (str): Name of the model.
    """
    estimator = model.estimators_[tree_index]
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, feature_names=X_train.columns, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree {tree_index} from {model_name.replace("_", " ").title()}')
    
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_tree_{tree_index}.png'))
    plt.close()
    print(f"Tree {tree_index} visualization saved to {figures_dir}/{model_name}_tree_{tree_index}.png")


def visualize_multiple_trees(model, X_train, project_dir, num_trees=3, max_depth=10, model_name='random_forest'):
    """
    Visualize multiple trees from a Random Forest model in a single image with improved spacing.

    Parameters:
    - model: Trained Random Forest model.
    - X_train (DataFrame): Training features.
    - project_dir (str): Path to save the plot.
    - num_trees (int): Number of trees to visualize.
    - max_depth (int): Maximum depth of the trees to display for simplicity.
    - model_name (str): Name of the model.
    """
    # Ensure the number of trees to plot does not exceed the number of estimators in the model
    num_trees = min(num_trees, len(model.estimators_))

    # Set up the plot grid with increased figure size and spacing
    fig, axes = plt.subplots(nrows=1, ncols=num_trees, figsize=(10 * num_trees, 8))
    if num_trees == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Plot each tree in the model
    for i in range(num_trees):
        estimator = model.estimators_[i]
        plot_tree(
            estimator, 
            feature_names=X_train.columns, 
            filled=True, 
            rounded=True, 
            ax=axes[i], 
            max_depth=max_depth,  # Limit the depth for clarity
            proportion=True,      # Scale nodes according to samples
            fontsize=10
        )
        axes[i].set_title(f'Tree {i + 1}', fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()

    # Save the plot to a file
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_multiple_trees.png'), bbox_inches='tight')
    plt.close()
    print(f"Multiple trees visualization saved to {figures_dir}/{model_name}_multiple_trees.png")

def main():
    """
    Main function to execute Random Forest Regression and Classification.
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
    # Random Forest Regression
    # -------------------------
    print("\n--- Random Forest Regression ---\n")
    # Load data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = load_data(project_dir, task='regression')

    # Train Random Forest Regressor
    rf_reg_model = random_forest_regression(X_train_reg, y_train_reg)

    # Evaluate Regression Model
    evaluate_regression_model(rf_reg_model, X_test_reg, y_test_reg, project_dir, model_name='random_forest')

    # Save Regression Model
    save_model(rf_reg_model, project_dir, filename='random_forest_regression_model.pkl')

    # ----------------------------
    # Random Forest Classification
    # ----------------------------
    print("\n--- Random Forest Classification ---\n")
    # Load data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = load_data(project_dir, task='classification')

    # Train Random Forest Classifier
    rf_clf_model = random_forest_classification(X_train_clf, y_train_clf)

    # Evaluate Classification Model
    evaluate_classification_model(rf_clf_model, X_test_clf, y_test_clf, project_dir, model_name='random_forest')

    # Save Classification Model
    save_model(rf_clf_model, project_dir, filename='random_forest_classification_model.pkl')
    visualize_single_tree(rf_clf_model, X_train_clf, project_dir, tree_index=0, model_name='random_forest')
    visualize_multiple_trees(rf_clf_model, X_train_clf, project_dir, num_trees=3, model_name='random_forest')

if __name__ == "__main__":
    main()
