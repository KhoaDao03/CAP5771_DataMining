# src/linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(project_dir):
    """
    Loads the pre-split training and testing datasets for regression from the outputs directory.

    :param project_dir: Path to the project root directory.
    :return: Training and testing datasets for features (X) and target (y).
    """
    outputs_dir = os.path.join(project_dir, 'outputs')
    X_train = pd.read_csv(os.path.join(outputs_dir, 'X_train_reg.csv'))
    X_test = pd.read_csv(os.path.join(outputs_dir, 'X_test_reg.csv'))
    y_train = pd.read_csv(os.path.join(outputs_dir, 'y_train_reg.csv'))
    y_test = pd.read_csv(os.path.join(outputs_dir, 'y_test_reg.csv'))
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Initializes and trains a LinearRegression model using the training data.

    :param X_train: Features for training the model.
    :param y_train: Target values (G3) for training the model.
    :return: Trained LinearRegression model.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def evaluate_model(model, X_test, y_test, project_dir):
    """
    Evaluates the trained LinearRegression model by predicting on the test data and calculating metrics.

    Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R² Score: Coefficient of Determination

    Also creates a plot comparing Actual vs Predicted G3 scores and saves it.

    :param model: Trained LinearRegression model.
    :param X_test: Test features.
    :param y_test: Actual test target values (G3).
    :param project_dir: Path to the project root directory for saving the plot.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print("Linear Regression Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test.values.flatten(), y=predictions.flatten())
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title("Actual vs Predicted Final Grades")
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    plt.savefig(os.path.join(
        figures_dir, 'linear_regression_actual_vs_predicted.png'))
    plt.close()


def save_model(model, project_dir, filename='linear_regression_model.pkl'):
    """
    Saves the trained LinearRegression model to the outputs/models/ directory using joblib.

    :param model: Trained LinearRegression model.
    :param project_dir: Path to the project root directory for saving the model.
    :param filename: Name of the file to save the model as (default is 'linear_regression_model.pkl').
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Model saved to outputs/models/{filename}")


def main():
    """
    Main function to load the data, train the model, evaluate it, and save both the model and the evaluation plot.
    """
    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Ensure output directories exist
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_data(project_dir)

    # Train model
    lr_model = train_linear_regression(X_train, y_train)

    # Evaluate model
    evaluate_model(lr_model, X_test, y_test, project_dir)

    # Save model
    save_model(lr_model, project_dir)


if __name__ == "__main__":
    main()
