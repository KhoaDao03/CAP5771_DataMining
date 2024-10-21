
# Student Performance Prediction Project

This project aims to predict student performance using both regression and classification techniques. Two types of models are built: a **Linear Regression** model to predict student grades, and a **Logistic Regression** model to classify whether a student will pass or fail. The data used in this project comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

## Project Structure

The project is structured into several components, each with its own functionality:

- **`data_preprocessing.py`**: Preprocesses the raw data to prepare it for modeling.
- **`linear_regression.py`**: Builds and evaluates a linear regression model to predict student grades (G3).
- **`logistic_regression.py`**: Builds and evaluates a logistic regression model to classify student pass/fail status.

### Directory Structure

```
├── data/
│   ├── student-mat.csv            # Raw Math dataset
│   └── student-por.csv            # Raw Portuguese dataset
│   └── student-merge.R            # R script that can be used to merge or process student datasets.
│   └── student.txt                # Text file with additional information about the datasets or the project.
├── outputs/
│   ├── figures/                   # Plots generated during evaluation
│   ├── models/                    # Saved models
│   ├── X_train_reg.csv            # Training features for regression
│   ├── X_test_reg.csv             # Testing features for regression
│   ├── y_train_reg.csv            # Training target (G3) for regression
│   ├── y_test_reg.csv             # Testing target (G3) for regression
│   ├── X_train_clf.csv            # Training features for classification
│   ├── X_test_clf.csv             # Testing features for classification
│   ├── y_train_clf.csv            # Training target (pass/fail) for classification
│   ├── y_test_clf.csv             # Testing target (pass/fail) for classification
│   └── preprocessed_data.csv      # Full preprocessed dataset
├── src/
│   ├── data_preprocessing.py      # Script to preprocess the data
│   ├── linear_regression.py       # Script to train and evaluate the linear regression model
│   └── logistic_regression.py     # Script to train and evaluate the logistic regression model
├── README.md                      # Documentation file
├── requirements.txt               # File containing the list of Python dependencies
```

## 1. Data Preprocessing

The first step involves preprocessing the raw datasets.

### Script: `data_preprocessing.py`

#### Functionality:

1. **Loading Data**:
   - Reads the Math (`student-mat.csv`) and Portuguese (`student-por.csv`) datasets.
   
2. **Merging Datasets**:
   - Merges both datasets based on common student attributes.
   - For students appearing in both datasets, it averages their grades.

3. **Preprocessing**:
   - **Handling Missing Values**: Replaces missing values ('?') with NaN and drops rows with missing values.
   - **Encoding Categorical Variables**: Uses `LabelEncoder` to convert categorical variables into numerical formats.
   - **Feature Scaling**: Applies `StandardScaler` to normalize numerical features.

4. **Creating Classification Target**:
   - A binary target column is created where 1 indicates a passing grade (G3 ≥ 10) and 0 indicates failing.

5. **Splitting Data**:
   - Splits the data into training and testing sets for both **regression** and **classification** tasks.

6. **Saving Preprocessed Data**:
   - The preprocessed data and the training/testing splits are saved in the `outputs/` folder for future use.

#### Running the Script:

To run the preprocessing:

```bash
python src/data_preprocessing.py
```

The preprocessed data will be saved in the `outputs/` folder.

## 2. Linear Regression Model

This script builds and evaluates a linear regression model to predict the final grade (G3).

### Script: `linear_regression.py`

#### Functionality:

1. **Loading Data**:
   - Reads the pre-split regression datasets from the `outputs/` folder.

2. **Training the Model**:
   - Initializes and trains a `LinearRegression` model using the training data.

3. **Evaluating the Model**:
   - Predicts on the test set and calculates evaluation metrics:
     - **MAE**: Mean Absolute Error
     - **MSE**: Mean Squared Error
     - **RMSE**: Root Mean Squared Error
     - **R² Score**: Coefficient of Determination
   - Plots Actual vs. Predicted G3 scores and saves the figure.

4. **Saving the Model**:
   - Saves the trained model using `joblib` for future use.

#### Running the Script:

```bash
python src/linear_regression.py
```

**Outputs**:
- **Console**: Review evaluation metrics printed.
- **`outputs/models/`:** Saved model file: `linear_regression_model.pkl`.
- **`outputs/figures/`:** Saved plot: `linear_regression_actual_vs_predicted.png`.

## 3. Logistic Regression Model

This script builds and evaluates a logistic regression model to classify whether a student passes or fails based on their final grade.

### Script: `logistic_regression.py`

#### Functionality:

1. **Loading Data**:
   - Reads the pre-split classification datasets from the `outputs/` folder.

2. **Training the Model**:
   - Initializes and trains a `LogisticRegression` model using the training data.
   - Uses `max_iter=1000` to ensure model convergence.

3. **Evaluating the Model**:
   - Predicts on the test set and calculates key metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
     - **ROC-AUC**
   - Generates and saves:
     - **Confusion Matrix**: Visual representation of actual vs. predicted classes.
     - **ROC Curve**: Illustrates the diagnostic ability of the classifier.

4. **Saving the Model**:
   - Saves the trained model using `joblib` for future use.

#### Running the Script:

```bash
python src/logistic_regression.py
```

**Outputs**:
- **Console**: Review evaluation metrics and classification report.
- **`outputs/models/`:** Saved model file: `logistic_regression_model.pkl`.
- **`outputs/figures/`:**
  - `logistic_regression_confusion_matrix.png` (Confusion Matrix)
  - `logistic_regression_roc_curve.png` (ROC Curve)

## Requirements

Make sure to install the required Python packages before running the scripts:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following dependencies:

```
pandas
scikit-learn
matplotlib
seaborn
joblib
```

## Conclusion

This project demonstrates how to preprocess data, train models for both regression and classification tasks, evaluate model performance, and save results. The linear regression model is used for predicting student grades, while the logistic regression model is used for classifying students as pass or fail based on their final grades.
