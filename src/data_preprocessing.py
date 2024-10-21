# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(mat_path, por_path):
    """
    Loads the Math and Portuguese student datasets.

    :param mat_path: Path to the Math dataset (student-mat.csv).
    :param por_path: Path to the Portuguese dataset (student-por.csv).
    :return: Two dataframes - one for Math and one for Portuguese.
    """
    df_mat = pd.read_csv(mat_path, sep=';')
    df_por = pd.read_csv(por_path, sep=';')
    return df_mat, df_por


def merge_datasets(df_mat, df_por):
    """
    Merges the Math and Portuguese datasets on common student attributes and handles duplicate students.
    For students appearing in both datasets, their grades are averaged.

    :param df_mat: Dataframe for the Math dataset.
    :param df_por: Dataframe for the Portuguese dataset.
    :return: A merged dataframe of both datasets.
    """
    # Merge datasets on all common columns except the grade columns
    common_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                   'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                   'traveltime', 'studytime', 'failures', 'schoolsup',
                   'famsup', 'paid', 'activities', 'nursery', 'higher',
                   'internet', 'romantic', 'famrel', 'freetime', 'goout',
                   'Dalc', 'Walc', 'health', 'absences']

    df_merged = pd.merge(df_mat, df_por, on=common_cols,
                         suffixes=('_mat', '_por'))

    # For students present in both datasets, take the average of their grades
    grade_cols = ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']
    for grade in ['G1', 'G2', 'G3']:
        df_merged[grade] = df_merged[[
            f'{grade}_mat', f'{grade}_por']].mean(axis=1)

    # Drop original grade columns
    df_merged.drop(columns=grade_cols, inplace=True)

    # Concatenate unique students from both datasets
    df_unique_mat = df_mat[~df_mat.index.isin(df_merged.index)][['school', 'sex', 'age', 'address',
                                                                 'famsize', 'Pstatus', 'Medu', 'Fedu',
                                                                 'Mjob', 'Fjob', 'reason', 'guardian',
                                                                 'traveltime', 'studytime', 'failures',
                                                                 'schoolsup', 'famsup', 'paid',
                                                                 'activities', 'nursery', 'higher',
                                                                 'internet', 'romantic', 'famrel',
                                                                 'freetime', 'goout', 'Dalc', 'Walc',
                                                                 'health', 'absences', 'G1', 'G2', 'G3']]
    df_unique_por = df_por[~df_por.index.isin(df_merged.index)][['school', 'sex', 'age', 'address',
                                                                 'famsize', 'Pstatus', 'Medu', 'Fedu',
                                                                 'Mjob', 'Fjob', 'reason', 'guardian',
                                                                 'traveltime', 'studytime', 'failures',
                                                                 'schoolsup', 'famsup', 'paid',
                                                                 'activities', 'nursery', 'higher',
                                                                 'internet', 'romantic', 'famrel',
                                                                 'freetime', 'goout', 'Dalc', 'Walc',
                                                                 'health', 'absences', 'G1', 'G2', 'G3']]
    df_final = pd.concat(
        [df_merged, df_unique_mat, df_unique_por], ignore_index=True)

    return df_final


def preprocess_data(df):
    """
    Handles missing values, encodes categorical variables, and scales numerical features.

    :param df: Dataframe to preprocess.
    :return: Preprocessed dataframe, label encoders used for categorical features, and the scaler.
    """
    # Handle missing values (if any)
    df = df.replace('?', np.nan)
    df = df.dropna()

    # Encode categorical variables
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid',
                        'activities', 'nursery', 'higher',
                        'internet', 'romantic']

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature Scaling
    scaler = StandardScaler()
    numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                    'failures', 'famrel', 'freetime', 'goout',
                    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, label_encoders, scaler


def split_data(df, target, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets.

    :param df: Dataframe to split.
    :param target: Target column for prediction (either G3 or pass).
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Seed used by the random number generator.
    :return: Training and testing sets for features and target.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_classification_target(df, threshold=10):
    """
    Creates a binary target column 'pass' where 1 represents a final grade (G3) of 10 or above, and 0 otherwise.

    :param df: Dataframe to add the binary 'pass' column to.
    :param threshold: Grade threshold to determine pass/fail.
    :return: Updated dataframe with the binary 'pass' column.
    """
    # Create a binary target: 1 if G3 >= threshold, else 0
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= threshold else 0)
    return df


def main():
    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Define file paths
    mat_path = os.path.join(project_dir, 'data', 'student-mat.csv')
    por_path = os.path.join(project_dir, 'data', 'student-por.csv')

    # Check if files exist
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Math dataset not found at {mat_path}")
    if not os.path.exists(por_path):
        raise FileNotFoundError(f"Portuguese dataset not found at {por_path}")

    # Load data
    df_mat, df_por = load_data(mat_path, por_path)

    # Merge datasets
    df = merge_datasets(df_mat, df_por)

    # Create classification target
    df = prepare_classification_target(df, threshold=10)

    # Preprocess data
    df, label_encoders, scaler = preprocess_data(df)

    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(project_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # Save preprocessed data
    preprocessed_path = os.path.join(outputs_dir, 'preprocessed_data.csv')
    df.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")

    # Split data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(df, 'G3')

    # Split data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(df, 'pass')

    # Save splits
    splits = {
        'X_train_reg.csv': X_train_reg,
        'X_test_reg.csv': X_test_reg,
        'y_train_reg.csv': y_train_reg,
        'y_test_reg.csv': y_test_reg,
        'X_train_clf.csv': X_train_clf,
        'X_test_clf.csv': X_test_clf,
        'y_train_clf.csv': y_train_clf,
        'y_test_clf.csv': y_test_clf
    }

    for filename, data in splits.items():
        path = os.path.join(outputs_dir, filename)
        data.to_csv(path, index=False)
        print(f"Saved {filename} to {path}")

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()