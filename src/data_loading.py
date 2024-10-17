# src/data_loading.py
import pandas as pd

# Load the datasets
data_mat = pd.read_csv('../data/student-mat.csv', sep=';')
data_por = pd.read_csv('../data/student-por.csv', sep=';')

# Merge the datasets if necessary
data = pd.concat([data_mat, data_por], ignore_index=True)

# Display basic information
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())
print("First few rows:\n", data.head())
