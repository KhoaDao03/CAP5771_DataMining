# src/k_means_clustering.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib

def load_data(project_dir):
    """
    Load the preprocessed data and scale it.
    """
    preprocessed_data = pd.read_csv(os.path.join(project_dir, 'outputs', 'preprocessed_data.csv'))

    # Exclude target variables
    data = preprocessed_data.drop(columns=['G3', 'pass'])

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler

def k_means_clustering(data_scaled, project_dir, n_clusters=3):
    """
    Perform K-Means clustering, evaluate, and visualize the results.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    labels = kmeans.labels_

    # Evaluate clustering
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
    davies_bouldin = davies_bouldin_score(data_scaled, labels)

    print(f"Evaluation Metrics for K-Means with {n_clusters} clusters:")
    print(f"Inertia (Within-Cluster Sum of Squares): {inertia:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}\n")

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        hue=labels,
        palette='Set1',
        alpha=0.6
    )
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    figures_dir = os.path.join(project_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'kmeans_clustering_{n_clusters}_clusters.png'))
    plt.close()

    # Analyze cluster distribution
    cluster_counts = pd.Series(labels).value_counts()
    print(f"Cluster counts:\n{cluster_counts}\n")

    return kmeans, pca

def main():
    """
    Main function to perform K-Means Clustering.
    """
    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Load and scale data
    data_scaled, scaler = load_data(project_dir)

    # Perform K-Means Clustering
    n_clusters = 3  # adjustable
    kmeans_model, pca_model = k_means_clustering(data_scaled, project_dir, n_clusters=n_clusters)

    # Save models
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({'kmeans': kmeans_model, 'scaler': scaler, 'pca': pca_model}, os.path.join(models_dir, 'kmeans_model.pkl'))
    print("K-Means model, scaler, and PCA transformer saved.")

if __name__ == "__main__":
    main()
