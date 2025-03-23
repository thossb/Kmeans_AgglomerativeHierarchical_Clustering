import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found in the path: {filepath}")

    # Load the CSV file with '\t' separator
    df = pd.read_csv(filepath, sep="\t")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_features]

    if data.shape[1] == 0:
        raise ValueError("No numeric features available. Check the CSV file format.")

    # Fill missing values and standardize the data
    data = data.fillna(data.median())
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def agglomerative_clustering(data_scaled, n_clusters, linkages=['single', 'complete', 'average', 'ward']):
    """
    Perform Agglomerative Hierarchical Clustering with different linkage methods.
    """
    clustering_results = {}
    labels_dict = {}

    for linkage in linkages:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        clustering_results[linkage] = score
        labels_dict[linkage] = labels
        print(f"Agglomerative Clustering with {linkage} linkage: Silhouette Score = {score:.4f}")

    print("\n")
    return clustering_results, labels_dict


def kmeans_clustering(data_scaled, k_range=range(2, 11), random_state=42):
    """
    Perform k-Means clustering with silhouette analysis.
    """
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        silhouette_scores.append(score)
        print(f"Silhouette Score for k={k}: {score:.4f}")

    # Determine optimal k based on maximum silhouette score.
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nOptimal k based on silhouette score: {optimal_k}\n")
    return optimal_k


def main():
    # Filepath of the dataset
    filepath = r"C:\Kuliah\Pelajaran\Semester_8\Pmes\Tugas2\marketing_campaign.csv"
    data_scaled = load_and_preprocess_data(filepath)

    # Perform k-Means clustering
    optimal_k = kmeans_clustering(data_scaled)

    # Perform Agglomerative Clustering with multiple methods
    linkages = ['single', 'complete', 'average', 'ward']
    agglomerative_results, _ = agglomerative_clustering(data_scaled, optimal_k, linkages)

    print("\nHierarchical Clustering Results (Silhouette Scores):")
    for linkage, score in agglomerative_results.items():
        print(f"{linkage.capitalize()} Linkage: Silhouette Score = {score:.4f}")


if __name__ == "__main__":
    main()
