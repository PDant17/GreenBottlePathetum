from Testing.Clustering.Algoritmi.KMEANS import KMeansAnalysis
from Testing.Clustering.Algoritmi.DBSCAN import DBSCANAnalysis
from Testing.Clustering.Algoritmi.GerarchicoDivisivo import HDClusterAnalysis
from Testing.Clustering.Algoritmi.GerarchicoAgglomerativo import HAClusterAnalysis
from Testing.Clustering.Algoritmi.KRUSKAL import KruskalClustering  # Assuming KruskalClustering is saved here
from Testing.Dataset import random_points_generator as rpg
from Testing.Dataset import iteration_dataframe_parser as parser



def generate_datasets():
    """
    Generate datasets to test clustering algorithms.
    """
    datasets = {
        "Gaussian Clusters": rpg.generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5),
        "Uniform Data": rpg.generate_uniform_data(n_points=300, x_range=(0, 100), y_range=(0, 100)),
        "Overlapping Clusters": rpg.generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10),
        "Non-Spherical Clusters": rpg.generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10),
        "Clusters with Outliers": rpg.generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20),
        "Grid Clusters": rpg.generate_grid_clusters()
    }
    return datasets

# Example execution for KMeans
def kmeans_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    kmeans_analysis = KMeansAnalysis(n_clusters=20, max_clusters=60)

    # Perform clustering on each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with KMeans...")
        data, kmeans = kmeans_analysis.perform_clustering(dataset)
        data, kmeans = kmeans_analysis.perform_clustering(dataset, False)

# Example execution for DBSCAN
def dbscan_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    dbscan_analysis = DBSCANAnalysis(eps=10, min_samples=5)

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with DBSCAN...")
        data, dbscan = dbscan_analysis.perform_clustering(dataset)

# Example execution for Divisive Clustering
def divisive_clustering_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    divisive_analysis = HDClusterAnalysis()

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with Divisive Clustering...")
        divisive_analysis.perform_clustering(dataset)

# Example execution for Agglomerative Clustering
def agglomerative_clustering_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    # Linkage methods to test
    linkage_methods = ["single", "complete", "average", "ward"]

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with Agglomerative Clustering...")

        for linkage in linkage_methods:
            print(f"\nUsing {linkage.capitalize()} Linkage:")

            ha_analysis = HAClusterAnalysis(linkage_method=linkage)

            model = ha_analysis.perform_clustering(dataset)

            best_threshold = ha_analysis.calculate_best_distance_threshold(dataset)
            print(
                f"Automatically calculated best distance threshold for {linkage.capitalize()} Linkage: {best_threshold:.2f}")

            print(f"Plotting dendrogram for {linkage.capitalize()} Linkage...")
            ha_analysis.plot_dendrogram(dataset, threshold_suggestion=best_threshold)

# Example execution for Kruskal Clustering
def kruskal_example(datasets):
    if datasets is None:
        datasets = generate_datasets()

    kruskal_analysis = KruskalClustering(n_clusters=9)

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting {dataset_name} with Kruskal Clustering...")
        clustered_data = kruskal_analysis.perform_clustering(dataset)

        print(f"Finished clustering for {dataset_name}.")
        kruskal_analysis.visualize_mst(dataset)  # Optional MST visualization


def main():

    #datasets = generate_datasets()
    datasets = parser.load_and_parse_iterations("./Data/hand_picked_points.csv")


    #print("\nRunning DBSCAN Example:")
    #dbscan_example(datasets)

    print("\nRunning Divisive Clustering Example:")
    divisive_clustering_example(datasets)

    print("\nRunning KMeans Example:")
    kmeans_example(datasets)

    #print("\nRunning Agglomerative Clustering Example:")
    #agglomerative_clustering_example(datasets)

    #print("\nRunning Kruskal Clustering Example:")
    #kruskal_example(datasets)

if __name__ == "__main__":
    main()