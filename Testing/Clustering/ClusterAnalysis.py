from abc import ABC, abstractmethod
from sklearn.metrics import silhouette_score
import Testing.Dataset.random_points_generator as rpg

class ClusterAnalysis(ABC):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    @abstractmethod
    def perform_clustering(self, data, n_clusters=None):
        """
        Esegue il clustering utilizzando l'algoritmo specifico e restituisce le etichette dei cluster generati.
        """
        pass

    def test_with_datasets(self):
        """
        Genera e testa il clustering su diversi dataset.
        """
        datasets = {
            "Dati Uniformi": rpg.generate_uniform_data(n_points=100, x_range=(0, 100), y_range=(0, 100)),
            "Cluster Gaussiani": rpg.generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5),
            "Cluster Sovrapposti": rpg.generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10),
            "Cluster Non Sferici": rpg.generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10),
            "Cluster con Outlier": rpg.generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20)
        }

        for dataset_name, dataset in datasets.items():
            print(f"\nTest in corso su {dataset_name}...")
            self.perform_clustering(dataset, n_clusters=self.n_clusters)

    def _calculate_silhouette_score(self, data):
        """
        Calcola il silhouette score per i risultati del clustering.
        """
        if len(set(data['label_cluster'])) > 1:
            silhouette_avg = silhouette_score(data[['x', 'y']], data['label_cluster'])
            print(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            print("Silhouette Score: Non disponibile (meno di 2 cluster)")
