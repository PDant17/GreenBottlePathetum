from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis
from Testing.Dataset import iteration_dataframe_parser as parser


class DBSCANAnalysis(ClusterAnalysis):
    def __init__(self, eps=5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_env = ClusterEnvironment()

    def perform_clustering(self, data, **kwargs):
        """
        Esegue il clustering DBSCAN e visualizza i risultati.
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        data['label_cluster'] = dbscan.fit_predict(data[['x', 'y']])

        # Verifica il numero di cluster unici (ignorando i punti considerati rumore, ovvero con etichetta -1)
        unique_clusters = set(data['label_cluster'])
        if len(unique_clusters - {-1}) >= 2:  # Almeno 2 cluster (escludendo il rumore)
            silhouette_avg = silhouette_score(
                data[data['label_cluster'] != -1][['x', 'y']],
                data[data['label_cluster'] != -1]['label_cluster']
            )
            print(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            silhouette_avg = None
            print("Silhouette Score: Non disponibile (meno di 2 cluster)")

        # Visualizza i risultati utilizzando ClusterEnvironment
        self.cluster_env.update_environment(data, step_title=f"Clustering DBSCAN (eps={self.eps}, min_samples={self.min_samples})")

        return data, dbscan