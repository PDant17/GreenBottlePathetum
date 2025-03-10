import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
from Testing.Clustering.PathetumEnviroment import ClusterEnvironment
from Testing.Clustering.ClusterAnalysis import ClusterAnalysis


class HAClusterAnalysis(ClusterAnalysis):

    def __init__(self, linkage_method='ward', cluster_env=None):
        super().__init__()
        self.linkage_method = linkage_method  # Metodo di collegamento: 'ward', 'complete', 'average', 'single', 'centroid'
        self.cluster_env = cluster_env if cluster_env else ClusterEnvironment()

    def run_hierarchical(self, data, distance_threshold=None, n_clusters=None):
        """
        Esegue il clustering gerarchico con un valore di soglia di distanza o un numero specificato di cluster.
        """
        # Validazione dei parametri
        if distance_threshold is not None and n_clusters is not None:
            raise ValueError("È necessario fornire esattamente uno tra 'distance_threshold' o 'n_clusters', non entrambi.")
        if distance_threshold is None and n_clusters is None:
            # Se non specificato, calcola la migliore soglia di distanza
            distance_threshold = self.calculate_best_distance_threshold(data)
            print(f"Utilizzo della soglia di distanza calcolata: {distance_threshold}")

        # Inizializza il clustering gerarchico con i parametri specificati
        if distance_threshold is not None:
            agglomerative = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold, linkage=self.linkage_method
            )
        else:
            agglomerative = AgglomerativeClustering(
                n_clusters=n_clusters, distance_threshold=None, linkage=self.linkage_method
            )

        # Esegue il clustering e assegna le etichette ai dati
        data['label_cluster'] = agglomerative.fit_predict(data[['x', 'y']])
        return agglomerative, data

    def perform_clustering(self, data, distance_threshold=None, n_clusters=None):
        """
        Esegue il clustering utilizzando il metodo Agglomerative Clustering.
        È possibile specificare una soglia di distanza o un numero di cluster.
        """
        # Esegui il clustering gerarchico
        model, clustered_data = self.run_hierarchical(data, distance_threshold=distance_threshold, n_clusters=n_clusters)

        # Visualizza i risultati usando ClusterEnvironment
        n_clusters_result = len(np.unique(clustered_data['label_cluster']))
        self.cluster_env.update_environment(
            clustered_data, n_clusters=n_clusters_result,
            step_title=f"Clustering Agglomerativo - {n_clusters_result} cluster"
        )

        return model

    def calculate_best_distance_threshold(self, data):
        """
        Calcola la migliore soglia di distanza basata sul gap più grande nella matrice di collegamento.
        """
        # Calcola la matrice di collegamento
        linked = linkage(data[['x', 'y']], method=self.linkage_method)

        # Estrai le distanze dalla matrice di collegamento
        distances = linked[:, 2]  # La colonna 2 contiene le distanze delle fusioni
        sorted_distances = np.sort(distances)

        # Calcola le differenze tra le distanze successive
        distance_gaps = np.diff(sorted_distances)

        # Trova il gap più grande e determina la soglia migliore
        largest_gap_index = np.argmax(distance_gaps)
        best_threshold = sorted_distances[largest_gap_index + 1]  # Soglia subito dopo il gap più grande

        return best_threshold

    def plot_dendrogram(self, data, threshold_suggestion=None):
        """
        Genera un dendrogramma interattivo per il clustering gerarchico utilizzando Plotly.
        """
        # Calcola la matrice di collegamento
        linked = linkage(data[['x', 'y']], method=self.linkage_method)

        # Crea il dendrogramma con Plotly
        fig = ff.create_dendrogram(
            linked,
            orientation='bottom',
            color_threshold=threshold_suggestion
        )

        # Posizioni delle foglie (punti dati individuali)
        leaf_positions = np.array([i for i in range(len(data))])

        # Genera i tick per l'asse X
        labels = list(data.index)  # Usa gli indici dei dati come etichette
        label_interval = 10  # Mostra un'etichetta ogni 10 dati

        sampled_labels = labels[::label_interval]  # Seleziona un'etichetta ogni N dati
        tickvals = leaf_positions[::label_interval]  # Posizioni corrispondenti

        fig.update_layout(
            xaxis=dict(
                tickvals=tickvals,
                ticktext=sampled_labels,
                tickangle=90,  # Ruota le etichette per evitare sovrapposizioni
                tickfont=dict(size=10)  # Regola la dimensione del carattere
            )
        )

        # Aggiunge una linea rossa per indicare la soglia suggerita (se fornita)
        if threshold_suggestion:
            fig.add_shape(
                type="line",
                x0=0,
                x1=len(data) - 1,  # Estende la linea su tutti i punti
                y0=threshold_suggestion,
                y1=threshold_suggestion,
                line=dict(color="red", width=2, dash="dash"),
                xref="x",
                yref="y"
            )

        # Modifica la dimensione della figura per migliorare la leggibilità
        fig.update_layout(
            title=f"Dendrogramma (Collegamento: {self.linkage_method.capitalize()})",
            xaxis_title="Punti Dati",
            yaxis_title="Distanza",
            showlegend=True,
            template="plotly_white",
            width=2000,  # Aumenta la larghezza
            height=800,  # Regola l'altezza
            margin=dict(l=50, r=50, t=50, b=150)  # Aggiunge margini per etichette
        )

        # Mostra il dendrogramma
        fig.show()
