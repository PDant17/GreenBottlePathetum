import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
import plotly.graph_objects as go


class ClusterEnvironment:
    def __init__(self):
        self.data = pd.DataFrame(columns=["x", "y", "label_cluster"])

    def update_environment(self, new_data, n_clusters=3, step_title="Algorithm Step"):
        """
        Aggiorna l'ambiente con nuovi dati e visualizza i risultati.
        """

        if isinstance(new_data, list):
            new_data = pd.DataFrame(new_data, columns=["x", "y"])


        if not {"x", "y"}.issubset(new_data.columns):
            raise ValueError("I dati di input devono includere le colonne 'x' e 'y'.")


        self.data = new_data.copy()
        self._visualize(step_title)

    def _visualize(self, title="Clustered Data"):
        """
        Visualizza i dati raggruppati e disegna gli involucri convessi dei cluster.
        """

        cluster_polygons = []


        for cluster_label in self.data["label_cluster"].unique():
            if cluster_label == -1: # Salta i punti considerati "outlier" (etichettati con -1)

                continue


            cluster_points = self.data[self.data["label_cluster"] == cluster_label][["x", "y"]]


            multipoint = MultiPoint(cluster_points.values)
            convex_hull = multipoint.convex_hull


            if convex_hull.geom_type == 'Polygon':
                hull_coords = list(convex_hull.exterior.coords)
            elif convex_hull.geom_type == 'LineString':
                hull_coords = list(convex_hull.coords)
            else:
                continue


            hull_x = [coord[0] for coord in hull_coords]
            hull_y = [coord[1] for coord in hull_coords]


            cluster_color = f"rgba({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)}, 0.3)"


            cluster_polygons.append(go.Scatter(
                x=hull_x,
                y=hull_y,
                fill="toself",
                fillcolor=cluster_color,
                line=dict(width=2, color='rgba(0,0,0,0)'),
                name=f"Cluster {cluster_label}",
                hoverinfo="text"
            ))


        scatter = go.Scatter(
            x=self.data["x"],
            y=self.data["y"],
            mode="markers",
            marker=dict(
                size=12,
                color=self.data["label_cluster"],
                colorscale="Rainbow",
                opacity=0.8,
                line=dict(width=2, color='black'),
                showscale=False
            ),
            text=self.data["label_cluster"].apply(lambda x: "Outliers" if x == -1 else f"Cluster {x}"),
            hoverinfo="text+x+y",
            name="Data Points"
        )

        # Impostazioni di Layout del plot.
        layout = go.Layout(
            title=title,
            title_font=dict(size=24, family='Arial, sans-serif', color='black'),
            xaxis=dict(title="X", range=[0, 100], showgrid=False, zeroline=False),
            yaxis=dict(title="Y", range=[0, 100], showgrid=False, zeroline=False),
            showlegend=True,
            font=dict(color='black'),
            hoverlabel=dict(bgcolor='white', font=dict(color='black')),
            shapes=[{
                'type': 'rect',
                'x0': 0,
                'y0': 0,
                'x1': 1,
                'y1': 1,
                'xref': 'paper',
                'yref': 'paper',
                'fillcolor': 'rgba(0, 0, 0, 0.1)',
                'layer': 'below',
                'line': {'width': 0}
            }],
            xaxis_showgrid=True,
            yaxis_showgrid=True
        )


        fig = go.Figure(data=[scatter] + cluster_polygons, layout=layout)
        fig.show()
