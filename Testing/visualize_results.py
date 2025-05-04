import os
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
import plotly.graph_objects as go
import re

# Path to your result directory
RESULT_PLOTS_DIR = './RESULT_PLOTS'
ITERATION_VISUALS_DIR = os.path.join(RESULT_PLOTS_DIR, 'iteration_visuals')

def extract_metadata(file_path):
    """
    Extract context (iteration), employee ID, and cluster ID from the file path.
    """
    context_match = re.search(r'Context_(Iteration_\d+)', file_path)
    employee_match = re.search(r'Employee_(E\d+)_Cluster_([-\d]+)\.csv', file_path)

    if context_match and employee_match:
        context = context_match.group(1)
        employee_id = employee_match.group(1)
        cluster_id = employee_match.group(2)
        return context, employee_id, cluster_id

    return None, None, None

def load_all_clusters(result_dir):
    """
    Walk through the result directory and load all CSVs with their metadata.
    """
    cluster_data = []

    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith('.csv') and file.startswith('Employee_'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                context, employee_id, cluster_id = extract_metadata(file_path)

                if context is not None:
                    df['context'] = context
                    df['employee_id'] = employee_id
                    df['cluster_id'] = cluster_id
                    cluster_data.append(df)

    return cluster_data

def plot_clusters_per_iteration(cluster_dataframes, output_dir="RESULT_PLOTS/iteration_visuals"):
    """
    Create one plotly figure per iteration showing all clusters
    with convex hull fills (semi-transparent) and thick contour lines.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique employee IDs for consistent color mapping
    employee_ids = list(set(df['employee_id'].iloc[0] for df in cluster_dataframes))

    # Define a color palette
    color_palette = [
        "rgba(255, 0, 0, 0.2)",      # Red
        "rgba(0, 255, 0, 0.2)",      # Green
        "rgba(0, 0, 255, 0.2)",      # Blue
        "rgba(255, 165, 0, 0.2)",    # Orange
        "rgba(128, 0, 128, 0.2)",    # Purple
        "rgba(0, 255, 255, 0.2)"     # Cyan
    ]

    # Expand palette if more employees than colors
    if len(employee_ids) > len(color_palette):
        additional_colors = [
            f"rgba({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)}, 0.2)"
            for _ in range(len(employee_ids) - len(color_palette))
        ]
        color_palette.extend(additional_colors)

    # Map employee IDs to fill colors + generate border colors
    employee_fill_colors = {emp_id: color_palette[idx] for idx, emp_id in enumerate(sorted(employee_ids))}
    employee_border_colors = {
        emp_id: color.replace('0.2', '1') if 'rgba' in color else color
        for emp_id, color in employee_fill_colors.items()
    }

    # Group data by iteration (context)
    iterations = {}
    for df in cluster_dataframes:
        context = df['context'].iloc[0]
        if context not in iterations:
            iterations[context] = []
        iterations[context].append(df)

    # Process each iteration separately
    for context, dfs_in_iteration in iterations.items():
        fig = go.Figure()

        for df in dfs_in_iteration:
            employee_id = df['employee_id'].iloc[0]
            cluster_id = df['cluster_id'].iloc[0]
            num_points = len(df)

            x = df['x']
            y = df['y']

            # Colors for fill and border
            fill_color = employee_fill_colors.get(employee_id, 'rgba(128,128,128,0.2)')
            border_color = employee_border_colors.get(employee_id, 'rgba(128,128,128,1)')

            # Scatter points inside the cluster with updated legend name
            legend_name = f'{employee_id} - Cluster {cluster_id} (Points: {num_points})'

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=8,
                    color=border_color,
                    line=dict(width=1, color='black')
                ),
                name=legend_name,
                hoverinfo='text',
                text=[f"{context}<br>{employee_id}<br>Cluster {cluster_id}<br>X: {xi:.2f}<br>Y: {yi:.2f}" for xi, yi in zip(x, y)],
                legendgroup=employee_id,
                showlegend=True
            ))

            # Convex hull for the cluster
            if len(df) >= 3:
                multipoint = MultiPoint(df[['x', 'y']].values)
                convex_hull = multipoint.convex_hull

                if convex_hull.geom_type in ['Polygon', 'LineString']:
                    hull_coords = list(convex_hull.exterior.coords) if convex_hull.geom_type == 'Polygon' else list(convex_hull.coords)

                    hull_x = [coord[0] for coord in hull_coords]
                    hull_y = [coord[1] for coord in hull_coords]

                    fig.add_trace(go.Scatter(
                        x=hull_x,
                        y=hull_y,
                        mode='lines',
                        fill='toself',
                        fillcolor=fill_color,            # semi-transparent fill
                        line=dict(
                            color=border_color,         # solid contour
                            width=1                     # thick contour line
                        ),
                        name=f"{employee_id} Hull - Cluster {cluster_id}",
                        hoverinfo='skip',
                        showlegend=False,
                        legendgroup=employee_id
                    ))

        # Layout settings for this iteration
        fig.update_layout(
            title=f'Clusters and Employee Assignments - {context}',
            title_font=dict(size=24),
            xaxis=dict(title="X Coordinate", range=[0, 100], showgrid=True, zeroline=False),
            yaxis=dict(title="Y Coordinate", range=[0, 100], showgrid=True, zeroline=False),
            showlegend=True,
            width=1000,
            height=800
        )

        # Save to HTML
        output_file = os.path.join(output_dir, f"{context}_clusters.html")
        fig.write_html(output_file)
        print(f"Saved plot for {context} to {output_file}")


def main():
    # Load all cluster CSV files
    cluster_dataframes = load_all_clusters(RESULT_PLOTS_DIR)

    if not cluster_dataframes:
        print(f"No cluster CSV files found in {RESULT_PLOTS_DIR}")
        return

    print(f"Loaded {len(cluster_dataframes)} cluster datasets.")

    # Plot one figure per iteration and save
    plot_clusters_per_iteration(cluster_dataframes, ITERATION_VISUALS_DIR)

if __name__ == "__main__":
    main()
