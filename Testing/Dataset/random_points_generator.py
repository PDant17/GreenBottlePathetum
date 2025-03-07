import numpy as np
import pandas as pd
import requests
import json
import random

"""

Questo script in Python è stato realizzato per dimostrare il variare del comportamento degli algoritmi di clustering
in base alla distribuzione dei punti.

"""

def load_api_key(config_file='random_config.json'):
    """
    Carica la chiave API dal file di configurazione.

    Parametri:
        config_file (str): Percorso del file di configurazione JSON.

    Ritorna:
        str o None: La chiave API se trovata, altrimenti None.
    """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            return config.get("api_key", None)
    except FileNotFoundError:
        print(f"File di configurazione {config_file} non trovato.")
        return None
    except json.JSONDecodeError:
        print("Errore nella decodifica del file di configurazione JSON.")
        return None


def get_random_seed(config_file='random_config.json'):
    """
    Ottiene un numero casuale utilizzando l'API Random.org (metodo generateIntegers).
    Se la chiamata API fallisce, utilizza il modulo random di Python come fallback.

    Parametri:
        config_file (str): Percorso del file di configurazione JSON.

    Ritorna:
        int: Un numero casuale intero.
    """
    api_key = load_api_key(config_file)
    if api_key is None:
        print("Chiave API non trovata. Uso il metodo di fallback.")
        return random.randint(0, 1000000)

    url = "https://api.random.org/json-rpc/2/invoke"
    headers = {"Content-Type": "application/json"}

    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": api_key,
            "n": 1,
            "min": 0,
            "max": 1000000,
            "replacement": True
        },
        "id": 42
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_data = response.json()

        if "result" in response_data:
            return response_data["result"]["random"]["data"][0]
        else:
            print("Errore nella risposta dell'API:", response_data)
            return random.randint(0, 1000000)
    except requests.RequestException as e:
        print("Richiesta HTTP fallita:", e)
        return random.randint(0, 1000000)
    except json.JSONDecodeError:
        print("Errore nella decodifica della risposta JSON.")
        return random.randint(0, 1000000)


def generate_uniform_data(n_points=100, x_range=(0, 100), y_range=(0, 100)):
    """
    Genera punti distribuiti uniformemente in un'area specificata.

    Parametri:
        n_points (int): Numero di punti da generare.
        x_range (tuple): Range dei valori X.
        y_range (tuple): Range dei valori Y.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(get_random_seed())
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    return pd.DataFrame({'x': x, 'y': y})


def generate_overlapping_clusters(n_clusters=3, n_points_per_cluster=100, overlap=10, random_seed=get_random_seed()):
    """
    Genera cluster sovrapposti con distribuzione normale attorno a centri casuali.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        overlap (float): Deviazione standard per la sovrapposizione dei cluster.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(40, 60), np.random.uniform(40, 60)
        points_x = np.random.normal(center_x, overlap, n_points_per_cluster)
        points_y = np.random.normal(center_y, overlap, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)



def generate_gaussian_clusters(n_clusters=3, n_points_per_cluster=100, cluster_spread=5, random_seed=get_random_seed()):
    """
    Genera cluster Gaussiani con centri casuali.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        cluster_spread (float): Deviazione standard per la distribuzione dei punti.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        points_x = np.random.normal(center_x, cluster_spread, n_points_per_cluster)
        points_y = np.random.normal(center_y, cluster_spread, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)


def generate_non_spherical_clusters(n_clusters=3, n_points_per_cluster=100, elongation=10,
                                    random_seed=get_random_seed()):
    """
    Genera cluster allungati in una direzione per simulare forme non sferiche.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        elongation (float): Allungamento lungo l'asse X.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        points_x = np.random.normal(center_x, elongation, n_points_per_cluster)
        points_y = np.random.normal(center_y, 1, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)


def generate_clusters_with_outliers(n_clusters=3, n_points_per_cluster=100, n_outliers=20,
                                    random_seed=get_random_seed()):
    """
    Genera cluster Gaussiani con l'aggiunta di punti outlier.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        n_outliers (int): Numero di punti outlier casuali.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati con outlier.
    """
    np.random.seed(random_seed)
    data = generate_gaussian_clusters(n_clusters, n_points_per_cluster, cluster_spread=5, random_seed=random_seed)
    outliers_x = np.random.uniform(0, 100, n_outliers)
    outliers_y = np.random.uniform(0, 100, n_outliers)
    outliers = pd.DataFrame({'x': outliers_x, 'y': outliers_y})
    return pd.concat([data, outliers], ignore_index=True)



def generate_density_clusters(n_clusters=3, n_points_per_cluster=100, density_factor=5, random_seed=get_random_seed()):
    """
    Genera cluster con aree a diversa densità di punti.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        density_factor (float): Fattore che determina la densità dei punti nel cluster.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        dense_points_x = np.random.normal(center_x, density_factor, int(n_points_per_cluster * 0.7))
        dense_points_y = np.random.normal(center_y, density_factor, int(n_points_per_cluster * 0.7))
        sparse_points_x = np.random.normal(center_x, density_factor * 3, int(n_points_per_cluster * 0.3))
        sparse_points_y = np.random.normal(center_y, density_factor * 3, int(n_points_per_cluster * 0.3))
        data.append(pd.DataFrame({'x': np.concatenate([dense_points_x, sparse_points_x]),
                                  'y': np.concatenate([dense_points_y, sparse_points_y])}))
    return pd.concat(data, ignore_index=True)


def generate_cluster_chains(n_clusters=3, n_points_per_cluster=100, chain_length=50, random_seed=get_random_seed()):
    """
    Genera cluster disposti in catene allineate casualmente.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        chain_length (float): Lunghezza massima dello spostamento tra cluster consecutivi.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        start_x, start_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        end_x, end_y = start_x + np.random.uniform(-chain_length, chain_length), start_y + np.random.uniform(-chain_length, chain_length)
        t = np.linspace(0, 1, n_points_per_cluster)
        points_x = start_x * (1 - t) + end_x * t + np.random.normal(0, 1, n_points_per_cluster)
        points_y = start_y * (1 - t) + end_y * t + np.random.normal(0, 1, n_points_per_cluster)
        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))
    return pd.concat(data, ignore_index=True)


def generate_hierarchical_clusters(n_clusters=3, n_points_per_cluster=100, subcluster_ratio=0.2, random_seed=get_random_seed()):
    """
    Genera cluster gerarchici con sotto-cluster interni.

    Parametri:
        n_clusters (int): Numero di cluster principali.
        n_points_per_cluster (int): Numero di punti per cluster.
        subcluster_ratio (float): Percentuale di punti assegnati ai sotto-cluster.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []
    for i in range(n_clusters):
        center_x, center_y = np.random.uniform(20, 80), np.random.uniform(20, 80)
        # Cluster principale
        main_points_x = np.random.normal(center_x, 5, int(n_points_per_cluster * (1 - subcluster_ratio)))
        main_points_y = np.random.normal(center_y, 5, int(n_points_per_cluster * (1 - subcluster_ratio)))
        data.append(pd.DataFrame({'x': main_points_x, 'y': main_points_y}))
        # Sotto-cluster
        subcluster_x = np.random.normal(center_x + np.random.uniform(-5, 5), 2, int(n_points_per_cluster * subcluster_ratio))
        subcluster_y = np.random.normal(center_y + np.random.uniform(-5, 5), 2, int(n_points_per_cluster * subcluster_ratio))
        data.append(pd.DataFrame({'x': subcluster_x, 'y': subcluster_y}))
    return pd.concat(data, ignore_index=True)


def generate_ring_clusters(n_clusters=3, n_points_per_cluster=100, radius_range=(10, 50), random_seed=None):
    """
    Genera cluster a forma di anello distribuiti casualmente.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        radius_range (tuple): Intervallo per i raggi degli anelli.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []
    for i in range(n_clusters):
        center_x = np.random.uniform(10, 90)
        center_y = np.random.uniform(10, 90)
        radius = np.random.uniform(*radius_range)
        angles = np.random.uniform(0, 2 * np.pi, n_points_per_cluster)
        points_x = radius * np.cos(angles) + np.random.normal(0, 1, n_points_per_cluster)
        points_y = radius * np.sin(angles) + np.random.normal(0, 1, n_points_per_cluster)

        points_x += center_x
        points_y += center_y

        points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min()) * 100
        points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min()) * 100

        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))

    return pd.concat(data, ignore_index=True)


def generate_spiral_clusters(n_clusters=3, n_points_per_cluster=100, random_seed=None):
    """
    Genera cluster a forma di spirale con centri casuali.

    Parametri:
        n_clusters (int): Numero di cluster.
        n_points_per_cluster (int): Numero di punti per cluster.
        random_seed (int): Valore del seme casuale.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []
    for i in range(n_clusters):
        center_x = np.random.uniform(10, 90)
        center_y = np.random.uniform(10, 90)

        theta = np.linspace(0, 4 * np.pi, n_points_per_cluster) + (2 * np.pi * i / n_clusters)
        r = theta + np.random.normal(0, 0.5, n_points_per_cluster)
        points_x = r * np.cos(theta) + np.random.normal(0, 0.2, n_points_per_cluster)
        points_y = r * np.sin(theta) + np.random.normal(0, 0.2, n_points_per_cluster)

        points_x += center_x
        points_y += center_y

        points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min()) * 100
        points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min()) * 100

        data.append(pd.DataFrame({'x': points_x, 'y': points_y}))

    return pd.concat(data, ignore_index=True)


def generate_grid_clusters(n_clusters_per_side=3, n_points_per_cluster=100, jitter=2, random_seed=get_random_seed()):
    """
    Genera cluster disposti in una griglia regolare con un certo livello di rumore (jitter).

    Parametri:
        n_clusters_per_side (int): Numero di cluster per lato della griglia (totale: n_clusters_per_side^2).
        n_points_per_cluster (int): Numero di punti per ogni cluster.
        jitter (float): Deviazione standard per la dispersione dei punti attorno ai centri della griglia.
        random_seed (int): Valore del seme casuale per garantire riproducibilità.

    Ritorna:
        DataFrame: Contiene le coordinate dei punti generati.
    """
    np.random.seed(random_seed)
    data = []


    grid_x, grid_y = np.linspace(20, 80, n_clusters_per_side), np.linspace(20, 80, n_clusters_per_side)

    for cx in grid_x:
        for cy in grid_y:
            points_x = np.random.normal(cx, jitter, n_points_per_cluster)
            points_y = np.random.normal(cy, jitter, n_points_per_cluster)
            data.append(pd.DataFrame({'x': points_x, 'y': points_y}))

    return pd.concat(data, ignore_index=True)





