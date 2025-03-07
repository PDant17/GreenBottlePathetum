import pandas as pd


def load_and_parse_iterations(csv_file_path):
    """
    Carica un file CSV e lo suddivide in sotto-dataframe in base alla colonna 'iteration'.

    Parametri:
        csv_file_path (str): Percorso del file CSV da caricare.

    Ritorna:
        dict: Un dizionario in cui le chiavi sono i valori unici della colonna 'iteration'
              e i valori sono i dataframe corrispondenti a ciascuna iterazione.
    """

    df = pd.read_csv(csv_file_path)

    if 'iteration' not in df.columns:
        print(f"Errore: La colonna 'iteration' non esiste nel file CSV in {csv_file_path}.")
        return None

    iterations_data = {}

    iterations = df['iteration'].unique()

    for iteration in iterations:
        iteration_df = df[df['iteration'] == iteration].copy()
        iterations_data[iteration] = iteration_df

        print(f"Dati Iterazione {iteration}:\n", iteration_df.head(), "\n")

    return iterations_data
