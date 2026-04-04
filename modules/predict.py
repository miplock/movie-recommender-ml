# Prediction module responsible for generating and saving rating outputs.
import pandas as pd

from modules.models.nmf_model import NMFRecommender
from modules.models.svd1_model import SVD1Recommender


def predict_ratings(
    input_file: str,
    model_path: str,
    output_file: str,
    alg: str,
) -> None:
    """Generate predictions and save them to output CSV file.

    EN:
    This function supports NMF and SVD1 algorithms.

    Args:
        input_file (str): Path to CSV file with prediction pairs.
        model_path (str): Path to a serialized trained model file.
        output_file (str): Path where generated predictions are saved.
        alg (str): Algorithm name requested by CLI.

    Raises:
        NotImplementedError: If unsupported algorithm is requested.

    Returns:
        None: Predictions are saved to disk as a side effect.

    PL:
    Funkcja obsługuje algorytmy NMF oraz SVD1.

    Argumenty:
        input_file (str): Ścieżka do pliku CSV z parami do predykcji.
        model_path (str): Ścieżka do zapisanego, wytrenowanego modelu.
        output_file (str): Ścieżka zapisu wygenerowanych predykcji.
        alg (str): Nazwa algorytmu wybrana w CLI.

    Wyjątki:
        NotImplementedError: Gdy wybrano nieobsługiwany algorytm.

    Zwraca:
        None: Predykcje są zapisywane na dysk jako efekt uboczny.
    """
    # Wczytujemy wcześniej wytrenowany model zgodny z algorytmem.
    if alg == "NMF":
        model = NMFRecommender.load(model_path)
    elif alg == "SVD1":
        model = SVD1Recommender.load(model_path)
    else:
        raise NotImplementedError(
            f"Algorithm {alg} is not implemented yet."
        )

    # Wczytujemy pary user-film, dla których mają być wyliczone oceny.
    input_df = pd.read_csv(input_file)
    # Generujemy DataFrame wynikowy z przewidzianą kolumną rating.
    output_df = model.predict_dataframe(input_df)
    # Zapisujemy wynik predykcji do pliku CSV bez kolumny indeksu.
    output_df.to_csv(output_file, index=False)
