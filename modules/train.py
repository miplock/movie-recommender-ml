# Training module responsible for model training workflow implementation.
import pandas as pd

from modules.models.nmf_model import NMFRecommender
from modules.models.svd1_model import SVD1Recommender


def train_model(train_file: str, model_path: str, alg: str) -> None:
    """Train and save a recommender model from input ratings file.

    EN:
    This function supports NMF and SVD1 algorithms.

    Args:
        train_file (str): Path to CSV file with training ratings.
        model_path (str): Path where serialized model is saved.
        alg (str): Algorithm name requested by CLI.

    Raises:
        NotImplementedError: If unsupported algorithm is requested.

    Returns:
        None: Model is saved to disk as a side effect.

    PL:
    Funkcja obsługuje algorytmy NMF oraz SVD1.

    Argumenty:
        train_file (str): Ścieżka do pliku CSV z danymi treningowymi.
        model_path (str): Ścieżka zapisu zserializowanego modelu.
        alg (str): Nazwa algorytmu wybrana w CLI.

    Wyjątki:
        NotImplementedError: Gdy wybrano nieobsługiwany algorytm.

    Zwraca:
        None: Model jest zapisywany na dysk jako efekt uboczny.
    """
    # Wczytujemy dane treningowe z pliku CSV.
    df = pd.read_csv(train_file)

    # Tworzymy model zgodny z wybranym algorytmem.
    if alg == "NMF":
        model = NMFRecommender(
            n_components=15,
            imputation_strategy="movie_mean",
        )
    elif alg == "SVD1":
        model = SVD1Recommender(
            n_components=15,
            imputation_strategy="movie_mean",
        )
    else:
        raise NotImplementedError(
            f"Algorithm {alg} is not implemented yet."
        )

    # Uczymy model na danych wejściowych.
    model.fit(df)
    # Zapisujemy wytrenowany model do wskazanej ścieżki.
    model.save(model_path)
