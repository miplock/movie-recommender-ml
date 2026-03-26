# Training module responsible for model training workflow implementation.
import pandas as pd

from modules.models.nmf_model import NMFRecommender


def train_model(train_file: str, model_path: str, alg: str) -> None:
    """Train and save an NMF model from input ratings file.

    EN:
    This function currently supports only the NMF algorithm.

    Args:
        train_file (str): Path to CSV file with training ratings.
        model_path (str): Path where serialized model is saved.
        alg (str): Algorithm name requested by CLI.

    Raises:
        NotImplementedError: If algorithm other than NMF is requested.

    Returns:
        None: Model is saved to disk as a side effect.

    PL:
    Funkcja obecnie obsługuje wyłącznie algorytm NMF.

    Argumenty:
        train_file (str): Ścieżka do pliku CSV z danymi treningowymi.
        model_path (str): Ścieżka zapisu zserializowanego modelu.
        alg (str): Nazwa algorytmu wybrana w CLI.

    Wyjątki:
        NotImplementedError: Gdy wybrano algorytm inny niż NMF.

    Zwraca:
        None: Model jest zapisywany na dysk jako efekt uboczny.
    """
    # Na ten moment wspieramy tylko NMF.
    if alg != "NMF":
        raise NotImplementedError(
            f"Algorithm {alg} is not implemented yet."
        )

    # Wczytujemy dane treningowe z pliku CSV.
    df = pd.read_csv(train_file)

    # Tworzymy model NMF z domyślną konfiguracją projektu.
    model = NMFRecommender(
        n_components=15,
        imputation_strategy="movie_mean",
    )
    # Uczymy model na danych wejściowych.
    model.fit(df)
    # Zapisujemy wytrenowany model do wskazanej ścieżki.
    model.save(model_path)
