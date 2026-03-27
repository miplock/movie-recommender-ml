# Base recommender module with shared utilities for derived models.
import os
import pickle

import numpy as np
import pandas as pd


class BaseRecommender:
    """Base class for recommenders with shared utility methods.

    EN:
    Provides common validation, mapping, fallback prediction, persistence,
    and post-processing logic used by concrete recommender models.

    PL:
    Klasa bazowa dla rekomenderów z metodami pomocniczymi współdzielonymi
    przez konkretne modele.
    """

    def __init__(self) -> None:
        """Initialize empty model state shared by child classes.

        EN:
        Creates placeholders for ID mappings, summary statistics, and fitted
        state flag.

        Returns:
            None: Initializes object attributes in place.

        PL:
        Tworzy pola na mapowania identyfikatorów, statystyki zbioru oraz
        flagę informującą, czy model został wytrenowany.

        Zwraca:
            None: Inicjalizuje atrybuty obiektu.
        """
        # Mapowanie userId -> indeks wewnętrzny.
        self.user_to_index = None
        # Mapowanie movieId -> indeks wewnętrzny.
        self.movie_to_index = None
        # Mapowanie odwrotne indeks -> userId.
        self.index_to_user = None
        # Mapowanie odwrotne indeks -> movieId.
        self.index_to_movie = None

        # Średnia globalna ocena w zbiorze treningowym.
        self.global_mean = None
        # Średnie oceny liczone dla poszczególnych użytkowników.
        self.user_means = None
        # Średnie oceny liczone dla poszczególnych filmów.
        self.movie_means = None

        # Flaga informująca, czy model przeszedł trening.
        self.is_fitted = False
        # Nazwa algorytmu raportowana w metadanych eksperymentu.
        self.algorithm_name = self.__class__.__name__
        # Słownik z hiperparametrami modelu ustawianymi w podklasie.
        self.model_params = {}

    def fit(self, ratings_df: pd.DataFrame) -> "BaseRecommender":
        """Fit model on training ratings.

        EN:
        Abstract method implemented by subclasses.

        Args:
            ratings_df (pd.DataFrame): DataFrame with training ratings.

        Returns:
            BaseRecommender: Trained model instance.

        Raises:
            NotImplementedError: Always raised in base class.

        PL:
        Metoda abstrakcyjna implementowana w klasach dziedziczących.

        Argumenty:
            ratings_df (pd.DataFrame): DataFrame z ocenami treningowymi.

        Zwraca:
            BaseRecommender: Wytrenowaną instancję modelu.

        Wyjątki:
            NotImplementedError: Zawsze zgłaszany w klasie bazowej.
        """
        # Metoda ma być nadpisana w klasie potomnej.
        raise NotImplementedError

    def _predict_known_pair(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a known user-movie pair.

        EN:
        Abstract method implemented by subclasses.

        Args:
            user_id (int): User identifier.
            movie_id (int): Movie identifier.

        Returns:
            float: Predicted rating value.

        Raises:
            NotImplementedError: Always raised in base class.

        PL:
        Metoda abstrakcyjna implementowana w klasach dziedziczących.

        Argumenty:
            user_id (int): Identyfikator użytkownika.
            movie_id (int): Identyfikator filmu.

        Zwraca:
            float: Przewidziana wartość oceny.

        Wyjątki:
            NotImplementedError: Zawsze zgłaszany w klasie bazowej.
        """
        # Metoda ma być nadpisana w klasie potomnej.
        raise NotImplementedError

    def predict(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for user-movie pairs.

        EN:
        Validates input, predicts ratings with model or cold-start fallback,
        and applies clipping and rounding.

        Args:
            pairs_df (pd.DataFrame): Input pairs with userId and movieId.

        Returns:
            np.ndarray: Predicted ratings.

        Raises:
            ValueError: If model is not fitted.
            ValueError: If required columns are missing.

        PL:
        Waliduje wejście, wylicza oceny przez model lub fallback cold-start
        oraz stosuje ograniczanie i zaokrąglanie.

        Argumenty:
            pairs_df (pd.DataFrame): Dane wejściowe z userId i movieId.

        Zwraca:
            np.ndarray: Przewidziane oceny.

        Wyjątki:
            ValueError: Gdy model nie jest wytrenowany.
            ValueError: Gdy brakuje wymaganych kolumn.
        """
        # Najpierw upewniamy się, że model jest gotowy do predykcji.
        self._check_is_fitted()
        # Walidujemy schemat wejścia dla danych predykcyjnych.
        self._validate_pair_columns(pairs_df)

        # Lista na przewidziane oceny dla kolejnych par.
        predictions = []

        # Iterujemy po parach (userId, movieId) z DataFrame wejściowego.
        for row in pairs_df.itertuples(index=False):
            # Odczytujemy userId z bieżącego rekordu.
            user_id = row.userId
            # Odczytujemy movieId z bieżącego rekordu.
            movie_id = row.movieId

            # Dla znanych obiektów używamy predyktora modelu.
            if self._is_known_user(user_id) and self._is_known_movie(movie_id):
                pred = self._predict_known_pair(user_id, movie_id)
            else:
                # Dla nowych obiektów używamy fallbacku cold-start.
                pred = self._cold_start_prediction(user_id, movie_id)

            # Ograniczamy wynik do dopuszczalnego zakresu ocen.
            pred = self._clip_rating(pred)
            # Zaokrąglamy ocenę do najbliższej połówki.
            pred = self._round_to_half(pred)
            # Dodajemy gotową ocenę do listy wyników.
            predictions.append(pred)

        # Zwracamy wyniki jako tablicę NumPy.
        return np.array(predictions)

    def predict_dataframe(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a DataFrame with rating column.

        EN:
        Copies input pairs and appends predicted rating values.

        Args:
            pairs_df (pd.DataFrame): Input pairs with userId and movieId.

        Returns:
            pd.DataFrame: Copy of input with added rating column.

        PL:
        Kopiuje pary wejściowe i dodaje kolumnę z przewidzianymi ocenami.

        Argumenty:
            pairs_df (pd.DataFrame): Dane wejściowe z userId i movieId.

        Zwraca:
            pd.DataFrame: Kopię wejścia z dodatkową kolumną rating.
        """
        # Tworzymy kopię, aby nie modyfikować obiektu wejściowego.
        result = pairs_df.copy()
        # Wyliczamy predykcje i dopisujemy je jako kolumnę "rating".
        result["rating"] = self.predict(pairs_df)
        # Zwracamy DataFrame gotowy do zapisu.
        return result

    def save(self, model_path: str) -> None:
        """Serialize model instance to disk.

        EN:
        Ensures parent directory exists and stores model as pickle.

        Args:
            model_path (str): Target path for saved model file.

        Returns:
            None: Model is written to disk as a side effect.

        PL:
        Upewnia się, że katalog istnieje, i zapisuje model jako pickle.

        Argumenty:
            model_path (str): Docelowa ścieżka pliku modelu.

        Zwraca:
            None: Model jest zapisywany na dysk jako efekt uboczny.
        """
        # Pobieramy ścieżkę katalogu nadrzędnego pliku modelu.
        parent = os.path.dirname(model_path)
        # Tworzymy katalog, jeśli jeszcze nie istnieje.
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Otwieramy plik modelu do zapisu binarnego.
        with open(model_path, "wb") as f:
            # Serializujemy bieżącą instancję modelu do pliku.
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str) -> "BaseRecommender":
        """Load serialized model instance from disk.

        EN:
        Restores model object from pickle file.

        Args:
            model_path (str): Path to saved model file.

        Returns:
            BaseRecommender: Loaded model instance.

        PL:
        Odtwarza obiekt modelu z pliku pickle.

        Argumenty:
            model_path (str): Ścieżka do zapisanego pliku modelu.

        Zwraca:
            BaseRecommender: Wczytaną instancję modelu.
        """
        # Otwieramy plik modelu do odczytu binarnego.
        with open(model_path, "rb") as f:
            # Deserializujemy obiekt modelu i zwracamy go do użycia.
            return pickle.load(f)

    def get_model_metadata(self) -> dict[str, object]:
        """Return unified model metadata for experiment logging.

        EN:
        Exposes algorithm name, concrete class name, and parameter dict.

        Returns:
            dict[str, object]: Serializable model metadata.

        PL:
        Zwraca zunifikowane metadane modelu do logowania eksperymentów.

        Zwraca:
            dict[str, object]: Serializowalne metadane modelu.
        """
        return {
            "algorithm_name": self.algorithm_name,
            "model_class": self.__class__.__name__,
            "model_params": dict(self.model_params),
        }

    def _build_mappings(self, ratings_df: pd.DataFrame) -> None:
        """Create ID-to-index and index-to-ID mappings.

        EN:
        Builds deterministic mappings for users and movies from input data.

        Args:
            ratings_df (pd.DataFrame): DataFrame with userId and movieId.

        Returns:
            None: Mapping attributes are updated in place.

        PL:
        Buduje deterministyczne mapowania użytkowników i filmów na indeksy.

        Argumenty:
            ratings_df (pd.DataFrame): DataFrame z userId i movieId.

        Zwraca:
            None: Atrybuty mapowań są aktualizowane w obiekcie.
        """
        # Pobieramy unikalne identyfikatory użytkowników i sortujemy je.
        user_ids = sorted(ratings_df["userId"].unique())
        # Pobieramy unikalne identyfikatory filmów i sortujemy je.
        movie_ids = sorted(ratings_df["movieId"].unique())

        # Budujemy mapę userId -> indeks.
        self.user_to_index = {
            user_id: idx for idx, user_id in enumerate(user_ids)
        }
        # Budujemy mapę movieId -> indeks.
        self.movie_to_index = {
            movie_id: idx for idx, movie_id in enumerate(movie_ids)
        }

        # Budujemy mapę odwrotną indeks -> userId.
        self.index_to_user = {
            idx: user_id for user_id, idx in self.user_to_index.items()
        }
        # Budujemy mapę odwrotną indeks -> movieId.
        self.index_to_movie = {
            idx: movie_id for movie_id, idx in self.movie_to_index.items()
        }

    def _compute_basic_statistics(self, ratings_df: pd.DataFrame) -> None:
        """Compute global, user, and movie mean ratings.

        EN:
        Calculates summary statistics used by cold-start fallback logic.

        Args:
            ratings_df (pd.DataFrame): DataFrame with userId, movieId, rating.

        Returns:
            None: Statistics are stored in object attributes.

        PL:
        Wylicza statystyki globalne oraz średnie użytkowników i filmów.

        Argumenty:
            ratings_df (pd.DataFrame): DataFrame z userId, movieId, rating.

        Zwraca:
            None: Statystyki są zapisywane w atrybutach obiektu.
        """
        # Liczymy globalną średnią ocen z całego zbioru.
        self.global_mean = float(ratings_df["rating"].mean())
        # Liczymy średnie oceny dla każdego użytkownika.
        self.user_means = (
            ratings_df.groupby("userId")["rating"].mean().to_dict()
        )
        # Liczymy średnie oceny dla każdego filmu.
        self.movie_means = (
            ratings_df.groupby("movieId")["rating"].mean().to_dict()
        )

    def _validate_rating_columns(self, ratings_df: pd.DataFrame) -> None:
        """Validate required columns for training data.

        EN:
        Requires userId, movieId, and rating columns.

        Args:
            ratings_df (pd.DataFrame): DataFrame to validate.

        Returns:
            None: Validation only.

        Raises:
            ValueError: If required columns are missing.

        PL:
        Waliduje wymagane kolumny danych treningowych: userId, movieId,
        rating.

        Argumenty:
            ratings_df (pd.DataFrame): DataFrame do walidacji.

        Zwraca:
            None: Funkcja tylko waliduje dane.

        Wyjątki:
            ValueError: Gdy brakuje wymaganych kolumn.
        """
        # Definiujemy wymagane kolumny dla danych treningowych.
        required_columns = {"userId", "movieId", "rating"}
        # Sprawdzamy, których kolumn brakuje w wejściu.
        missing = required_columns - set(ratings_df.columns)
        # Zgłaszamy błąd, jeśli dane mają niepoprawny schemat.
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _validate_pair_columns(self, pairs_df: pd.DataFrame) -> None:
        """Validate required columns for prediction pairs.

        EN:
        Requires userId and movieId columns.

        Args:
            pairs_df (pd.DataFrame): DataFrame to validate.

        Returns:
            None: Validation only.

        Raises:
            ValueError: If required columns are missing.

        PL:
        Waliduje wymagane kolumny danych do predykcji: userId i movieId.

        Argumenty:
            pairs_df (pd.DataFrame): DataFrame do walidacji.

        Zwraca:
            None: Funkcja tylko waliduje dane.

        Wyjątki:
            ValueError: Gdy brakuje wymaganych kolumn.
        """
        # Definiujemy wymagane kolumny dla danych predykcyjnych.
        required_columns = {"userId", "movieId"}
        # Sprawdzamy, których kolumn brakuje w wejściu.
        missing = required_columns - set(pairs_df.columns)
        # Zgłaszamy błąd, jeśli dane mają niepoprawny schemat.
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _check_is_fitted(self) -> None:
        """Ensure the model has been fitted before prediction.

        EN:
        Checks fitted state flag and raises an error when model is untrained.

        Returns:
            None: Validation only.

        Raises:
            ValueError: If model has not been fitted.

        PL:
        Sprawdza flagę wytrenowania i zgłasza błąd dla modelu
        niewytrenowanego.

        Zwraca:
            None: Funkcja tylko waliduje stan modelu.

        Wyjątki:
            ValueError: Gdy model nie został wytrenowany.
        """
        # Blokujemy predykcję, jeśli model nie został wcześniej dopasowany.
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

    def _is_known_user(self, user_id: int) -> bool:
        """Check whether user identifier is known to model mappings.

        EN:
        Returns True when user_id exists in training mappings.

        Args:
            user_id (int): User identifier.

        Returns:
            bool: True if user is known, otherwise False.

        PL:
        Sprawdza, czy identyfikator użytkownika istnieje w mapowaniach.

        Argumenty:
            user_id (int): Identyfikator użytkownika.

        Zwraca:
            bool: True gdy użytkownik jest znany, w przeciwnym razie False.
        """
        # Sprawdzamy obecność user_id w mapowaniu użytkowników.
        return user_id in self.user_to_index

    def _is_known_movie(self, movie_id: int) -> bool:
        """Check whether movie identifier is known to model mappings.

        EN:
        Returns True when movie_id exists in training mappings.

        Args:
            movie_id (int): Movie identifier.

        Returns:
            bool: True if movie is known, otherwise False.

        PL:
        Sprawdza, czy identyfikator filmu istnieje w mapowaniach.

        Argumenty:
            movie_id (int): Identyfikator filmu.

        Zwraca:
            bool: True gdy film jest znany, w przeciwnym razie False.
        """
        # Sprawdzamy obecność movie_id w mapowaniu filmów.
        return movie_id in self.movie_to_index

    def _cold_start_prediction(self, user_id: int, movie_id: int) -> float:
        """Estimate rating for unknown user or movie.

        EN:
        Uses movie mean first, then user mean, then global mean fallback.

        Args:
            user_id (int): User identifier.
            movie_id (int): Movie identifier.

        Returns:
            float: Fallback rating estimate.

        PL:
        Dla cold-start używa kolejno: średniej filmu, średniej użytkownika
        i średniej globalnej.

        Argumenty:
            user_id (int): Identyfikator użytkownika.
            movie_id (int): Identyfikator filmu.

        Zwraca:
            float: Oszacowana ocena fallback.
        """
        # Najpierw próbujemy użyć średniej oceny filmu.
        if movie_id in self.movie_means:
            return float(self.movie_means[movie_id])
        # Jeśli filmu nie ma, próbujemy średniej użytkownika.
        if user_id in self.user_means:
            return float(self.user_means[user_id])
        # Ostateczny fallback to średnia globalna.
        return float(self.global_mean)

    @staticmethod
    def _round_to_half(x: float) -> float:
        """Round value to nearest 0.5 increment.

        EN:
        Converts a continuous score into half-step scale.

        Args:
            x (float): Input value.

        Returns:
            float: Rounded value.

        PL:
        Zaokrągla wartość do najbliższego kroku 0.5.

        Argumenty:
            x (float): Wartość wejściowa.

        Zwraca:
            float: Wartość po zaokrągleniu.
        """
        # Zaokrąglamy wynik do najbliższego kroku co 0.5.
        return np.round(float(x) * 2.0) / 2.0

    @staticmethod
    def _clip_rating(
        x: float,
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ) -> float:
        """Clip rating to allowed range.

        EN:
        Restricts value to interval [min_rating, max_rating].

        Args:
            x (float): Input rating.
            min_rating (float): Lower bound.
            max_rating (float): Upper bound.

        Returns:
            float: Clipped rating.

        PL:
        Ogranicza wartość do przedziału [min_rating, max_rating].

        Argumenty:
            x (float): Wejściowa wartość oceny.
            min_rating (float): Dolna granica.
            max_rating (float): Górna granica.

        Zwraca:
            float: Ocena po ograniczeniu.
        """
        # Ograniczamy wartość do przedziału [min_rating, max_rating].
        return float(np.clip(x, min_rating, max_rating))
