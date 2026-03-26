# NMF recommender built on top of shared base recommender utilities.
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from modules.data_utils import build_rating_matrix
from modules.models.base_model import BaseRecommender
from modules.preprocessing import impute_missing_values


class NMFRecommender(BaseRecommender):
    """NMF recommender implementation based on BaseRecommender utilities.

    EN:
    Learns user and movie latent factors and predicts ratings from
    reconstructed matrix.

    PL:
    Uczy ukryte czynniki użytkowników i filmów oraz przewiduje oceny na
    podstawie macierzy rekonstrukcji.
    """

    def __init__(
        self,
        n_components: int = 15,
        imputation_strategy: str = "movie_mean",
        random_state: int = 0,
        max_iter: int = 300,
    ) -> None:
        """Initialize NMF hyperparameters and runtime state.

        EN:
        Configure training parameters and allocate placeholders for learned
        matrices.

        Args:
            n_components (int): Number of latent components for NMF.
            imputation_strategy (str): Missing-value strategy for matrix.
            random_state (int): Random seed used by sklearn NMF.
            max_iter (int): Maximum number of NMF optimization iterations.

        Returns:
            None: Constructor initializes object attributes.

        PL:
        Konfiguruje parametry treningu i tworzy pola na wyuczone macierze.

        Argumenty:
            n_components (int): Liczba ukrytych składowych modelu NMF.
            imputation_strategy (str): Strategia uzupełniania braków.
            random_state (int): Ziarno losowe przekazane do sklearn NMF.
            max_iter (int): Maksymalna liczba iteracji optymalizacji NMF.

        Zwraca:
            None: Konstruktor inicjalizuje atrybuty obiektu.
        """
        # Inicjalizujemy pola wspólne z klasy bazowej.
        super().__init__()
        # Zapisujemy docelową liczbę ukrytych komponentów NMF.
        self.n_components = n_components
        # Zapamiętujemy strategię uzupełniania brakujących wartości.
        self.imputation_strategy = imputation_strategy
        # Zapisujemy ziarno losowe dla powtarzalności.
        self.random_state = random_state
        # Zapisujemy maksymalną liczbę iteracji optymalizacji.
        self.max_iter = max_iter

        # Obiekt modelu sklearn, tworzony podczas fit().
        self.model: NMF | None = None
        # Macierz cech użytkowników po factorization.
        self.W: np.ndarray | None = None
        # Macierz cech filmów po factorization.
        self.H: np.ndarray | None = None
        # Zrekonstruowana macierz predykcji dla znanych par.
        self.Z_pred: np.ndarray | None = None

    def fit(self, ratings_df: pd.DataFrame) -> "NMFRecommender":
        """Fit NMF model on training ratings.

        EN:
        Validate input data, build user-item matrix, impute missing values,
        train NMF, and cache reconstructed prediction matrix.

        Args:
            ratings_df (pd.DataFrame): Table with userId, movieId, rating.

        Returns:
            NMFRecommender: Fitted model instance.

        Raises:
            ValueError: If required columns are missing.

        PL:
        Waliduje dane, buduje macierz użytkownik-film, uzupełnia braki,
        trenuje NMF i zapisuje macierz rekonstrukcji predykcji.

        Argumenty:
            ratings_df (pd.DataFrame): Tabela z userId, movieId, rating.

        Zwraca:
            NMFRecommender: Wytrenowaną instancję modelu.

        Wyjątki:
            ValueError: Gdy brakuje wymaganych kolumn.
        """
        # Sprawdzamy, czy dane wejściowe mają wymagane kolumny.
        self._validate_rating_columns(ratings_df)
        # Budujemy mapowania ID <-> indeksy macierzy.
        self._build_mappings(ratings_df)
        # Liczymy statystyki potrzebne do fallbacku cold-start.
        self._compute_basic_statistics(ratings_df)

        # Budujemy macierz użytkownik-film z wartościami NaN dla braków.
        Z = build_rating_matrix(
            ratings_df,
            user_to_index=self.user_to_index,
            movie_to_index=self.movie_to_index,
        )

        # Uzupełniamy brakujące oceny zgodnie z wybraną strategią.
        Z_filled = impute_missing_values(
            Z,
            strategy=self.imputation_strategy,
            global_mean=self.global_mean,
        )

        # Tworzymy obiekt NMF z ustalonymi hiperparametrami.
        self.model = NMF(
            n_components=self.n_components,
            init="random",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        # Uczymy macierz czynników użytkowników.
        self.W = self.model.fit_transform(Z_filled)
        # Odczytujemy macierz czynników filmów.
        self.H = self.model.components_
        # Rekonstruujemy pełną macierz predykcji dla szybkiego odczytu.
        self.Z_pred = np.dot(self.W, self.H)

        # Oznaczamy model jako wytrenowany.
        self.is_fitted = True
        # Zwracamy self, żeby wspierać łańcuchowanie.
        return self

    def _predict_known_pair(self, user_id: int, movie_id: int) -> float:
        """Predict rating for known identifiers using cached matrix.

        EN:
        Convert user and movie IDs to matrix indices and read cached value.

        Args:
            user_id (int): User identifier known to model mapping.
            movie_id (int): Movie identifier known to model mapping.

        Returns:
            float: Predicted rating for the given pair.

        PL:
        Zamienia identyfikatory na indeksy i odczytuje wynik z macierzy
        predykcji.

        Argumenty:
            user_id (int): Identyfikator użytkownika znany w mapowaniu.
            movie_id (int): Identyfikator filmu znany w mapowaniu.

        Zwraca:
            float: Przewidziana ocena dla podanej pary.
        """
        # Pobieramy indeks użytkownika z mapowania.
        u_idx = self.user_to_index[user_id]
        # Pobieramy indeks filmu z mapowania.
        m_idx = self.movie_to_index[movie_id]
        # Zwracamy predykcję z gotowej macierzy rekonstrukcji.
        return float(self.Z_pred[u_idx, m_idx])
