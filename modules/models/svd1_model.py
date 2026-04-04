# SVD1 recommender built on top of shared base recommender utilities.
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from modules.data_utils import build_rating_matrix
from modules.models.base_model import BaseRecommender
from modules.preprocessing import impute_missing_values


class SVD1Recommender(BaseRecommender):
    """SVD1 recommender implementation based on BaseRecommender utilities.

    EN:
    Learns user and movie latent factors from truncated SVD and predicts
    ratings from reconstructed matrix.

    PL:
    Uczy ukryte czynniki użytkowników i filmów z obciętego SVD oraz
    przewiduje oceny na podstawie macierzy rekonstrukcji.
    """

    def __init__(
        self,
        n_components: int = 15,
        imputation_strategy: str = "movie_mean",
        random_state: int = 0,
    ) -> None:
        """Initialize SVD1 hyperparameters and runtime state.

        EN:
        Configure training parameters and allocate placeholders for learned
        matrices.

        Args:
            n_components (int): Number of latent components for SVD.
            imputation_strategy (str): Missing-value strategy for matrix.
            random_state (int): Random seed used by TruncatedSVD.

        Returns:
            None: Constructor initializes object attributes.

        PL:
        Konfiguruje parametry treningu i tworzy pola na wyuczone macierze.

        Argumenty:
            n_components (int): Liczba ukrytych składowych modelu SVD.
            imputation_strategy (str): Strategia uzupełniania braków.
            random_state (int): Ziarno losowe przekazane do TruncatedSVD.

        Zwraca:
            None: Konstruktor inicjalizuje atrybuty obiektu.
        """
        # Inicjalizujemy pola wspólne z klasy bazowej.
        super().__init__()
        # Zapisujemy docelową liczbę ukrytych komponentów SVD.
        self.n_components = n_components
        # Zapamiętujemy strategię uzupełniania brakujących wartości.
        self.imputation_strategy = imputation_strategy
        # Zapisujemy ziarno losowe dla powtarzalności.
        self.random_state = random_state
        # Ujednolicone metadane algorytmu dla logów eksperymentów.
        self.algorithm_name = "SVD1"
        self.model_params = {
            "n_components": self.n_components,
            "imputation_strategy": self.imputation_strategy,
            "random_state": self.random_state,
        }

        # Obiekt modelu sklearn, tworzony podczas fit().
        self.model: TruncatedSVD | None = None
        # Macierz cech użytkowników U_r z dekompozycji SVD.
        self.W: np.ndarray | None = None
        # Macierz cech filmów Lambda_r * V_r^T.
        self.H: np.ndarray | None = None
        # Zrekonstruowana macierz predykcji dla znanych par.
        self.Z_pred: np.ndarray | None = None

    def fit(self, ratings_df: pd.DataFrame) -> "SVD1Recommender":
        """Fit SVD1 model on training ratings.

        EN:
        Validate input data, build user-item matrix, impute missing values,
        train truncated SVD, and cache reconstructed prediction matrix.

        Args:
            ratings_df (pd.DataFrame): Table with userId, movieId, rating.

        Returns:
            SVD1Recommender: Fitted model instance.

        Raises:
            ValueError: If required columns are missing.

        PL:
        Waliduje dane, buduje macierz użytkownik-film, uzupełnia braki,
        trenuje obcięte SVD i zapisuje macierz rekonstrukcji predykcji.

        Argumenty:
            ratings_df (pd.DataFrame): Tabela z userId, movieId, rating.

        Zwraca:
            SVD1Recommender: Wytrenowaną instancję modelu.

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

        # Tworzymy obiekt TruncatedSVD z ustalonymi hiperparametrami.
        self.model = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )

        # Dopasowujemy dekompozycję SVD do macierzy wejściowej.
        self.model.fit(Z_filled)
        # Pobieramy wartości osobliwe Sigma_r.
        singular_values = self.model.singular_values_

        # Wyliczamy W = U_r z zależności transform(Z) = U_r * Sigma_r.
        transformed = self.model.transform(Z_filled)
        # Zabezpieczenie przed dzieleniem przez zero przy zerowych sigma.
        safe_sigma = np.where(singular_values == 0.0, 1.0, singular_values)
        self.W = transformed / safe_sigma

        # Składamy H = Lambda_r * V_r^T.
        sigma_matrix = np.diag(singular_values)
        self.H = np.dot(sigma_matrix, self.model.components_)
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
