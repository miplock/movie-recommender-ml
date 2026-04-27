# SVD2 recommender with iterative SVD reconstruction (missing values refinement).
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from modules.data_utils import build_rating_matrix
from modules.models.base_model import BaseRecommender
from modules.preprocessing import impute_missing_values


class SVD2Recommender(BaseRecommender):
    """SVD2 recommender with iterative matrix completion using truncated SVD.

    EN:
    Performs iterative low-rank approximation where missing values are
    repeatedly updated from reconstructed matrix.

    PL:
    Iteracyjnie przybliża macierz ocen poprzez SVD, uzupełniając brakujące
    wartości na podstawie rekonstrukcji.
    """

    def __init__(
        self,
        n_components: int = 15,
        imputation_strategy: str = "zero",
        random_state: int = 0,
        n_iters: int = 10,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.imputation_strategy = imputation_strategy
        self.random_state = random_state
        self.n_iters = n_iters
        self.tol = tol

        self.algorithm_name = "SVD2"
        self.model_params = {
            "n_components": self.n_components,
            "imputation_strategy": self.imputation_strategy,
            "random_state": self.random_state,
            "n_iters": self.n_iters,
            "tol": self.tol,
        }

        self.model: TruncatedSVD | None = None
        self.W: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.Z_pred: np.ndarray | None = None

    def fit(self, ratings_df: pd.DataFrame) -> "SVD2Recommender":
        # --- identyczne jak w SVD1 ---
        self._validate_rating_columns(ratings_df)
        self._build_mappings(ratings_df)
        self._compute_basic_statistics(ratings_df)

        Z = build_rating_matrix(
            ratings_df,
            user_to_index=self.user_to_index,
            movie_to_index=self.movie_to_index,
        )

        # maska znanych wartości
        mask = ~np.isnan(Z)

        # initial fill
        Z_filled = impute_missing_values(
            Z,
            strategy=self.imputation_strategy,
            global_mean=None,
        )
        #Z_centered = Z_filled - self.global_mean #

        # --- ITERACYJNE SVD ---
        for iteration in range(self.n_iters + 1):
            Z_old = Z_filled.copy()

            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state,
            )

            self.model.fit(Z_filled) #

            singular_values = self.model.singular_values_
            transformed = self.model.transform(Z_filled) #

            safe_sigma = np.where(singular_values == 0.0, 1.0, singular_values)
            W = transformed / safe_sigma

            sigma_matrix = np.diag(singular_values)
            H = np.dot(sigma_matrix, self.model.components_)

            Z_reconstructed = np.dot(W, H)
            #Z_reconstructed += self.global_mean #

            # zachowujemy oryginalne wartości, aktualizujemy tylko braki
            Z_filled[~mask] = Z_reconstructed[~mask]

            # sprawdzamy zbieżność
            delta = (Z_filled - Z_old)[~mask]
            base = Z_old[~mask]
            diff = np.linalg.norm(delta) / (np.linalg.norm(base) + 1e-8)
            if iteration % 10 == 0:
                print(f"[SVD2] Iter {iteration}, diff={diff:.8f}")

            if diff < self.tol and iteration >= 3:
                print("[SVD2] Converged")
                break

        # zapisujemy końcowe macierze
        self.W = W
        self.H = H
        
        #Z_clipped = np.clip(Z_filled, 0.0, 5.0)
        #self.Z_pred = np.round(Z_clipped * 2) / 2
        Z_filled = np.clip(Z_filled, 0, 5)
        self.Z_pred = Z_filled  # finalna macierz po iteracjach

        self.is_fitted = True
        return self

    def _predict_known_pair(self, user_id: int, movie_id: int) -> float:
        u_idx = self.user_to_index[user_id]
        m_idx = self.movie_to_index[movie_id]
        return float(self.Z_pred[u_idx, m_idx])