# NMF recommender module for training, predicting, and model persistence.
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


class NMFRecommender:
    """NMF-based recommender with simple missing-value and cold-start handling.

    EN:
    The class builds user-item latent factors with Non-negative Matrix
    Factorization and provides prediction, persistence, and fallback logic.

    PL:
    Klasa buduje ukryte czynniki użytkownik-przedmiot metodą NMF oraz
    udostępnia predykcję, zapis modelu i logikę fallback.
    """

    def __init__(
        self,
        n_components: int = 15,
        imputation_strategy: str = "movie_mean",
        random_state: int = 0,
        max_iter: int = 300,
    ) -> None:
        """Initialize model hyperparameters and internal storage.

        EN:
        Configure NMF settings and create empty attributes required during
        fitting and prediction.

        Args:
            n_components (int): Number of latent factors used by NMF.
            imputation_strategy (str): Strategy for filling missing ratings.
            random_state (int): Random seed passed to sklearn NMF.
            max_iter (int): Maximum number of NMF optimization iterations.

        Returns:
            None: Constructor initializes object state in place.

        PL:
        Konfiguruje parametry NMF i tworzy puste pola potrzebne podczas
        trenowania oraz predykcji.

        Argumenty:
            n_components (int): Liczba ukrytych czynników modelu NMF.
            imputation_strategy (str): Strategia uzupełniania braków ocen.
            random_state (int): Ziarno losowe przekazywane do sklearn NMF.
            max_iter (int): Maksymalna liczba iteracji optymalizacji NMF.

        Zwraca:
            None: Konstruktor inicjalizuje stan obiektu.
        """
        # Zapisujemy liczbę ukrytych wymiarów przestrzeni latentnej.
        self.n_components = n_components
        # Zapamiętujemy wybraną strategię uzupełniania brakujących ocen.
        self.imputation_strategy = imputation_strategy
        # Przechowujemy seed zapewniający powtarzalność wyników.
        self.random_state = random_state
        # Zapisujemy limit iteracji optymalizacji NMF.
        self.max_iter = max_iter

        # Tu będzie obiekt modelu sklearn po wywołaniu fit.
        self.model = None
        # W to miejsce trafi macierz cech użytkowników.
        self.W = None
        # W to miejsce trafi macierz cech filmów.
        self.H = None

        # Mapowanie userId -> indeks w macierzy.
        self.user_to_index = None
        # Mapowanie movieId -> indeks w macierzy.
        self.movie_to_index = None
        # Mapowanie odwrotne indeks -> userId.
        self.index_to_user = None
        # Mapowanie odwrotne indeks -> movieId.
        self.index_to_movie = None

        # Globalna średnia ocen używana jako ostateczny fallback.
        self.global_mean = None
        # Średnia ocena na film do obsługi cold-start dla nowego usera.
        self.movie_means = None
        # Średnia ocena na użytkownika do obsługi cold-start.
        # Używamy jej, gdy film jest nowy i brak go w danych treningowych.
        self.user_means = None

    def fit(self, ratings_df: pd.DataFrame) -> "NMFRecommender":
        """Fit the recommender from a ratings DataFrame.

        EN:
        Validate input columns, build mapping dictionaries, impute missing
        values in user-item matrix, and learn NMF latent factors.

        Args:
            ratings_df (pd.DataFrame): Table with userId, movieId, rating.

        Returns:
            NMFRecommender: Fitted instance for chaining.

        Raises:
            ValueError: If required columns are missing.

        PL:
        Waliduje kolumny wejściowe, buduje mapowania indeksów, uzupełnia
        braki w macierzy użytkownik-przedmiot i uczy czynniki NMF.

        Argumenty:
            ratings_df (pd.DataFrame): Tabela z kolumnami userId, movieId,
                rating.

        Zwraca:
            NMFRecommender: Wytrenowaną instancję do łańcuchowania.

        Wyjątki:
            ValueError: Gdy brakuje wymaganych kolumn.
        """
        # Definiujemy minimalny zestaw kolumn wymagany do uczenia modelu.
        required_columns = {"userId", "movieId", "rating"}
        # Sprawdzamy, których kolumn brakuje w dostarczonym DataFrame.
        missing = required_columns - set(ratings_df.columns)
        # Przerywamy działanie, jeśli wejście nie ma wymaganej struktury.
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Liczymy średnią globalną ocen jako wartość fallback.
        self.global_mean = float(ratings_df["rating"].mean())

        # Pobieramy unikalne identyfikatory i sortujemy je dla stabilności.
        user_ids = sorted(ratings_df["userId"].unique())
        movie_ids = sorted(ratings_df["movieId"].unique())

        # Budujemy mapowanie identyfikatorów użytkowników na indeksy.
        self.user_to_index = {
            user_id: idx for idx, user_id in enumerate(user_ids)
        }
        # Budujemy mapowanie identyfikatorów filmów na indeksy.
        self.movie_to_index = {
            movie_id: idx for idx, movie_id in enumerate(movie_ids)
        }
        # Tworzymy mapowanie odwrotne indeksów użytkowników.
        self.index_to_user = {
            idx: user_id for user_id, idx in self.user_to_index.items()
        }
        # Tworzymy mapowanie odwrotne indeksów filmów.
        self.index_to_movie = {
            idx: movie_id for movie_id, idx in self.movie_to_index.items()
        }

        # Wyznaczamy rozmiar macierzy użytkownik-przedmiot.
        n_users = len(user_ids)
        n_movies = len(movie_ids)

        # Tworzymy pustą macierz ocen z NaN tam, gdzie brak obserwacji.
        Z = np.full((n_users, n_movies), np.nan, dtype=float)

        # Wypełniamy macierz znanymi ocenami z wejściowego DataFrame.
        for row in ratings_df.itertuples(index=False):
            # Odczytujemy indeks użytkownika.
            u_idx = self.user_to_index[row.userId]
            # Odczytujemy indeks filmu.
            m_idx = self.movie_to_index[row.movieId]
            # Wpisujemy ocenę do odpowiedniej komórki macierzy.
            Z[u_idx, m_idx] = float(row.rating)

        # Uzupełniamy braki zgodnie z wybraną strategią imputacji.
        Z_filled = self._impute_missing_values(Z)

        # Inicjalizujemy model NMF z zapisanymi hiperparametrami.
        self.model = NMF(
            n_components=self.n_components,
            init="random",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        # Uczymy macierz cech użytkowników na podstawie danych wejściowych.
        self.W = self.model.fit_transform(Z_filled)
        # Pobieramy wyuczoną macierz cech filmów.
        self.H = self.model.components_

        # Liczymy średnie ocen per film do fallbacku cold-start.
        self.movie_means = (
            ratings_df.groupby("movieId")["rating"].mean().to_dict()
        )
        # Liczymy średnie ocen per użytkownik do fallbacku cold-start.
        self.user_means = (
            ratings_df.groupby("userId")["rating"].mean().to_dict()
        )

        # Zwracamy self, aby umożliwić łańcuchowe wywołania metod.
        return self

    def predict(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for user-movie pairs.

        EN:
        Generate predictions from learned latent factors and use cold-start
        fallback for unknown users or movies.

        Args:
            pairs_df (pd.DataFrame): Table with userId and movieId columns.

        Returns:
            np.ndarray: Predicted ratings clipped to valid range and rounded
                to halves.

        Raises:
            ValueError: If required columns are missing.
            ValueError: If model has not been fitted.

        PL:
        Generuje predykcje z wyuczonych czynników ukrytych i stosuje
        fallback cold-start dla nieznanych użytkowników lub filmów.

        Argumenty:
            pairs_df (pd.DataFrame): Tabela z kolumnami userId i movieId.

        Zwraca:
            np.ndarray: Przewidziane oceny ograniczone do zakresu i
                zaokrąglone do połówek.

        Wyjątki:
            ValueError: Gdy brakuje wymaganych kolumn.
            ValueError: Gdy model nie został wytrenowany.
        """
        # Definiujemy wymagane kolumny dla zestawu par do predykcji.
        required_columns = {"userId", "movieId"}
        # Weryfikujemy kompletność schematu wejścia.
        missing = required_columns - set(pairs_df.columns)
        # Zgłaszamy błąd, jeśli nie da się wyliczyć predykcji.
        # Powód: w wejściu brakuje wymaganych kolumn.
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sprawdzamy, czy model został wcześniej wytrenowany.
        if self.W is None or self.H is None:
            raise ValueError("Model is not fitted yet.")

        # Odtwarzamy pełną macierz predykcji przez iloczyn W i H.
        Z_pred = self.W @ self.H
        # Przygotowujemy listę na wyniki dla kolejnych par.
        predictions = []

        # Iterujemy po parach user-film przekazanych do predykcji.
        for row in pairs_df.itertuples(index=False):
            # Pobieramy identyfikator użytkownika z bieżącego wiersza.
            user_id = row.userId
            # Pobieramy identyfikator filmu z bieżącego wiersza.
            movie_id = row.movieId

            # Jeśli znamy użytkownika i film, używamy predykcji z macierzy.
            if (
                user_id in self.user_to_index
                and movie_id in self.movie_to_index
            ):
                # Odczytujemy indeks użytkownika.
                u_idx = self.user_to_index[user_id]
                # Odczytujemy indeks filmu.
                m_idx = self.movie_to_index[movie_id]
                # Pobieramy gotową predykcję dla tej pary indeksów.
                pred = float(Z_pred[u_idx, m_idx])
            else:
                # Dla nowych obiektów stosujemy logikę cold-start.
                pred = self._cold_start_prediction(user_id, movie_id)

            # Ograniczamy wynik do dozwolonego zakresu ocen.
            pred = self._clip_rating(pred)
            # Zaokrąglamy wynik do połówkowych kroków skali ocen.
            pred = self._round_to_half(pred)
            # Zapisujemy gotowy wynik do listy.
            predictions.append(pred)

        # Zwracamy predykcje jako tablicę NumPy.
        return np.array(predictions)

    def predict_dataframe(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a copied DataFrame with rating column.

        EN:
        Keep the original pair columns and append predicted ratings.

        Args:
            pairs_df (pd.DataFrame): Input pairs with userId and movieId.

        Returns:
            pd.DataFrame: Copy of input with an added rating column.

        PL:
        Zachowuje oryginalne kolumny par i dodaje przewidziane oceny.

        Argumenty:
            pairs_df (pd.DataFrame): Dane wejściowe z userId i movieId.

        Zwraca:
            pd.DataFrame: Kopię wejścia z dodatkową kolumną rating.
        """
        # Tworzymy kopię, aby nie modyfikować obiektu wejściowego.
        result = pairs_df.copy()
        # Wyliczamy predykcje i dopisujemy je jako kolumnę "rating".
        result["rating"] = self.predict(pairs_df)
        # Zwracamy rozszerzony DataFrame.
        return result

    def save(self, model_path: str) -> None:
        """Serialize the recommender instance to disk using pickle.

        EN:
        Persist the full fitted object state to a binary file.

        Args:
            model_path (str): Target file path for serialized model.

        Returns:
            None: Model is written to disk as a side effect.

        PL:
        Zapisuje pełny stan instancji modelu na dysk przy użyciu pickle.

        Argumenty:
            model_path (str): Docelowa ścieżka pliku z modelem.

        Zwraca:
            None: Model jest zapisywany na dysku jako efekt uboczny.
        """
        # Otwieramy plik docelowy w trybie binarnym zapisu.
        with open(model_path, "wb") as f:
            # Serializujemy cały obiekt modelu do pliku.
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str) -> "NMFRecommender":
        """Load a serialized recommender from disk.

        EN:
        Read a pickled model file and return restored instance.

        Args:
            model_path (str): Path to serialized model file.

        Returns:
            NMFRecommender: Restored recommender object.

        PL:
        Wczytuje zapisany model z pliku pickle i zwraca odtworzoną instancję.

        Argumenty:
            model_path (str): Ścieżka do pliku z zapisanym modelem.

        Zwraca:
            NMFRecommender: Odtworzony obiekt rekomendera.
        """
        # Otwieramy plik modelu w trybie binarnym odczytu.
        with open(model_path, "rb") as f:
            # Deserializujemy obiekt i zwracamy go do dalszego użycia.
            return pickle.load(f)

    def _impute_missing_values(self, Z: np.ndarray) -> np.ndarray:
        """Fill missing ratings in matrix according to selected strategy.

        EN:
        Replace NaN cells with zero, movie mean, or user mean depending on
        configuration.

        Args:
            Z (np.ndarray): User-item matrix containing NaN for missing data.

        Returns:
            np.ndarray: Matrix without NaN values.

        Raises:
            ValueError: If imputation strategy is unsupported.

        PL:
        Uzupełnia komórki NaN zerem, średnią filmu
        lub średnią użytkownika zgodnie z konfiguracją.

        Argumenty:
            Z (np.ndarray): Macierz użytkownik-przedmiot z NaN dla braków.

        Zwraca:
            np.ndarray: Macierz bez wartości NaN.

        Wyjątki:
            ValueError: Gdy strategia imputacji jest nieobsługiwana.
        """
        # Tworzymy kopię, aby nie zmieniać oryginalnej macierzy wejściowej.
        Z_filled = Z.copy()

        # Strategia "zero" zastępuje wszystkie braki wartością 0.0.
        if self.imputation_strategy == "zero":
            return np.nan_to_num(Z_filled, nan=0.0)

        # Strategia "movie_mean" wypełnia NaN średnią kolumny (filmu).
        if self.imputation_strategy == "movie_mean":
            # Liczymy średnie kolumnowe, ignorując wartości NaN.
            col_means = np.nanmean(Z_filled, axis=0)
            # Dla pustych kolumn używamy średniej globalnej.
            col_means = np.where(
                np.isnan(col_means), self.global_mean, col_means
            )
            # Pobieramy indeksy komórek zawierających NaN.
            inds = np.where(np.isnan(Z_filled))
            # Podmieniamy NaN odpowiednimi średnimi dla filmów.
            Z_filled[inds] = col_means[inds[1]]
            # Zwracamy macierz po imputacji.
            return Z_filled

        # Strategia "user_mean" wypełnia NaN średnią wiersza (użytkownika).
        if self.imputation_strategy == "user_mean":
            # Liczymy średnie wierszowe, ignorując wartości NaN.
            row_means = np.nanmean(Z_filled, axis=1)
            # Dla pustych wierszy używamy średniej globalnej.
            row_means = np.where(
                np.isnan(row_means), self.global_mean, row_means
            )
            # Pobieramy indeksy komórek zawierających NaN.
            inds = np.where(np.isnan(Z_filled))
            # Podmieniamy NaN odpowiednimi średnimi dla użytkowników.
            Z_filled[inds] = row_means[inds[0]]
            # Zwracamy macierz po imputacji.
            return Z_filled

        # Zgłaszamy błąd dla nieznanej nazwy strategii imputacji.
        raise ValueError(
            f"Unknown imputation strategy: {self.imputation_strategy}"
        )

    def _cold_start_prediction(self, user_id: int, movie_id: int) -> float:
        """Estimate rating for unseen users or movies.

        EN:
        Prefer movie mean, then user mean, and fallback to global mean.

        Args:
            user_id (int): User identifier.
            movie_id (int): Movie identifier.

        Returns:
            float: Fallback rating estimate.

        PL:
        Preferuje średnią filmu, potem średnią użytkownika,
        a na końcu średnią globalną.

        Argumenty:
            user_id (int): Identyfikator użytkownika.
            movie_id (int): Identyfikator filmu.

        Zwraca:
            float: Ocena oszacowana przez fallback.
        """
        # Najpierw próbujemy użyć średniej oceny dla danego filmu.
        if movie_id in self.movie_means:
            return float(self.movie_means[movie_id])
        # Jeśli film jest nowy, używamy średniej oceny użytkownika.
        if user_id in self.user_means:
            return float(self.user_means[user_id])
        # Ostateczny fallback to globalna średnia całego zbioru.
        return float(self.global_mean)

    @staticmethod
    def _round_to_half(x: float) -> float:
        """Round value to nearest 0.5 step.

        EN:
        Convert continuous score into half-star style rating.

        Args:
            x (float): Value to round.

        Returns:
            float: Rounded value in 0.5 increments.

        PL:
        Zaokrągla wartość do najbliższego kroku 0.5.

        Argumenty:
            x (float): Wartość do zaokrąglenia.

        Zwraca:
            float: Wartość zaokrąglona do przyrostu 0.5.
        """
        # Mnożymy przez 2, zaokrąglamy do całości i dzielimy z powrotem.
        return np.round(x * 2.0) / 2.0

    @staticmethod
    def _clip_rating(
        x: float,
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ) -> float:
        """Clip rating to allowed interval.

        EN:
        Restrict prediction to the configured minimum and maximum value.

        Args:
            x (float): Rating value to clip.
            min_rating (float): Lower allowed rating bound.
            max_rating (float): Upper allowed rating bound.

        Returns:
            float: Clipped rating value.

        PL:
        Ogranicza predykcję do dozwolonego przedziału wartości.

        Argumenty:
            x (float): Wartość oceny do ograniczenia.
            min_rating (float): Dolna granica dozwolonej oceny.
            max_rating (float): Górna granica dozwolonej oceny.

        Zwraca:
            float: Ocena po ograniczeniu do zakresu.
        """
        # Ograniczamy wartość do przedziału [min_rating, max_rating].
        return np.clip(x, min_rating, max_rating)
