# SGD recommender module using PyTorch matrix factorization with biases.
"""SGD recommender implementation with PyTorch matrix factorization.

EN:
Provides an SGD-style recommender based on latent factors, user/movie
biases, and mini-batch optimization.

PL:
Zawiera implementację rekomendera SGD opartego na latent factors,
biasach user/movie i optymalizacji mini-batch.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer

from modules.models.base_model import BaseRecommender


class MatrixFactorizationModule(nn.Module):
    """Matrix factorization network used by SGDRecommender.

    EN:
    Learns user and movie embeddings with additive user/movie bias terms.
    The forward pass returns predicted rating values for index batches.

    PL:
    Uczy embeddingi użytkowników i filmów oraz dodaje bias użytkownika
    i filmu. Przebieg forward zwraca przewidywane oceny dla batcha.
    """

    def __init__(self, n_users: int, n_movies: int, rank: int) -> None:
        """Initialize embeddings and bias vectors.

        EN:
        Creates trainable user/movie factor matrices and one-dimensional
        bias embeddings for users and movies.

        Args:
            n_users (int): Number of users in mapping.
            n_movies (int): Number of movies in mapping.
            rank (int): Number of latent factors.

        Returns:
            None: Initializes torch layers in place.

        PL:
        Tworzy trenowalne macierze czynników oraz jednowymiarowe biasy
        użytkowników i filmów.

        Argumenty:
            n_users (int): Liczba użytkowników w mapowaniu.
            n_movies (int): Liczba filmów w mapowaniu.
            rank (int): Liczba ukrytych cech.

        Zwraca:
            None: Inicjalizuje warstwy torch w obiekcie.
        """
        # Inicjalizujemy klasę bazową torch.
        super().__init__()

        # Tworzymy embedding dla latent factors użytkowników.
        self.user_factors = nn.Embedding(n_users, rank)
        # Tworzymy embedding dla latent factors filmów.
        self.movie_factors = nn.Embedding(n_movies, rank)
        # Tworzymy embedding dla biasu użytkownika.
        self.user_bias = nn.Embedding(n_users, 1)
        # Tworzymy embedding dla biasu filmu.
        self.movie_bias = nn.Embedding(n_movies, 1)

        # Inicjalizujemy czynniki rozkładem normalnym.
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.05)
        # Inicjalizujemy czynniki rozkładem normalnym.
        nn.init.normal_(self.movie_factors.weight, mean=0.0, std=0.05)
        # Bias użytkownika startuje od zera.
        nn.init.zeros_(self.user_bias.weight)
        # Bias filmu startuje od zera.
        nn.init.zeros_(self.movie_bias.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass for rating prediction.

        EN:
        Fetches embeddings and biases for batch indices, computes latent
        dot product, and adds global mean and bias components.

        Args:
            user_idx (torch.Tensor): Tensor with user indices.
            movie_idx (torch.Tensor): Tensor with movie indices.
            global_mean (torch.Tensor): Scalar tensor with global mean.

        Returns:
            torch.Tensor: Predicted ratings for the given batch.

        PL:
        Pobiera embeddingi i biasy dla indeksów batcha, liczy iloczyn
        skalarny czynników i dodaje średnią globalną oraz biasy.

        Argumenty:
            user_idx (torch.Tensor): Tensor z indeksami użytkowników.
            movie_idx (torch.Tensor): Tensor z indeksami filmów.
            global_mean (torch.Tensor): Skalar tensor ze średnią globalną.

        Zwraca:
            torch.Tensor: Przewidywane oceny dla podanego batcha.
        """
        # Pobieramy latent vectors użytkowników.
        user_latent = self.user_factors(user_idx)
        # Pobieramy latent vectors filmów.
        movie_latent = self.movie_factors(movie_idx)
        # Pobieramy bias użytkownika i redukujemy wymiar.
        user_bias = self.user_bias(user_idx).squeeze(-1)
        # Pobieramy bias filmu i redukujemy wymiar.
        movie_bias = self.movie_bias(movie_idx).squeeze(-1)

        # Liczymy iloczyn skalarny latent vectors.
        dot_product = torch.sum(user_latent * movie_latent, dim=1)
        # Budujemy końcową predykcję ratingu.
        return global_mean + user_bias + movie_bias + dot_product


class SGDRecommender(BaseRecommender):
    """PyTorch recommender trained with SGD or Adam optimizer.

    EN:
    Fits matrix factorization parameters on observed ratings and predicts
    scores for known and cold-start user-movie pairs.

    PL:
    Trenuje parametry faktoryzacji macierzy na obserwowanych ocenach
    i przewiduje oceny dla znanych oraz nowych par user-film.
    """

    def __init__(
        self,
        rank: int = 32,
        lr: float = 0.02,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 4096,
        optimizer_name: str = "adam",
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        """Initialize hyperparameters and runtime state.

        EN:
        Stores optimizer and training settings and prepares placeholders
        for fitted torch model and tensors.

        Args:
            rank (int): Number of latent factors.
            lr (float): Learning rate for optimizer.
            weight_decay (float): L2 regularization value.
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size for training.
            optimizer_name (str): Optimizer name: adam or sgd.
            device (str): Torch device identifier.
            seed (int): Random seed for reproducibility.
            verbose (bool): Whether to print training progress.

        Returns:
            None: Initializes attributes used during training/prediction.

        PL:
        Zapisuje ustawienia optymalizacji i treningu oraz przygotowuje
        pola na model torch i potrzebne tensory.

        Argumenty:
            rank (int): Liczba ukrytych cech.
            lr (float): Learning rate optymalizatora.
            weight_decay (float): Wartość regularyzacji L2.
            epochs (int): Liczba epok treningowych.
            batch_size (int): Rozmiar mini-batcha w treningu.
            optimizer_name (str): Nazwa optymalizatora: adam lub sgd.
            device (str): Identyfikator urządzenia torch.
            seed (int): Ziarno losowe dla powtarzalności.
            verbose (bool): Czy wypisywać postęp treningu.

        Zwraca:
            None: Inicjalizuje atrybuty używane w treningu i predykcji.
        """
        # Inicjalizujemy pola wspólne klasy bazowej.
        super().__init__()

        # Zapisujemy liczbę latent factors.
        self.rank = rank
        # Zapisujemy krok uczenia.
        self.lr = lr
        # Zapisujemy regularyzację L2.
        self.weight_decay = weight_decay
        # Zapisujemy liczbę epok.
        self.epochs = epochs
        # Zapisujemy rozmiar mini-batcha.
        self.batch_size = batch_size
        # Normalizujemy nazwę optymalizatora do małych liter.
        self.optimizer_name = optimizer_name.lower()
        # Zapisujemy wskazane urządzenie.
        self.device = device
        # Zapisujemy ziarno losowe.
        self.seed = seed
        # Zapisujemy flagę logowania postępu.
        self.verbose = verbose

        # Ujednolicamy nazwę algorytmu dla metadanych.
        self.algorithm_name = "SGD"
        # Zapisujemy komplet hiperparametrów modelu.
        self.model_params = {
            "rank": rank,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_name": self.optimizer_name,
            "device": device,
            "seed": seed,
        }

        # Pole na obiekt modelu torch po fit.
        self.torch_model: MatrixFactorizationModule | None = None
        # Pole na obiekt urządzenia torch.
        self.device_obj = torch.device(device)
        # Pole na tensor ze średnią globalną.
        self._global_mean_tensor: torch.Tensor | None = None

    def fit(self, ratings_df: pd.DataFrame) -> "SGDRecommender":
        """Train model using observed ratings.

        EN:
        Validates training input, builds mappings and statistics, trains
        matrix factorization in mini-batches, and marks model as fitted.

        Args:
            ratings_df (pd.DataFrame): DataFrame with rating triples.

        Returns:
            SGDRecommender: Trained model instance.

        PL:
        Waliduje dane treningowe, buduje mapowania i statystyki, trenuje
        faktoryzację w mini-batchach i oznacza model jako wytrenowany.

        Argumenty:
            ratings_df (pd.DataFrame): DataFrame z trójkami ocen.

        Zwraca:
            SGDRecommender: Wytrenowaną instancję modelu.
        """
        # Sprawdzamy, czy wejście ma kolumny userId/movieId/rating.
        self._validate_rating_columns(ratings_df)
        # Ograniczamy wejście do wymaganych kolumn.
        train_df = ratings_df[["userId", "movieId", "rating"]].copy()

        # Ustawiamy seed dla NumPy.
        np.random.seed(self.seed)
        # Ustawiamy seed dla PyTorch.
        torch.manual_seed(self.seed)

        # Budujemy mapowania ID na indeksy.
        self._build_mappings(train_df)
        # Liczymy statystyki potrzebne m.in. dla cold-start.
        self._compute_basic_statistics(train_df)

        # Tworzymy obiekt modelu torch na docelowym urządzeniu.
        self._initialize_torch_model()
        # Tworzymy tensor ze średnią globalną.
        self._initialize_global_mean_tensor()

        # Kodujemy dane treningowe do tensorów indeksów i ocen.
        user_idx, movie_idx, ratings = self._prepare_training_tensors(train_df)
        # Tworzymy optymalizator na parametrach modelu.
        optimizer = self._create_optimizer(self.torch_model.parameters())

        # Ustalamy liczbę rekordów treningowych.
        n_samples = len(train_df)
        # Budujemy tablicę indeksów używaną przy tasowaniu.
        shuffled_indices = np.arange(n_samples)

        # Przełączamy model w tryb treningu.
        self.torch_model.train()

        # Iterujemy po kolejnych epokach uczenia.
        for epoch_idx in range(self.epochs):
            # Tasujemy kolejność przykładów dla bieżącej epoki.
            np.random.shuffle(shuffled_indices)
            # Trenujemy jedną epokę i dostajemy RMSE treningowe.
            train_rmse = self._run_epoch(
                user_idx=user_idx,
                movie_idx=movie_idx,
                ratings=ratings,
                sample_order=shuffled_indices,
                optimizer=optimizer,
            )
            # Opcjonalnie logujemy postęp treningu.
            self._print_epoch_log(epoch_idx=epoch_idx, train_rmse=train_rmse)

        # Oznaczamy model jako gotowy do predykcji.
        self.is_fitted = True
        # Zwracamy instancję dla chainingu.
        return self

    def _initialize_torch_model(self) -> None:
        """Create and place torch model on selected device.

        EN:
        Uses prepared user/movie mappings to create factorization module.

        Returns:
            None: Updates `self.torch_model`.

        PL:
        Tworzy moduł faktoryzacji na podstawie mapowań użytkowników
        i filmów oraz przenosi go na wybrane urządzenie.

        Zwraca:
            None: Aktualizuje `self.torch_model`.
        """
        # Ustalamy liczbę użytkowników w mapowaniu.
        n_users = len(self.user_to_index)
        # Ustalamy liczbę filmów w mapowaniu.
        n_movies = len(self.movie_to_index)

        # Tworzymy model faktoryzacji z ustalonym rank.
        model = MatrixFactorizationModule(
            n_users=n_users,
            n_movies=n_movies,
            rank=self.rank,
        )
        # Przenosimy model na docelowe urządzenie.
        self.torch_model = model.to(self.device_obj)

    def _initialize_global_mean_tensor(self) -> None:
        """Create scalar tensor with global rating mean.

        EN:
        Converts computed global mean to float32 torch tensor on selected
        device.

        Returns:
            None: Updates `_global_mean_tensor`.

        PL:
        Zamienia policzoną średnią globalną na tensor float32 na wybranym
        urządzeniu.

        Zwraca:
            None: Aktualizuje `_global_mean_tensor`.
        """
        # Tworzymy tensor z globalną średnią ocen.
        self._global_mean_tensor = torch.tensor(
            self.global_mean,
            dtype=torch.float32,
            device=self.device_obj,
        )

    def _prepare_training_tensors(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert training DataFrame into torch tensors.

        EN:
        Maps user/movie IDs to indices and creates tensors for users,
        movies, and ratings on the configured device.

        Args:
            train_df (pd.DataFrame): Training ratings DataFrame.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            User indices, movie indices, and rating values.

        PL:
        Mapuje identyfikatory user/movie na indeksy i tworzy tensory
        użytkowników, filmów oraz ocen na wybranym urządzeniu.

        Argumenty:
            train_df (pd.DataFrame): DataFrame z ocenami treningowymi.

        Zwraca:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Indeksy użytkowników, indeksy filmów i wartości ocen.
        """
        # Mapujemy userId na indeksy wewnętrzne modelu.
        user_idx_np = train_df["userId"].map(self.user_to_index).to_numpy()
        # Mapujemy movieId na indeksy wewnętrzne modelu.
        movie_idx_np = train_df["movieId"].map(self.movie_to_index).to_numpy()
        # Pobieramy surowe wartości ocen.
        ratings_np = train_df["rating"].to_numpy()

        # Konwertujemy indeksy użytkowników do tensora torch.
        user_idx = torch.tensor(
            user_idx_np,
            dtype=torch.long,
            device=self.device_obj,
        )
        # Konwertujemy indeksy filmów do tensora torch.
        movie_idx = torch.tensor(
            movie_idx_np,
            dtype=torch.long,
            device=self.device_obj,
        )
        # Konwertujemy oceny do tensora float32.
        ratings = torch.tensor(
            ratings_np,
            dtype=torch.float32,
            device=self.device_obj,
        )
        # Zwracamy przygotowane tensory treningowe.
        return user_idx, movie_idx, ratings

    def _run_epoch(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        ratings: torch.Tensor,
        sample_order: np.ndarray,
        optimizer: Optimizer,
    ) -> float:
        """Train model for one epoch and return training RMSE.

        EN:
        Iterates over mini-batches in shuffled order, performs gradient
        steps with MSE loss, and computes RMSE for the epoch.

        Args:
            user_idx (torch.Tensor): Tensor with user indices.
            movie_idx (torch.Tensor): Tensor with movie indices.
            ratings (torch.Tensor): Tensor with true rating values.
            sample_order (np.ndarray): Shuffled row indices.
            optimizer (Optimizer): Torch optimizer instance.

        Returns:
            float: RMSE computed on processed epoch samples.

        PL:
        Iteruje po mini-batchach w potasowanej kolejności, wykonuje kroki
        gradientowe z funkcją MSE i liczy RMSE dla całej epoki.

        Argumenty:
            user_idx (torch.Tensor): Tensor z indeksami użytkowników.
            movie_idx (torch.Tensor): Tensor z indeksami filmów.
            ratings (torch.Tensor): Tensor z prawdziwymi ocenami.
            sample_order (np.ndarray): Potasowane indeksy rekordów.
            optimizer (Optimizer): Instancja optymalizatora torch.

        Zwraca:
            float: RMSE obliczone na próbkach z danej epoki.
        """
        # Inicjalizujemy akumulator ważonej straty epoki.
        epoch_loss_sum = 0.0
        # Inicjalizujemy licznik próbek przetworzonych w epoce.
        samples_seen = 0

        # Iterujemy po batchach zgodnie z ustawionym batch_size.
        for batch_ids in self._iter_minibatch_indices(sample_order):
            # Konwertujemy indeksy batcha do tensora.
            batch_ids_t = torch.tensor(
                batch_ids,
                dtype=torch.long,
                device=self.device_obj,
            )

            # Wycinamy indeksy użytkowników dla batcha.
            batch_users = user_idx[batch_ids_t]
            # Wycinamy indeksy filmów dla batcha.
            batch_movies = movie_idx[batch_ids_t]
            # Wycinamy prawdziwe oceny dla batcha.
            batch_ratings = ratings[batch_ids_t]

            # Zerujemy gradienty przed krokiem optymalizacji.
            optimizer.zero_grad()
            # Wyliczamy predykcje modelu dla batcha.
            preds = self.torch_model(
                batch_users,
                batch_movies,
                self._global_mean_tensor,
            )
            # Liczymy średni błąd kwadratowy dla batcha.
            loss = torch.mean((preds - batch_ratings) ** 2)
            # Liczymy gradienty względem parametrów modelu.
            loss.backward()
            # Wykonujemy krok optymalizatora.
            optimizer.step()

            # Zliczamy faktyczny rozmiar bieżącego batcha.
            batch_size_actual = len(batch_ids)
            # Akumulujemy stratę ważoną rozmiarem batcha.
            epoch_loss_sum += float(loss.item()) * batch_size_actual
            # Aktualizujemy licznik przetworzonych próbek.
            samples_seen += batch_size_actual

        # Zabezpieczamy dzielenie przy pustej epoce.
        safe_count = max(samples_seen, 1)
        # Przeliczamy MSE epoki na RMSE.
        return float(np.sqrt(epoch_loss_sum / safe_count))

    def _iter_minibatch_indices(
        self,
        sample_order: np.ndarray,
    ) -> Iterator[np.ndarray]:
        """Yield mini-batch index arrays from shuffled order.

        EN:
        Splits shuffled sample indices into contiguous chunks controlled by
        `batch_size`.

        Args:
            sample_order (np.ndarray): Shuffled sample indices.

        Yields:
            np.ndarray: Slice of indices for one mini-batch.

        PL:
        Dzieli potasowane indeksy próbek na kolejne fragmenty sterowane
        parametrem `batch_size`.

        Argumenty:
            sample_order (np.ndarray): Potasowane indeksy próbek.

        Zwraca:
            Iterator[np.ndarray]: Kolejne tablice indeksów mini-batchy.
        """
        # Pobieramy liczbę wszystkich próbek.
        n_samples = len(sample_order)
        # Iterujemy po pozycjach startowych kolejnych batchy.
        for start_idx in range(0, n_samples, self.batch_size):
            # Wyliczamy pozycję końcową bieżącego batcha.
            end_idx = start_idx + self.batch_size
            # Zwracamy indeksy należące do bieżącego batcha.
            yield sample_order[start_idx:end_idx]

    def _print_epoch_log(self, epoch_idx: int, train_rmse: float) -> None:
        """Print epoch summary when verbose mode is enabled.

        EN:
        Formats and prints one line with epoch number and RMSE.

        Args:
            epoch_idx (int): Zero-based epoch index.
            train_rmse (float): Epoch training RMSE value.

        Returns:
            None: Produces console output only when verbose is true.

        PL:
        Formatuje i wypisuje jedną linię z numerem epoki i RMSE.

        Argumenty:
            epoch_idx (int): Indeks epoki liczony od zera.
            train_rmse (float): Wartość RMSE dla danej epoki.

        Zwraca:
            None: Generuje log konsolowy tylko gdy verbose jest true.
        """
        # Przerywamy, gdy logowanie jest wyłączone.
        if not self.verbose:
            return
        # Wypisujemy podsumowanie bieżącej epoki.
        print(
            f"Epoch {epoch_idx + 1:3d}/{self.epochs} | "
            f"train RMSE: {train_rmse:.4f}"
        )

    def _predict_known_pair(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a known user-movie pair.

        EN:
        Converts IDs to torch indices, runs model in evaluation mode,
        and returns scalar prediction.

        Args:
            user_id (int): Known user identifier.
            movie_id (int): Known movie identifier.

        Returns:
            float: Predicted rating for the pair.

        PL:
        Zamienia identyfikatory na indeksy torch, uruchamia model
        w trybie ewaluacji i zwraca skalarną predykcję.

        Argumenty:
            user_id (int): Znany identyfikator użytkownika.
            movie_id (int): Znany identyfikator filmu.

        Zwraca:
            float: Przewidziana ocena dla pary.
        """
        # Sprawdzamy, czy model przeszedł trening.
        self._check_is_fitted()
        # Sprawdzamy, czy wewnętrzne obiekty torch są gotowe.
        self._validate_torch_state()

        # Pobieramy indeks użytkownika z mapowania.
        user_idx = self.user_to_index[user_id]
        # Pobieramy indeks filmu z mapowania.
        movie_idx = self.movie_to_index[movie_id]

        # Przełączamy model w tryb ewaluacji.
        self.torch_model.eval()
        # Wyłączamy śledzenie gradientu podczas predykcji.
        with torch.no_grad():
            # Budujemy jednoelementowy tensor indeksu użytkownika.
            user_tensor = torch.tensor(
                [user_idx],
                dtype=torch.long,
                device=self.device_obj,
            )
            # Budujemy jednoelementowy tensor indeksu filmu.
            movie_tensor = torch.tensor(
                [movie_idx],
                dtype=torch.long,
                device=self.device_obj,
            )
            # Uruchamiamy model i pobieramy wartość skalarną.
            pred_value = self.torch_model(
                user_tensor,
                movie_tensor,
                self._global_mean_tensor,
            ).item()

        # Zwracamy wynik jako float.
        return float(pred_value)

    def predict(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for many pairs with vectorized known-path.

        EN:
        Predicts known pairs in one torch batch and applies base cold-start
        fallback for unknown pairs, then clips and rounds outputs.

        Args:
            pairs_df (pd.DataFrame): DataFrame with userId and movieId.

        Returns:
            np.ndarray: Array with predicted ratings.

        PL:
        Dla znanych par liczy predykcje wektorowo w torch, dla nieznanych
        stosuje fallback z klasy bazowej, a na końcu klipuje i zaokrągla.

        Argumenty:
            pairs_df (pd.DataFrame): DataFrame z kolumnami userId i movieId.

        Zwraca:
            np.ndarray: Tablica z przewidzianymi ocenami.
        """
        # Sprawdzamy, czy model jest wytrenowany.
        self._check_is_fitted()
        # Walidujemy wymagane kolumny wejściowe.
        self._validate_pair_columns(pairs_df)
        # Sprawdzamy, czy obiekty torch są gotowe.
        self._validate_torch_state()

        # Tworzymy wynikową tablicę ocen.
        result = np.empty(len(pairs_df), dtype=np.float32)
        # Budujemy maskę par znanych dla modelu.
        known_mask = (
            pairs_df["userId"].isin(self.user_to_index)
            & pairs_df["movieId"].isin(self.movie_to_index)
        )

        # Dla znanych par wykonujemy predykcję wsadową.
        self._predict_known_pairs_batch(
            pairs_df=pairs_df,
            known_mask=known_mask,
            result=result,
        )
        # Dla nieznanych par wykonujemy fallback cold-start.
        self._predict_unknown_pairs(
            pairs_df=pairs_df,
            known_mask=known_mask,
            result=result,
        )

        # Przycinamy oceny do dozwolonego zakresu.
        result = np.clip(result, 0.5, 5.0)
        # Zaokrąglamy oceny do najbliższej połówki.
        result = np.round(result * 2.0) / 2.0
        # Zwracamy gotowe predykcje.
        return result

    def _predict_known_pairs_batch(
        self,
        pairs_df: pd.DataFrame,
        known_mask: pd.Series,
        result: np.ndarray,
    ) -> None:
        """Fill predictions for known pairs in one torch pass.

        EN:
        Selects known rows, converts them to index tensors, predicts in
        evaluation mode, and writes values into result array.

        Args:
            pairs_df (pd.DataFrame): Input user-movie pairs.
            known_mask (pd.Series): Mask marking known pairs.
            result (np.ndarray): Output array modified in place.

        Returns:
            None: Updates `result` for known rows.

        PL:
        Wybiera znane rekordy, mapuje je do tensorów indeksów, wylicza
        predykcje w trybie ewaluacji i wpisuje je do tablicy wynikowej.

        Argumenty:
            pairs_df (pd.DataFrame): Wejściowe pary user-film.
            known_mask (pd.Series): Maska oznaczająca znane pary.
            result (np.ndarray): Tablica wyjściowa modyfikowana w miejscu.

        Zwraca:
            None: Aktualizuje `result` dla znanych wierszy.
        """
        # Kończymy, jeśli nie ma żadnych znanych par.
        if not known_mask.any():
            return

        # Filtrujemy tylko znane rekordy wejściowe.
        known_df = pairs_df.loc[known_mask]
        # Mapujemy userId znanych par na indeksy.
        known_user_idx_np = known_df["userId"].map(
            self.user_to_index
        ).to_numpy()
        # Mapujemy movieId znanych par na indeksy.
        known_movie_idx_np = known_df["movieId"].map(
            self.movie_to_index
        ).to_numpy()

        # Tworzymy tensor indeksów użytkowników.
        user_idx = torch.tensor(
            known_user_idx_np,
            dtype=torch.long,
            device=self.device_obj,
        )
        # Tworzymy tensor indeksów filmów.
        movie_idx = torch.tensor(
            known_movie_idx_np,
            dtype=torch.long,
            device=self.device_obj,
        )

        # Przełączamy model na tryb ewaluacji.
        self.torch_model.eval()
        # Wyłączamy obliczanie gradientów przy predykcji.
        with torch.no_grad():
            # Liczymy predykcje dla całego batcha znanych par.
            preds_known = self.torch_model(
                user_idx,
                movie_idx,
                self._global_mean_tensor,
            ).cpu().numpy()

        # Wypełniamy wyniki tylko na pozycjach znanych par.
        result[known_mask.to_numpy()] = preds_known.astype(np.float32)

    def _predict_unknown_pairs(
        self,
        pairs_df: pd.DataFrame,
        known_mask: pd.Series,
        result: np.ndarray,
    ) -> None:
        """Fill predictions for unknown pairs using cold-start fallback.

        EN:
        Iterates through unknown rows and computes fallback prediction from
        BaseRecommender statistics.

        Args:
            pairs_df (pd.DataFrame): Input user-movie pairs.
            known_mask (pd.Series): Mask marking known pairs.
            result (np.ndarray): Output array modified in place.

        Returns:
            None: Updates `result` for unknown rows.

        PL:
        Iteruje po nieznanych rekordach i wylicza fallback cold-start
        na podstawie statystyk z klasy bazowej.

        Argumenty:
            pairs_df (pd.DataFrame): Wejściowe pary user-film.
            known_mask (pd.Series): Maska oznaczająca znane pary.
            result (np.ndarray): Tablica wyjściowa modyfikowana w miejscu.

        Zwraca:
            None: Aktualizuje `result` dla nieznanych wierszy.
        """
        # Kończymy, jeśli nie ma żadnych nieznanych par.
        if not (~known_mask).any():
            return

        # Pobieramy indeksy nieznanych rekordów w danych wejściowych.
        unknown_indices = np.where(~known_mask.to_numpy())[0]
        # Iterujemy po wszystkich nieznanych pozycjach.
        for idx in unknown_indices:
            # Pobieramy rekord wejściowy dla danej pozycji.
            row = pairs_df.iloc[idx]
            # Wyliczamy predykcję fallback dla nowej pary.
            pred_value = self._cold_start_prediction(
                user_id=int(row["userId"]),
                movie_id=int(row["movieId"]),
            )
            # Zapisujemy wynik w tablicy docelowej.
            result[idx] = np.float32(pred_value)

    def _validate_torch_state(self) -> None:
        """Validate torch-specific fields required for prediction.

        EN:
        Checks whether internal torch model and global mean tensor are
        initialized.

        Raises:
            ValueError: If model or global mean tensor is missing.

        Returns:
            None: Validation only.

        PL:
        Sprawdza, czy wewnętrzny model torch i tensor średniej globalnej
        zostały poprawnie zainicjalizowane.

        Wyjątki:
            ValueError: Gdy brakuje modelu lub tensora średniej globalnej.

        Zwraca:
            None: Funkcja wykonuje tylko walidację.
        """
        # Sprawdzamy obecność modelu torch.
        if self.torch_model is None:
            raise ValueError("Torch model is not initialized.")
        # Sprawdzamy obecność tensora średniej globalnej.
        if self._global_mean_tensor is None:
            raise ValueError("Global mean tensor is not initialized.")

    def _create_optimizer(
        self,
        parameters: Iterator[torch.nn.Parameter],
    ) -> Optimizer:
        """Create optimizer configured by `optimizer_name`.

        EN:
        Supports SGD and Adam optimizers with shared learning rate and
        weight decay settings.

        Args:
            parameters (Iterator[torch.nn.Parameter]): Model parameters.

        Returns:
            Optimizer: Initialized optimizer instance.

        Raises:
            ValueError: If optimizer name is unsupported.

        PL:
        Tworzy optymalizator SGD albo Adam ze wspólnymi ustawieniami
        learning rate i weight decay.

        Argumenty:
            parameters (Iterator[torch.nn.Parameter]): Parametry modelu.

        Zwraca:
            Optimizer: Zainicjalizowana instancja optymalizatora.

        Wyjątki:
            ValueError: Gdy nazwa optymalizatora nie jest wspierana.
        """
        # Tworzymy klasyczny optymalizator SGD.
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        # Tworzymy optymalizator Adam.
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        # Zgłaszamy błąd dla nieobsługiwanej nazwy.
        raise ValueError(
            f"Unsupported optimizer_name={self.optimizer_name!r}. "
            "Use 'sgd' or 'adam'."
        )
