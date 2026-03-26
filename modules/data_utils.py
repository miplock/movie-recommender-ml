# Data utilities for converting rating tables into matrix representation.
import numpy as np
import pandas as pd


def build_rating_matrix(
    df: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> np.ndarray:
    """Build sparse user-movie matrix from rating rows.

    EN:
    Convert ratings table into dense matrix with NaN for missing entries.

    Args:
        df (pd.DataFrame): Input table with userId, movieId, and rating.
        user_to_index (dict[int, int]): Mapping from userId to matrix row.
        movie_to_index (dict[int, int]): Mapping from movieId to matrix col.

    Returns:
        np.ndarray: Rating matrix filled with known ratings and NaN gaps.

    PL:
    Konwertuje tabelę ocen do macierzy z NaN w miejscach brakujących danych.

    Argumenty:
        df (pd.DataFrame): Tabela z kolumnami userId, movieId, rating.
        user_to_index (dict[int, int]): Mapa userId -> indeks wiersza.
        movie_to_index (dict[int, int]): Mapa movieId -> indeks kolumny.

    Zwraca:
        np.ndarray: Macierz ocen z wpisanymi ocenami i brakami jako NaN.
    """
    # Liczymy liczbę użytkowników na podstawie mapowania.
    n_users = len(user_to_index)
    # Liczymy liczbę filmów na podstawie mapowania.
    n_movies = len(movie_to_index)

    # Tworzymy macierz i wypełniamy ją NaN dla brakujących ocen.
    Z = np.full((n_users, n_movies), np.nan, dtype=float)

    # Iterujemy po każdym rekordzie wejściowym (userId, movieId, rating).
    for row in df.itertuples(index=False):
        # Zamieniamy userId na indeks wiersza macierzy.
        u_idx = user_to_index[row.userId]
        # Zamieniamy movieId na indeks kolumny macierzy.
        m_idx = movie_to_index[row.movieId]
        # Wpisujemy ocenę do odpowiedniej komórki macierzy.
        Z[u_idx, m_idx] = float(row.rating)

    # Zwracamy gotową macierz użytkownik-film.
    return Z
