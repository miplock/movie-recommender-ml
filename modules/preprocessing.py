# Preprocessing utilities for missing-value imputation in rating matrices.
import numpy as np


def impute_missing_values(
    Z: np.ndarray,
    strategy: str = "movie_mean",
    global_mean: float | None = None,
) -> np.ndarray:
    """Impute missing values in rating matrix using selected strategy.

    EN:
    Fill NaN values with zero, movie means, or user means.

    Args:
        Z (np.ndarray): Rating matrix with NaN entries for missing ratings.
        strategy (str): One of: zero, movie_mean, user_mean.
        global_mean (float | None): Fallback mean when row/col is all NaN.

    Returns:
        np.ndarray: Matrix with missing values imputed.

    Raises:
        ValueError: If strategy is unknown.

    PL:
    Uzupełnia brakujące wartości NaN zerem, średnią filmu lub użytkownika.

    Argumenty:
        Z (np.ndarray): Macierz ocen z brakami oznaczonymi przez NaN.
        strategy (str): Jedna z wartości: zero, movie_mean, user_mean.
        global_mean (float | None): Fallback, gdy cały wiersz/kolumna to NaN.

    Zwraca:
        np.ndarray: Macierz po imputacji brakujących wartości.

    Wyjątki:
        ValueError: Gdy podano nieznaną strategię.
    """
    # Tworzymy kopię wejścia, aby nie modyfikować oryginalnej macierzy.
    Z_filled = Z.copy()

    # Strategia "zero": wszystkie NaN zamieniamy na 0.0.
    if strategy == "zero":
        return np.nan_to_num(Z_filled, nan=0.0)

    # Strategia "movie_mean": imputacja średnią dla każdej kolumny.
    if strategy == "movie_mean":
        # Liczymy średnią kolumnową, ignorując NaN.
        col_means = np.nanmean(Z_filled, axis=0)
        # Ustalamy fallback dla pustych kolumn.
        fallback = (
            global_mean if global_mean is not None else np.nanmean(Z_filled)
        )
        # Tam gdzie średnia kolumny jest NaN, podstawiamy fallback.
        col_means = np.where(np.isnan(col_means), fallback, col_means)

        # Pobieramy indeksy wszystkich brakujących komórek.
        inds = np.where(np.isnan(Z_filled))
        # Wypełniamy braki średnią odpowiedniej kolumny.
        Z_filled[inds] = col_means[inds[1]]
        # Zwracamy macierz po imputacji.
        return Z_filled

    # Strategia "user_mean": imputacja średnią dla każdego wiersza.
    if strategy == "user_mean":
        # Liczymy średnią wierszową, ignorując NaN.
        row_means = np.nanmean(Z_filled, axis=1)
        # Ustalamy fallback dla pustych wierszy.
        fallback = (
            global_mean if global_mean is not None else np.nanmean(Z_filled)
        )
        # Tam gdzie średnia wiersza jest NaN, podstawiamy fallback.
        row_means = np.where(np.isnan(row_means), fallback, row_means)

        # Pobieramy indeksy wszystkich brakujących komórek.
        inds = np.where(np.isnan(Z_filled))
        # Wypełniamy braki średnią odpowiedniego wiersza.
        Z_filled[inds] = row_means[inds[0]]
        # Zwracamy macierz po imputacji.
        return Z_filled

    # Dla nieznanej strategii zwracamy jednoznaczny błąd.
    raise ValueError(f"Unknown imputation strategy: {strategy}")
