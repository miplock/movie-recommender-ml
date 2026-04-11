# Script generating distribution of user-rated movie percentages.
"""Create chart for percentage of catalog rated by each user.

EN:
Loads ratings data, computes for each user the percentage of unique
movies they have rated, creates histogram chart, and saves chart with
metadata.

PL:
Wczytuje dane ocen, liczy dla każdego użytkownika procent unikalnych
filmów z katalogu, które ocenił, tworzy histogram i zapisuje wykres
z metadanymi.
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Wyznaczamy katalog główny repozytorium projektu.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Dodajemy katalog projektu do sys.path, jeśli jeszcze go tam nie ma.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.plots.plot_utils import load_palette
from modules.plots.plot_utils import set_plot_style

# Domyślna ścieżka danych wejściowych.
DEFAULT_INPUT_FILE = Path("data/ratings.csv")
# Domyślny katalog zapisu wyników.
DEFAULT_OUTPUT_DIR = Path("results/plots")
# Domyślna ścieżka motywu wykresów.
DEFAULT_THEME_FILE = Path("theme/colors.yml")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the user coverage chart script.

    EN:
    Defines input file, output directory, and theme file arguments.

    Returns:
        argparse.Namespace: Parsed command-line parameters.

    PL:
    Definiuje argumenty pliku wejściowego, katalogu wyjściowego oraz
    pliku motywu.

    Zwraca:
        argparse.Namespace: Sparsowane parametry z wiersza poleceń.
    """
    # Tworzymy parser argumentów.
    parser = argparse.ArgumentParser(
        description="Chart of percentage of rated movies per user."
    )
    # Dodajemy opcję pliku wejściowego.
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to input ratings CSV file.",
    )
    # Dodajemy opcję katalogu zapisu wyników.
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files.",
    )
    # Dodajemy opcję motywu kolorów.
    parser.add_argument(
        "--theme_file",
        type=Path,
        default=DEFAULT_THEME_FILE,
        help="Path to YAML color theme file.",
    )
    # Dodajemy opcję zapisu dodatkowej wersji PNG wykresu.
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Also save chart as PNG file.",
    )
    # Zwracamy sparsowane argumenty.
    return parser.parse_args()


def validate_input(input_file: Path) -> None:
    """Validate input ratings file existence.

    EN:
    Checks whether the input CSV path exists in filesystem.

    Args:
        input_file (Path): Path to ratings CSV.

    Raises:
        FileNotFoundError: If file does not exist.

    Returns:
        None: Validation only.

    PL:
    Sprawdza, czy wskazany plik CSV istnieje w systemie plików.

    Argumenty:
        input_file (Path): Ścieżka do pliku CSV z ocenami.

    Wyjątki:
        FileNotFoundError: Gdy plik nie istnieje.

    Zwraca:
        None: Funkcja wykonuje tylko walidację.
    """
    # Sprawdzamy, czy plik wejściowy istnieje.
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")


def load_ratings(input_file: Path) -> pd.DataFrame:
    """Load ratings CSV and validate required columns.

    EN:
    Reads ratings table and validates presence of userId and movieId.

    Args:
        input_file (Path): Path to ratings CSV.

    Returns:
        pd.DataFrame: Loaded ratings data.

    Raises:
        ValueError: If required columns are missing.

    PL:
    Wczytuje tabelę ocen i waliduje obecność kolumn userId i movieId.

    Argumenty:
        input_file (Path): Ścieżka do pliku CSV z ocenami.

    Zwraca:
        pd.DataFrame: Wczytane dane ocen.

    Wyjątki:
        ValueError: Gdy brakuje wymaganych kolumn.
    """
    # Wczytujemy dane z pliku CSV.
    ratings_df = pd.read_csv(input_file)
    # Określamy wymagane kolumny minimalne.
    required_columns = {"userId", "movieId"}
    # Wyznaczamy brakujące kolumny.
    missing_columns = required_columns - set(ratings_df.columns)
    # Jeśli brakuje kolumn, zgłaszamy błąd walidacji.
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    # Zwracamy poprawny DataFrame.
    return ratings_df


def compute_user_movie_coverage(
    ratings_df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    """Compute percentage of catalog rated by each user.

    EN:
    Calculates unique movie count per user, divides by total unique movie
    count, and returns per-user percentages with summary statistics.

    Args:
        ratings_df (pd.DataFrame): Ratings data with userId and movieId.

    Returns:
        tuple[pd.Series, dict[str, float]]: Per-user percentages and stats.

    PL:
    Liczy liczbę unikalnych filmów ocenionych przez użytkownika, dzieli
    przez całkowitą liczbę unikalnych filmów i zwraca procenty oraz
    statystyki podsumowujące.

    Argumenty:
        ratings_df (pd.DataFrame): Dane ocen z kolumnami userId i movieId.

    Zwraca:
        tuple[pd.Series, dict[str, float]]:
        Procenty per użytkownik i statystyki.
    """
    # Liczymy liczbę unikalnych filmów w całym zbiorze.
    n_unique_movies = int(ratings_df["movieId"].nunique())
    # Grupujemy po userId i liczymy unikalne filmy per użytkownik.
    movies_per_user = ratings_df.groupby("userId")["movieId"].nunique()

    # Zabezpieczamy dzielenie przez zero.
    safe_total_movies = max(n_unique_movies, 1)
    # Przeliczamy liczniki na procent katalogu filmów.
    coverage_pct = movies_per_user / safe_total_movies * 100.0

    # Budujemy słownik statystyk opisowych.
    stats = {
        "n_users": int(coverage_pct.shape[0]),
        "n_unique_movies": n_unique_movies,
        "min_pct": float(coverage_pct.min()),
        "mean_pct": float(coverage_pct.mean()),
        "median_pct": float(coverage_pct.median()),
        "max_pct": float(coverage_pct.max()),
    }
    # Zwracamy serię procentów i statystyki.
    return coverage_pct, stats


def build_chart(
    coverage_pct: pd.Series,
    theme_file: Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Build histogram for user movie coverage percentages.

    EN:
    Creates histogram of user coverage percentages using project theme.

    Args:
        coverage_pct (pd.Series): Percentage values per user.
        theme_file (Path): Path to YAML theme file.

    Returns:
        tuple[plt.Figure, plt.Axes]: Created matplotlib figure and axis.

    PL:
    Tworzy histogram procentowego pokrycia katalogu filmów przez
    użytkowników z użyciem motywu projektu.

    Argumenty:
        coverage_pct (pd.Series): Procenty przypisane do użytkowników.
        theme_file (Path): Ścieżka do pliku motywu YAML.

    Zwraca:
        tuple[plt.Figure, plt.Axes]: Utworzona figura i oś matplotlib.
    """
    # Ustawiamy styl i paletę wykresów.
    set_plot_style(theme_file=theme_file)
    # Pobieramy paletę kolorów z motywu.
    palette = load_palette(theme_file=theme_file)

    # Tworzymy figurę i oś wykresu.
    fig, ax = plt.subplots(figsize=(10, 6))
    # Tworzymy histogram procentów z 20 przedziałami.
    ax.hist(
        coverage_pct,
        bins=20,
        color=palette[0],
        edgecolor="white",
        linewidth=1.0,
    )
    # Ustawiamy zakres osi X na wartości procentowe.
    ax.set_xlim(0.0, 100.0)
    # Ustawiamy opis osi X.
    ax.set_xlabel("% filmów ocenionych przez użytkownika")
    # Ustawiamy opis osi Y.
    ax.set_ylabel("Liczba użytkowników")
    # Ustawiamy tytuł wykresu.
    ax.set_title("Rozkład % filmów ocenionych przez użytkowników")
    # Dociągamy układ, by etykiety nie nachodziły.
    fig.tight_layout()
    # Zwracamy gotowy wykres.
    return fig, ax


def build_output_paths(
    output_dir: Path,
) -> tuple[Path, Path, Path, str]:
    """Build output paths for chart and metadata files.

    EN:
    Creates timestamped filenames in output directory for pickle chart and
    JSON metadata files.

    Args:
        output_dir (Path): Target directory for output artifacts.

    Returns:
        tuple[Path, Path, Path, str]:
        Chart path, metadata path, PNG path, and timestamp.

    PL:
    Tworzy nazwy plików z datą i godziną w katalogu wynikowym dla wykresu
    pickle i metadanych JSON.

    Argumenty:
        output_dir (Path): Katalog docelowy artefaktów.

    Zwraca:
        tuple[Path, Path, Path, str]:
        Ścieżka wykresu, metadanych, PNG i timestamp.
    """
    # Tworzymy katalog wyjściowy, jeśli nie istnieje.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Budujemy znacznik czasu data+godzina.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Budujemy bazową nazwę plików wynikowych.
    base_name = f"sec_1_user_movies_pct_{timestamp}"
    # Budujemy ścieżkę pliku wykresu pickle.
    chart_path = output_dir / f"{base_name}.pkl"
    # Budujemy ścieżkę pliku metadanych JSON.
    metadata_path = output_dir / f"{base_name}.json"
    # Budujemy ścieżkę pliku obrazu PNG.
    png_path = output_dir / f"{base_name}.png"
    # Zwracamy komplet ścieżek oraz znacznik czasu.
    return chart_path, metadata_path, png_path, timestamp


def save_chart(fig: plt.Figure, chart_path: Path) -> None:
    """Save matplotlib figure to pickle file.

    EN:
    Serializes generated figure object into `.pkl` file.

    Args:
        fig (plt.Figure): Figure to serialize.
        chart_path (Path): Path to output pickle file.

    Returns:
        None: Writes file to disk.

    PL:
    Serializuje utworzoną figurę matplotlib do pliku `.pkl`.

    Argumenty:
        fig (plt.Figure): Figura do serializacji.
        chart_path (Path): Ścieżka wyjściowa pliku pickle.

    Zwraca:
        None: Zapisuje plik na dysk.
    """
    # Otwieramy plik docelowy binarnie.
    with chart_path.open("wb") as file_obj:
        # Zapisujemy figurę do pliku pickle.
        pickle.dump(fig, file_obj)


def save_chart_png(fig: plt.Figure, png_path: Path) -> None:
    """Save figure object to PNG image file.

    EN:
    Exports matplotlib figure to raster PNG with fixed resolution.

    Args:
        fig (plt.Figure): Figure object to export.
        png_path (Path): Output PNG path.

    Returns:
        None: Writes PNG file to disk.

    PL:
    Eksportuje figurę matplotlib do rastrowego pliku PNG o stałej
    rozdzielczości.

    Argumenty:
        fig (plt.Figure): Obiekt figury do eksportu.
        png_path (Path): Ścieżka wyjściowa PNG.

    Zwraca:
        None: Zapisuje plik PNG na dysk.
    """
    # Zapisujemy figurę do pliku PNG z czytelną rozdzielczością.
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")


def save_metadata(metadata: dict, metadata_path: Path) -> None:
    """Save metadata dictionary to JSON file.

    EN:
    Stores summary statistics and artifact paths as JSON metadata.

    Args:
        metadata (dict): Metadata payload.
        metadata_path (Path): Target JSON file path.

    Returns:
        None: Writes metadata file.

    PL:
    Zapisuje statystyki podsumowujące oraz ścieżki artefaktów jako
    metadane w pliku JSON.

    Argumenty:
        metadata (dict): Ładunek metadanych.
        metadata_path (Path): Docelowa ścieżka JSON.

    Zwraca:
        None: Zapisuje plik metadanych.
    """
    # Otwieramy plik JSON do zapisu tekstowego.
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        # Zapisujemy metadane w czytelnym formacie.
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    """Run chart generation pipeline for user movie coverage.

    EN:
    Parses arguments, computes per-user coverage percentages, generates
    chart, and saves both chart pickle and metadata JSON.

    Returns:
        None: Produces output files on disk.

    PL:
    Parsuje argumenty, liczy procenty pokrycia per użytkownik, generuje
    wykres i zapisuje pliki pickle oraz JSON z metadanymi.

    Zwraca:
        None: Tworzy pliki wynikowe na dysku.
    """
    # Parsujemy argumenty uruchomieniowe.
    args = parse_arguments()
    # Walidujemy plik wejściowy.
    validate_input(input_file=args.input_file)
    # Wczytujemy dane ocen.
    ratings_df = load_ratings(input_file=args.input_file)

    # Liczymy procent filmów ocenionych przez każdego użytkownika.
    coverage_pct, stats = compute_user_movie_coverage(ratings_df=ratings_df)
    # Tworzymy wykres histogramu.
    fig, _ = build_chart(
        coverage_pct=coverage_pct,
        theme_file=args.theme_file,
    )

    # Budujemy ścieżki wyjściowe dla artefaktów.
    chart_path, metadata_path, png_path, timestamp = build_output_paths(
        output_dir=args.output_dir
    )
    # Zapisujemy figurę do pliku pickle.
    save_chart(fig=fig, chart_path=chart_path)
    # Opcjonalnie zapisujemy również wersję PNG.
    if args.save_png:
        save_chart_png(fig=fig, png_path=png_path)

    # Budujemy komplet metadanych do zapisu.
    metadata = {
        "created_at_local": timestamp,
        "input_file": str(args.input_file),
        "chart_file": str(chart_path),
        "png_file": str(png_path) if args.save_png else None,
        "metadata_file": str(metadata_path),
        "metric": "percent_of_movies_rated_per_user",
        **stats,
    }
    # Zapisujemy metadane do pliku JSON.
    save_metadata(metadata=metadata, metadata_path=metadata_path)

    # Zamykamy figurę po zapisie.
    plt.close(fig)
    # Wypisujemy informacje o zapisanych plikach.
    print(f"[INFO] Saved chart pickle to: {chart_path}")
    if args.save_png:
        print(f"[INFO] Saved chart PNG to: {png_path}")
    print(f"[INFO] Saved metadata JSON to: {metadata_path}")


if __name__ == "__main__":
    # Uruchamiamy główny przepływ przy bezpośrednim wywołaniu pliku.
    main()
