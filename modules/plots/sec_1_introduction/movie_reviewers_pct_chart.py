# Script generating distribution of movie reviewer percentages.
"""Create chart for percentage of users rating each movie.

EN:
Loads ratings data, computes for each movie the percentage of unique
users that reviewed it, creates histogram chart, and saves artifacts.

PL:
Wczytuje dane ocen, liczy dla każdego filmu procent unikalnych
użytkowników, którzy go ocenili, tworzy histogram i zapisuje artefakty.
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
    """Parse CLI arguments for movie reviewer coverage script.

    EN:
    Defines input file, output directory, and theme file options.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    PL:
    Definiuje opcje pliku wejściowego, katalogu wynikowego i motywu.

    Zwraca:
        argparse.Namespace: Sparsowane argumenty z CLI.
    """
    # Tworzymy parser argumentów skryptu.
    parser = argparse.ArgumentParser(
        description="Chart of percentage of reviewers per movie."
    )
    # Dodajemy opcję pliku wejściowego CSV.
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to input ratings CSV file.",
    )
    # Dodajemy opcję katalogu wyjściowego.
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files.",
    )
    # Dodajemy opcję pliku motywu kolorów.
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
    # Zwracamy sparsowane argumenty użytkownika.
    return parser.parse_args()


def validate_input(input_file: Path) -> None:
    """Validate existence of input ratings file.

    EN:
    Ensures the provided input CSV file exists.

    Args:
        input_file (Path): Input path to ratings CSV.

    Raises:
        FileNotFoundError: If file path does not exist.

    Returns:
        None: Validation only.

    PL:
    Sprawdza, czy wskazany plik CSV wejściowy istnieje.

    Argumenty:
        input_file (Path): Wejściowa ścieżka pliku ocen CSV.

    Wyjątki:
        FileNotFoundError: Gdy podana ścieżka nie istnieje.

    Zwraca:
        None: Funkcja realizuje tylko walidację.
    """
    # Weryfikujemy istnienie pliku wejściowego.
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")


def load_ratings(input_file: Path) -> pd.DataFrame:
    """Load ratings and check required columns.

    EN:
    Reads ratings CSV into DataFrame and validates userId/movieId schema.

    Args:
        input_file (Path): Path to ratings CSV file.

    Returns:
        pd.DataFrame: Loaded ratings DataFrame.

    Raises:
        ValueError: If userId or movieId columns are missing.

    PL:
    Wczytuje plik ocen CSV do DataFrame i waliduje schemat userId/movieId.

    Argumenty:
        input_file (Path): Ścieżka do pliku ocen CSV.

    Zwraca:
        pd.DataFrame: Wczytany DataFrame z ocenami.

    Wyjątki:
        ValueError: Gdy brakuje kolumn userId lub movieId.
    """
    # Odczytujemy dane ocen z pliku CSV.
    ratings_df = pd.read_csv(input_file)
    # Definiujemy minimalny zestaw kolumn wymaganych.
    required_columns = {"userId", "movieId"}
    # Wyliczamy brakujące kolumny z wymaganej listy.
    missing_columns = required_columns - set(ratings_df.columns)
    # Zgłaszamy błąd, gdy schemat danych jest niepełny.
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    # Zwracamy poprawny DataFrame.
    return ratings_df


def compute_movie_reviewer_coverage(
    ratings_df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    """Compute percentage of users who reviewed each movie.

    EN:
    Calculates unique user count per movie, divides by total unique user
    count, and returns percentages with summary statistics.

    Args:
        ratings_df (pd.DataFrame): Ratings table with userId and movieId.

    Returns:
        tuple[pd.Series, dict[str, float]]:
        Per-movie percentages and summary statistics.

    PL:
    Liczy liczbę unikalnych użytkowników oceniających film, dzieli przez
    całkowitą liczbę unikalnych użytkowników i zwraca procenty oraz
    statystyki podsumowujące.

    Argumenty:
        ratings_df (pd.DataFrame): Tabela ocen z userId i movieId.

    Zwraca:
        tuple[pd.Series, dict[str, float]]:
        Procenty per film oraz statystyki podsumowujące.
    """
    # Liczymy całkowitą liczbę unikalnych użytkowników.
    n_unique_users = int(ratings_df["userId"].nunique())
    # Grupujemy po movieId i liczymy unikalnych recenzentów filmu.
    reviewers_per_movie = ratings_df.groupby("movieId")["userId"].nunique()

    # Zabezpieczamy dzielenie przed zerowym mianownikiem.
    safe_total_users = max(n_unique_users, 1)
    # Przeliczamy liczbę recenzentów na procent populacji użytkowników.
    reviewer_pct = reviewers_per_movie / safe_total_users * 100.0

    # Przygotowujemy statystyki opisowe dla metadanych.
    stats = {
        "n_movies": int(reviewer_pct.shape[0]),
        "n_unique_users": n_unique_users,
        "min_pct": float(reviewer_pct.min()),
        "mean_pct": float(reviewer_pct.mean()),
        "median_pct": float(reviewer_pct.median()),
        "max_pct": float(reviewer_pct.max()),
    }
    # Zwracamy serię procentów i statystyki.
    return reviewer_pct, stats


def build_chart(
    reviewer_pct: pd.Series,
    theme_file: Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Build histogram for movie reviewer percentages.

    EN:
    Creates histogram chart presenting distribution of reviewer coverage
    percentages across movies.

    Args:
        reviewer_pct (pd.Series): Percentage values per movie.
        theme_file (Path): Path to YAML theme file.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis objects.

    PL:
    Tworzy histogram pokazujący rozkład procentowego pokrycia recenzentów
    dla filmów.

    Argumenty:
        reviewer_pct (pd.Series): Wartości procentowe per film.
        theme_file (Path): Ścieżka do pliku motywu YAML.

    Zwraca:
        tuple[plt.Figure, plt.Axes]: Obiekty figury i osi.
    """
    # Ustawiamy wspólny styl i paletę z motywu.
    set_plot_style(theme_file=theme_file)
    # Wczytujemy paletę kolorów wykresowych.
    palette = load_palette(theme_file=theme_file)

    # Tworzymy figurę i oś wykresu histogramu.
    fig, ax = plt.subplots(figsize=(10, 6))
    # Rysujemy histogram procentów recenzentów dla filmów.
    ax.hist(
        reviewer_pct,
        bins=20,
        color=palette[1],
        edgecolor="white",
        linewidth=1.0,
    )
    # Ustawiamy zakres osi X jako 0-100%.
    ax.set_xlim(0.0, 100.0)
    # Ustawiamy etykietę osi X.
    ax.set_xlabel("% użytkowników, którzy ocenili film")
    # Ustawiamy etykietę osi Y.
    ax.set_ylabel("Liczba filmów")
    # Ustawiamy tytuł wykresu.
    ax.set_title("Rozkład % użytkowników oceniających każdy film")
    # Poprawiamy układ elementów figury.
    fig.tight_layout()
    # Zwracamy gotowy wykres.
    return fig, ax


def build_output_paths(
    output_dir: Path,
) -> tuple[Path, Path, Path, str]:
    """Build output paths for reviewer coverage artifacts.

    EN:
    Generates timestamp-based filenames in output directory for pickle
    chart and JSON metadata.

    Args:
        output_dir (Path): Directory for output artifacts.

    Returns:
        tuple[Path, Path, Path, str]:
        Chart path, metadata path, PNG path, and timestamp.

    PL:
    Generuje nazwy plików oparte na czasie w katalogu wynikowym dla
    wykresu pickle i metadanych JSON.

    Argumenty:
        output_dir (Path): Katalog docelowy artefaktów.

    Zwraca:
        tuple[Path, Path, Path, str]:
        Ścieżki wykresu, metadanych, PNG i timestamp.
    """
    # Tworzymy katalog wyjściowy, jeśli nie istnieje.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Budujemy znacznik czasu z datą i godziną.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Tworzymy bazową nazwę plików wynikowych.
    base_name = f"sec_1_movie_reviewers_pct_{timestamp}"
    # Definiujemy ścieżkę wykresu w formacie pickle.
    chart_path = output_dir / f"{base_name}.pkl"
    # Definiujemy ścieżkę metadanych w formacie JSON.
    metadata_path = output_dir / f"{base_name}.json"
    # Definiujemy ścieżkę obrazu wykresu w formacie PNG.
    png_path = output_dir / f"{base_name}.png"
    # Zwracamy komplet ścieżek i timestamp.
    return chart_path, metadata_path, png_path, timestamp


def save_chart(fig: plt.Figure, chart_path: Path) -> None:
    """Save figure object to pickle file.

    EN:
    Persists matplotlib figure as pickle artifact.

    Args:
        fig (plt.Figure): Figure object to save.
        chart_path (Path): Output pickle path.

    Returns:
        None: Writes chart file.

    PL:
    Zapisuje figurę matplotlib jako artefakt pickle.

    Argumenty:
        fig (plt.Figure): Obiekt figury do zapisu.
        chart_path (Path): Ścieżka wyjściowa pliku pickle.

    Zwraca:
        None: Zapisuje plik wykresu.
    """
    # Otwieramy plik wyjściowy binarnie.
    with chart_path.open("wb") as file_obj:
        # Serializujemy figurę do pickle.
        pickle.dump(fig, file_obj)


def save_chart_png(fig: plt.Figure, png_path: Path) -> None:
    """Save figure object to PNG image file.

    EN:
    Exports matplotlib figure to raster PNG with fixed resolution.

    Args:
        fig (plt.Figure): Figure object to export.
        png_path (Path): Output PNG path.

    Returns:
        None: Writes PNG file.

    PL:
    Eksportuje figurę matplotlib do rastrowego pliku PNG o ustalonej
    rozdzielczości.

    Argumenty:
        fig (plt.Figure): Obiekt figury do eksportu.
        png_path (Path): Ścieżka wyjściowa PNG.

    Zwraca:
        None: Zapisuje plik PNG.
    """
    # Zapisujemy figurę do pliku PNG z czytelną rozdzielczością.
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")


def save_metadata(metadata: dict, metadata_path: Path) -> None:
    """Save metadata payload to JSON file.

    EN:
    Stores chart context, summary statistics, and file paths as JSON.

    Args:
        metadata (dict): Metadata dictionary.
        metadata_path (Path): Destination JSON path.

    Returns:
        None: Writes metadata file.

    PL:
    Zapisuje kontekst wykresu, statystyki podsumowujące i ścieżki plików
    w postaci JSON.

    Argumenty:
        metadata (dict): Słownik metadanych.
        metadata_path (Path): Docelowa ścieżka pliku JSON.

    Zwraca:
        None: Zapisuje plik metadanych.
    """
    # Otwieramy plik metadanych do zapisu UTF-8.
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        # Zapisujemy metadane z czytelnym formatowaniem.
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    """Run chart generation pipeline for movie reviewer coverage.

    EN:
    Parses CLI args, computes per-movie reviewer percentages, generates
    histogram, and saves chart pickle with metadata JSON.

    Returns:
        None: Produces output artifact files.

    PL:
    Parsuje argumenty CLI, liczy procent recenzentów per film, tworzy
    histogram oraz zapisuje wykres pickle i metadane JSON.

    Zwraca:
        None: Tworzy pliki artefaktów wyjściowych.
    """
    # Parsujemy argumenty wejściowe skryptu.
    args = parse_arguments()
    # Walidujemy obecność pliku wejściowego.
    validate_input(input_file=args.input_file)
    # Wczytujemy dane ocen.
    ratings_df = load_ratings(input_file=args.input_file)

    # Obliczamy procent użytkowników oceniających każdy film.
    reviewer_pct, stats = compute_movie_reviewer_coverage(
        ratings_df=ratings_df
    )
    # Budujemy histogram dla obliczonych wartości.
    fig, _ = build_chart(
        reviewer_pct=reviewer_pct,
        theme_file=args.theme_file,
    )

    # Tworzymy ścieżki wyjściowe plików.
    chart_path, metadata_path, png_path, timestamp = build_output_paths(
        output_dir=args.output_dir
    )
    # Zapisujemy figurę wykresu do pickle.
    save_chart(fig=fig, chart_path=chart_path)
    # Opcjonalnie zapisujemy również wersję PNG.
    if args.save_png:
        save_chart_png(fig=fig, png_path=png_path)

    # Budujemy metadane obejmujące statystyki i kontekst zapisu.
    metadata = {
        "created_at_local": timestamp,
        "input_file": str(args.input_file),
        "chart_file": str(chart_path),
        "png_file": str(png_path) if args.save_png else None,
        "metadata_file": str(metadata_path),
        "metric": "percent_of_users_reviewing_each_movie",
        **stats,
    }
    # Zapisujemy metadane w formacie JSON.
    save_metadata(metadata=metadata, metadata_path=metadata_path)

    # Zamykamy figurę po zakończeniu zapisu.
    plt.close(fig)
    # Logujemy ścieżki zapisanych artefaktów.
    print(f"[INFO] Saved chart pickle to: {chart_path}")
    if args.save_png:
        print(f"[INFO] Saved chart PNG to: {png_path}")
    print(f"[INFO] Saved metadata JSON to: {metadata_path}")


if __name__ == "__main__":
    # Uruchamiamy główną funkcję tylko dla bezpośredniego wywołania.
    main()
