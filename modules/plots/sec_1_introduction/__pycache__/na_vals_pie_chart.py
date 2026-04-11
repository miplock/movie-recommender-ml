# Script for generating and saving a pie chart of matrix missingness.
"""Create pie chart showing missing percentage in user-movie matrix.

EN:
Loads ratings data, computes matrix sparsity as missing vs observed
cells, builds a pie chart, and saves chart object plus metadata files.

PL:
Wczytuje dane ocen, oblicza rzadkość macierzy jako udział pustych oraz
wypełnionych komórek, tworzy wykres kołowy i zapisuje wykres z metadanymi.
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from modules.plots.plot_utils import load_palette
from modules.plots.plot_utils import set_plot_style

# Domyślna ścieżka do danych źródłowych z ocenami.
DEFAULT_INPUT_FILE = Path("data/ratings.csv")
# Domyślny katalog zapisu artefaktów wykresu.
DEFAULT_OUTPUT_DIR = Path("results/plots")
# Domyślna ścieżka pliku motywu kolorów.
DEFAULT_THEME_FILE = Path("theme/colors.yml")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for pie chart generation script.

    EN:
    Defines CLI options for input ratings file, output directory, and
    theme file used to style chart colors.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    PL:
    Definiuje opcje CLI dla pliku wejściowego, katalogu wyjściowego oraz
    pliku motywu używanego do kolorystyki wykresu.

    Zwraca:
        argparse.Namespace: Sparsowane argumenty wiersza poleceń.
    """
    # Tworzymy parser argumentów dla skryptu CLI.
    parser = argparse.ArgumentParser(
        description="Generate pie chart for matrix missingness."
    )
    # Dodajemy opcję ścieżki pliku wejściowego z ocenami.
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to ratings CSV file.",
    )
    # Dodajemy opcję katalogu, gdzie zapiszemy pliki wynikowe.
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for chart and metadata files.",
    )
    # Dodajemy opcję pliku motywu z paletą kolorów.
    parser.add_argument(
        "--theme_file",
        type=Path,
        default=DEFAULT_THEME_FILE,
        help="Path to YAML file with chart colors.",
    )
    # Zwracamy sparsowane argumenty użytkownika.
    return parser.parse_args()


def validate_input_file(input_file: Path) -> None:
    """Validate whether input ratings file exists.

    EN:
    Ensures the input file path points to an existing CSV source.

    Args:
        input_file (Path): Path to ratings CSV file.

    Raises:
        FileNotFoundError: If file does not exist.

    Returns:
        None: Validation only.

    PL:
    Sprawdza, czy ścieżka wejściowa wskazuje istniejący plik CSV.

    Argumenty:
        input_file (Path): Ścieżka do pliku CSV z ocenami.

    Wyjątki:
        FileNotFoundError: Gdy plik nie istnieje.

    Zwraca:
        None: Funkcja wykonuje tylko walidację.
    """
    # Sprawdzamy istnienie pliku wejściowego.
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")


def load_ratings(input_file: Path) -> pd.DataFrame:
    """Load ratings data from CSV file.

    EN:
    Reads source CSV and validates required columns for sparsity analysis.

    Args:
        input_file (Path): Path to ratings CSV file.

    Returns:
        pd.DataFrame: Loaded ratings table.

    Raises:
        ValueError: If required columns are missing.

    PL:
    Wczytuje źródłowy plik CSV i waliduje wymagane kolumny do analizy
    rzadkości macierzy.

    Argumenty:
        input_file (Path): Ścieżka do pliku CSV z ocenami.

    Zwraca:
        pd.DataFrame: Wczytana tabela ocen.

    Wyjątki:
        ValueError: Gdy brakuje wymaganych kolumn.
    """
    # Wczytujemy plik CSV do DataFrame.
    ratings_df = pd.read_csv(input_file)
    # Definiujemy zestaw wymaganych kolumn.
    required_columns = {"userId", "movieId", "rating"}
    # Wyznaczamy brakujące kolumny względem wymagań.
    missing_columns = required_columns - set(ratings_df.columns)
    # Zgłaszamy błąd, gdy schemat wejścia jest niepełny.
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    # Zwracamy poprawnie wczytane dane.
    return ratings_df


def compute_matrix_missingness(ratings_df: pd.DataFrame) -> dict[str, float]:
    """Compute observed and missing percentages in ratings matrix.

    EN:
    Uses unique users and movies to define full matrix size, then computes
    observed cell count from unique user-movie pairs and missing count.

    Args:
        ratings_df (pd.DataFrame): Ratings table with userId and movieId.

    Returns:
        dict[str, float]: Counts and percentages for observed/missing cells.

    PL:
    Wykorzystuje liczbę unikalnych użytkowników i filmów do wyznaczenia
    pełnego rozmiaru macierzy oraz liczy komórki wypełnione i puste.

    Argumenty:
        ratings_df (pd.DataFrame): Tabela ocen z userId i movieId.

    Zwraca:
        dict[str, float]: Liczności i procenty komórek pełnych oraz pustych.
    """
    # Liczymy liczbę unikalnych użytkowników.
    n_users = int(ratings_df["userId"].nunique())
    # Liczymy liczbę unikalnych filmów.
    n_movies = int(ratings_df["movieId"].nunique())
    # Liczymy pełny rozmiar macierzy user-film.
    total_cells = int(n_users * n_movies)

    # Usuwamy duplikaty par user-film przed zliczeniem obserwacji.
    unique_pairs = ratings_df[["userId", "movieId"]].drop_duplicates()
    # Liczymy liczbę wypełnionych komórek macierzy.
    observed_cells = int(len(unique_pairs))
    # Liczymy liczbę pustych komórek macierzy.
    missing_cells = int(total_cells - observed_cells)

    # Zabezpieczamy dzielenie, gdy macierz miałaby rozmiar zero.
    safe_total_cells = max(total_cells, 1)
    # Liczymy procent komórek wypełnionych.
    observed_pct = float(observed_cells / safe_total_cells * 100.0)
    # Liczymy procent komórek pustych.
    missing_pct = float(missing_cells / safe_total_cells * 100.0)

    # Zwracamy komplet metryk sparsity.
    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "total_cells": total_cells,
        "observed_cells": observed_cells,
        "missing_cells": missing_cells,
        "observed_pct": observed_pct,
        "missing_pct": missing_pct,
    }


def build_pie_chart(
    missingness: dict[str, float],
    theme_file: Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Build pie chart figure for matrix missingness summary.

    EN:
    Creates matplotlib figure with two slices: missing and observed cells.

    Args:
        missingness (dict[str, float]): Computed sparsity metrics.
        theme_file (Path): Path to YAML theme used for chart colors.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes objects.

    PL:
    Tworzy figurę matplotlib z dwoma udziałami: puste i wypełnione
    komórki macierzy danych.

    Argumenty:
        missingness (dict[str, float]): Obliczone metryki rzadkości.
        theme_file (Path): Ścieżka motywu YAML dla kolorów wykresu.

    Zwraca:
        tuple[plt.Figure, plt.Axes]: Obiekty figury i osi wykresu.
    """
    # Ustawiamy spójny styl wykresów z motywu projektu.
    set_plot_style(theme_file=theme_file)
    # Pobieramy paletę kolorów dla serii wykresowych.
    palette = load_palette(theme_file=theme_file)

    # Tworzymy figurę i oś pod wykres kołowy.
    fig, ax = plt.subplots(figsize=(7, 7))
    # Definiujemy wartości procentowe dla dwóch części wykresu.
    chart_values = [
        missingness["missing_pct"],
        missingness["observed_pct"],
    ]
    # Definiujemy etykiety opisujące udziały.
    chart_labels = ["Puste komórki", "Wypełnione komórki"]
    # Wybieramy kolory z palety motywu dla obu fragmentów.
    chart_colors = [palette[0], palette[1]]

    # Tworzymy wykres kołowy z opisem procentowym.
    ax.pie(
        chart_values,
        labels=chart_labels,
        autopct="%1.2f%%",
        startangle=90,
        colors=chart_colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    # Zapewniamy okrągły kształt wykresu.
    ax.axis("equal")
    # Ustawiamy tytuł wykresu.
    ax.set_title("Udział pustych komórek w macierzy user-film")
    # Zwracamy gotową figurę i oś.
    return fig, ax


def build_output_paths(output_dir: Path) -> tuple[Path, Path, str]:
    """Build timestamped output file paths for chart and metadata.

    EN:
    Creates output directory and returns paths based on
    `sec_1_introduction_{date_time}` naming pattern.

    Args:
        output_dir (Path): Directory where artifacts should be saved.

    Returns:
        tuple[Path, Path, str]: Chart path, metadata path, and timestamp.

    PL:
    Tworzy katalog wyjściowy i zwraca ścieżki zgodne ze wzorcem nazwy
    `sec_1_introduction_{data_godzina}`.

    Argumenty:
        output_dir (Path): Katalog docelowy zapisu artefaktów.

    Zwraca:
        tuple[Path, Path, str]: Ścieżki wykresu, metadanych i znacznik czasu.
    """
    # Tworzymy katalog wynikowy, jeśli jeszcze nie istnieje.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Budujemy znacznik czasu data+godzina do nazwy pliku.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Budujemy bazową nazwę pliku zgodnie z wymaganym prefiksem.
    base_name = f"sec_1_introduction_{timestamp}"
    # Tworzymy ścieżkę pliku pickle z obiektem wykresu.
    chart_path = output_dir / f"{base_name}.pkl"
    # Tworzymy ścieżkę pliku JSON z metadanymi.
    metadata_path = output_dir / f"{base_name}.json"
    # Zwracamy komplet ścieżek oraz znacznik czasu.
    return chart_path, metadata_path, timestamp


def save_chart_pickle(fig: plt.Figure, chart_path: Path) -> None:
    """Save matplotlib figure object to pickle file.

    EN:
    Serializes figure object into a `.pkl` file for later loading.

    Args:
        fig (plt.Figure): Figure object to save.
        chart_path (Path): Target pickle path.

    Returns:
        None: Writes file to disk.

    PL:
    Serializuje obiekt figury do pliku `.pkl`, aby można było go później
    wczytać.

    Argumenty:
        fig (plt.Figure): Obiekt figury do zapisu.
        chart_path (Path): Docelowa ścieżka pliku pickle.

    Zwraca:
        None: Zapisuje plik na dysku.
    """
    # Otwieramy plik wyjściowy w trybie binarnym.
    with chart_path.open("wb") as file_obj:
        # Zapisujemy obiekt figury do formatu pickle.
        pickle.dump(fig, file_obj)


def save_metadata_json(metadata: dict, metadata_path: Path) -> None:
    """Save metadata dictionary to JSON file.

    EN:
    Stores chart generation metadata and computed sparsity statistics.

    Args:
        metadata (dict): Metadata content to write.
        metadata_path (Path): Path to output JSON file.

    Returns:
        None: Writes JSON file to disk.

    PL:
    Zapisuje metadane generowania wykresu oraz obliczone statystyki
    rzadkości do pliku JSON.

    Argumenty:
        metadata (dict): Zawartość metadanych do zapisu.
        metadata_path (Path): Ścieżka wyjściowego pliku JSON.

    Zwraca:
        None: Zapisuje plik JSON na dysk.
    """
    # Otwieramy plik metadanych w trybie tekstowym UTF-8.
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        # Zapisujemy metadane jako czytelny JSON.
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    """Run full pie chart generation workflow.

    EN:
    Parses CLI arguments, computes matrix missingness, creates pie chart,
    and writes chart pickle plus metadata file in output directory.

    Returns:
        None: Produces output artifacts and prints saved file paths.

    PL:
    Parsuje argumenty CLI, oblicza rzadkość macierzy, tworzy wykres
    kołowy i zapisuje plik pickle oraz plik metadanych.

    Zwraca:
        None: Tworzy artefakty wyjściowe i wypisuje ścieżki plików.
    """
    # Parsujemy argumenty przekazane do skryptu.
    args = parse_arguments()
    # Walidujemy istnienie pliku wejściowego.
    validate_input_file(input_file=args.input_file)
    # Wczytujemy dane ocen do DataFrame.
    ratings_df = load_ratings(input_file=args.input_file)
    # Liczymy statystyki pustych i wypełnionych komórek.
    missingness = compute_matrix_missingness(ratings_df=ratings_df)

    # Budujemy wykres kołowy prezentujący udział pustych danych.
    fig, _ = build_pie_chart(
        missingness=missingness,
        theme_file=args.theme_file,
    )
    # Tworzymy ścieżki plików wyjściowych z datą i godziną.
    chart_path, metadata_path, timestamp = build_output_paths(
        output_dir=args.output_dir
    )
    # Zapisujemy obiekt wykresu do pliku pickle.
    save_chart_pickle(fig=fig, chart_path=chart_path)

    # Budujemy słownik metadanych dla zapisanego wykresu.
    metadata = {
        "created_at_local": timestamp,
        "input_file": str(args.input_file),
        "output_chart_file": str(chart_path),
        "output_metadata_file": str(metadata_path),
        "n_users": missingness["n_users"],
        "n_movies": missingness["n_movies"],
        "total_cells": missingness["total_cells"],
        "observed_cells": missingness["observed_cells"],
        "missing_cells": missingness["missing_cells"],
        "observed_pct": missingness["observed_pct"],
        "missing_pct": missingness["missing_pct"],
    }
    # Zapisujemy metadane do pliku JSON o tej samej nazwie bazowej.
    save_metadata_json(metadata=metadata, metadata_path=metadata_path)
    # Zwalniamy zasoby figury po zapisie.
    plt.close(fig)

    # Informujemy użytkownika o zapisanych artefaktach.
    print(f"[INFO] Saved pie chart pickle to: {chart_path}")
    print(f"[INFO] Saved metadata JSON to: {metadata_path}")


if __name__ == "__main__":
    # Uruchamiamy główny przepływ tylko przy bezpośrednim wywołaniu pliku.
    main()
