# Utilities for loading plot theme and applying seaborn style settings.
"""Plot helper utilities shared across reporting scripts.

EN:
Provides functions to read the theme color palette from YAML and apply
consistent seaborn styling in visualization scripts.

PL:
Zawiera funkcje do odczytu palety kolorów z YAML oraz ustawienia
spójnego stylu seaborn w skryptach wizualizacji.
"""
from pathlib import Path

import seaborn as sns
import yaml

# Domyślna ścieżka do pliku z kolorami motywu raportu.
DEFAULT_THEME_FILE = Path("theme/colors.yml")


def load_theme_colors(theme_file: Path = DEFAULT_THEME_FILE) -> dict:
    """Load theme color dictionary from YAML file.

    EN:
    Reads YAML file containing report color definitions and returns parsed
    dictionary with all configured sections.

    Args:
        theme_file (Path): Path to YAML file with colors.

    Returns:
        dict: Parsed YAML content with color configuration.

    Raises:
        FileNotFoundError: If theme file does not exist.
        ValueError: If YAML file is empty or invalid.

    PL:
    Wczytuje plik YAML z definicją kolorów raportu i zwraca sparsowany
    słownik ze wszystkimi sekcjami konfiguracji.

    Argumenty:
        theme_file (Path): Ścieżka do pliku YAML z kolorami.

    Zwraca:
        dict: Sparsowana zawartość YAML z konfiguracją kolorów.

    Wyjątki:
        FileNotFoundError: Gdy plik motywu nie istnieje.
        ValueError: Gdy plik YAML jest pusty lub niepoprawny.
    """
    # Sprawdzamy, czy plik motywu istnieje w systemie plików.
    if not theme_file.exists():
        raise FileNotFoundError(
            f"Theme file does not exist: {theme_file}"
        )

    # Otwieramy plik YAML w trybie tekstowym UTF-8.
    with theme_file.open("r", encoding="utf-8") as file_obj:
        # Parsujemy zawartość YAML do słownika Pythona.
        theme_colors = yaml.safe_load(file_obj)

    # Walidujemy, czy YAML zwrócił niepustą strukturę danych.
    if not isinstance(theme_colors, dict) or not theme_colors:
        raise ValueError("Theme YAML content is empty or invalid.")
    # Zwracamy pełny słownik kolorów motywu.
    return theme_colors


def load_palette(theme_file: Path = DEFAULT_THEME_FILE) -> list[str]:
    """Load chart palette list from theme configuration.

    EN:
    Extracts `chart` section from the YAML theme and returns it as list of
    color strings.

    Args:
        theme_file (Path): Path to YAML file with colors.

    Returns:
        list[str]: List of colors used for charts.

    Raises:
        ValueError: If `chart` section is missing or has wrong type.

    PL:
    Wyciąga sekcję `chart` z motywu YAML i zwraca ją jako listę
    łańcuchów kolorów.

    Argumenty:
        theme_file (Path): Ścieżka do pliku YAML z kolorami.

    Zwraca:
        list[str]: Lista kolorów używanych na wykresach.

    Wyjątki:
        ValueError: Gdy sekcja `chart` nie istnieje lub ma zły typ.
    """
    # Wczytujemy pełną konfigurację kolorów z pliku YAML.
    theme_colors = load_theme_colors(theme_file=theme_file)
    # Pobieramy sekcję wykresową z konfiguracji.
    chart_palette = theme_colors.get("chart")

    # Sprawdzamy, czy paleta ma poprawny format listy.
    if not isinstance(chart_palette, list) or not chart_palette:
        raise ValueError("Theme field 'chart' must be a non-empty list.")
    # Zwracamy gotową listę kolorów.
    return chart_palette


def set_plot_style(theme_file: Path = DEFAULT_THEME_FILE) -> None:
    """Apply global seaborn style and palette from theme file.

    EN:
    Sets seaborn whitegrid style and applies custom palette loaded from
    the provided YAML theme file.

    Args:
        theme_file (Path): Path to YAML file with colors.

    Returns:
        None: Updates seaborn global style state.

    PL:
    Ustawia styl seaborn `whitegrid` i nakłada własną paletę kolorów
    odczytaną z podanego pliku motywu YAML.

    Argumenty:
        theme_file (Path): Ścieżka do pliku YAML z kolorami.

    Zwraca:
        None: Aktualizuje globalny styl biblioteki seaborn.
    """
    # Ustawiamy bazowy styl siatki dla wszystkich wykresów.
    sns.set_style("whitegrid")
    # Ustawiamy domyślną paletę kolorów ze wskazanego motywu.
    sns.set_palette(load_palette(theme_file=theme_file))
