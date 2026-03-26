# Entry point CLI module that routes execution to training or prediction.
import argparse
import os
import sys

from modules.train import train_model
from modules.predict import predict_ratings


# Lista dostępnych trybów uruchomienia aplikacji.
ALLOWED_MODES = ["train", "predict"]
# Lista wspieranych nazw algorytmów rekomendacji.
ALLOWED_ALGS = ["NMF", "SVD1", "SVD2", "SGD", "BEST"]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the recommendation CLI.

    EN:
    Build and validate the CLI interface definition and return parsed
    arguments used by training and prediction workflows.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    PL:
    Buduje oraz waliduje definicję interfejsu CLI i zwraca sparsowane
    argumenty używane w przepływach trenowania i predykcji.

    Zwraca:
        argparse.Namespace: Sparsowane argumenty wiersza poleceń.
    """
    # Tworzymy parser odpowiedzialny za analizę argumentów CLI.
    parser = argparse.ArgumentParser(
        description="Movie recommendation system - Project 1"
    )

    # Definiujemy obowiązkowy tryb działania programu.
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=ALLOWED_MODES,
        help="Mode of operation: train or predict",
    )

    # Definiujemy opcjonalną ścieżkę danych treningowych.
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to training data file (used in train mode)",
    )

    # Definiujemy opcjonalną ścieżkę pliku wejściowego do predykcji.
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file for prediction (used in predict mode)",
    )

    # Definiujemy obowiązkową ścieżkę modelu do zapisu lub odczytu.
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to save/load trained model",
    )

    # Definiujemy opcjonalną ścieżkę pliku wyjściowego dla predykcji.
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save predictions (used in predict mode)",
    )

    # Definiujemy obowiązkowy wybór algorytmu.
    parser.add_argument(
        "--alg",
        type=str,
        required=True,
        choices=ALLOWED_ALGS,
        help="Algorithm to use: NMF, SVD1, SVD2, SGD, BEST",
    )

    # Zwracamy gotowy obiekt z wartościami argumentów przekazanych w CLI.
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate required argument combinations for each execution mode.

    EN:
    Ensure that required file paths are present depending on selected mode.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ValueError: If required arguments are missing for a selected mode.

    Returns:
        None: This function only validates input and does not return data.

    PL:
    Zapewnia obecność wymaganych ścieżek plików zależnie od
    wybranego trybu.

    Argumenty:
        args (argparse.Namespace): Sparsowane argumenty wiersza poleceń.

    Wyjątki:
        ValueError: Gdy brakuje wymaganych argumentów dla wybranego trybu.

    Zwraca:
        None: Funkcja tylko waliduje dane wejściowe i nic nie zwraca.
    """
    # Dla trybu treningu wymagamy podania pliku treningowego.
    if args.mode == "train":
        # Jeśli użytkownik nie podał pliku, zgłaszamy błąd walidacji.
        if args.train_file is None:
            raise ValueError("In train mode, --train_file is required.")

    # Dla trybu predykcji wymagamy wejścia i pliku wyjściowego.
    if args.mode == "predict":
        # Bez pliku wejściowego nie da się wygenerować predykcji.
        if args.input_file is None:
            raise ValueError("In predict mode, --input_file is required.")
        # Bez pliku wyjściowego nie ma gdzie zapisać wyników.
        if args.output_file is None:
            raise ValueError("In predict mode, --output_file is required.")


def ensure_parent_dir(path: str) -> None:
    """Create a parent directory for a file path if it does not exist.

    EN:
    Create the immediate parent directory tree when a non-empty directory
    part is present in the provided file path.

    Args:
        path (str): File path whose parent directory should be ensured.

    Returns:
        None: Directory is created as a side effect when needed.

    PL:
    Tworzy katalog nadrzędny, gdy w podanej ścieżce pliku istnieje niepusta
    część katalogowa.

    Argumenty:
        path (str): Ścieżka pliku, dla której katalog nadrzędny ma istnieć.

    Zwraca:
        None: Katalog jest tworzony jako efekt uboczny, jeśli potrzeba.
    """
    # Pobieramy fragment ścieżki odpowiadający katalogowi nadrzędnemu.
    parent_dir = os.path.dirname(path)
    # Tworzymy katalog tylko wtedy, gdy część katalogowa faktycznie istnieje.
    if parent_dir:
        # exist_ok=True zapobiega błędowi, jeśli katalog już istnieje.
        os.makedirs(parent_dir, exist_ok=True)


def main() -> None:
    """Run the application entry point for training or prediction.

    EN:
    Parse and validate arguments, dispatch execution to selected mode, and
    report errors to stderr with a non-zero exit code.

    Raises:
        SystemExit: Terminates the process with status code 1 on failure.

    Returns:
        None: The function controls program flow and prints status messages.

    PL:
    Parsuje i waliduje argumenty, uruchamia wybrany tryb działania oraz
    raportuje błędy do stderr z niezerowym kodem zakończenia.

    Wyjątki:
        SystemExit: Kończy proces kodem 1 w przypadku niepowodzenia.

    Zwraca:
        None: Funkcja steruje przebiegiem programu i wypisuje komunikaty.
    """
    # Cały przebieg zamykamy w bloku obsługi błędów.
    try:
        # Parsujemy argumenty podane przy uruchomieniu programu.
        args = parse_arguments()
        # Sprawdzamy, czy zestaw argumentów jest logicznie poprawny.
        validate_arguments(args)

        # Upewniamy się, że katalog modelu istnieje przed użyciem ścieżki.
        ensure_parent_dir(args.model_path)

        # Rozpoczynamy ścieżkę treningu, jeśli wybrano tryb "train".
        if args.mode == "train":
            # Logujemy podstawowe informacje diagnostyczne o uruchomieniu.
            print(f"[INFO] Training started using algorithm: {args.alg}")
            print(f"[INFO] Training file: {args.train_file}")
            print(f"[INFO] Model will be saved to: {args.model_path}")

            # Wywołujemy funkcję treningu z odpowiednimi argumentami.
            train_model(
                train_file=args.train_file,
                model_path=args.model_path,
                alg=args.alg,
            )

            # Informujemy o poprawnym zakończeniu treningu.
            print("[INFO] Training finished successfully.")

        # Rozpoczynamy ścieżkę predykcji, jeśli wybrano tryb "predict".
        elif args.mode == "predict":
            # Tworzymy katalog na plik wyjściowy predykcji.
            ensure_parent_dir(args.output_file)

            # Logujemy informacje diagnostyczne o predykcji.
            print(f"[INFO] Prediction started using algorithm: {args.alg}")
            print(f"[INFO] Input file: {args.input_file}")
            print(f"[INFO] Model will be loaded from: {args.model_path}")
            print(f"[INFO] Predictions will be saved to: {args.output_file}")

            # Wywołujemy moduł predykcji z wymaganymi danymi.
            predict_ratings(
                input_file=args.input_file,
                model_path=args.model_path,
                output_file=args.output_file,
                alg=args.alg,
            )

            # Informujemy o poprawnym zakończeniu predykcji.
            print("[INFO] Prediction finished successfully.")

        else:
            # Dodatkowa ochrona przy nieobsługiwanym trybie działania.
            raise ValueError(f"Unsupported mode: {args.mode}")

    # Przechwytujemy dowolny wyjątek i kończymy program kodem błędu.
    except Exception as e:
        # Wypisujemy szczegóły błędu na standardowe wyjście błędów.
        print(f"[ERROR] {e}", file=sys.stderr)
        # Zwracamy kod 1, aby system wiedział, że wystąpił błąd.
        sys.exit(1)


# Standardowy punkt wejścia uruchamiany przy bezpośrednim wykonaniu pliku.
if __name__ == "__main__":
    # Przekazujemy sterowanie do głównej funkcji programu.
    main()
