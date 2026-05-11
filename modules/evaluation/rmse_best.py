# RMSE bootstrap evaluation entrypoint for BEST configuration.
"""Run bootstrap RMSE evaluation for BEST model configuration.

EN:
Builds evaluation config for BEST and delegates execution to the shared
bootstrap RMSE routine.

PL:
Buduje konfigurację ewaluacji dla BEST i deleguje wykonanie do
współdzielonej procedury RMSE z bootstrapem.
"""
import sys
from pathlib import Path

from modules.evaluation.rmse_common import RMSEEvalConfig
from modules.evaluation.rmse_common import evaluate_bootstrap_rmse

# Wyznaczamy katalog główny projektu na podstawie ścieżki pliku.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Dopisujemy katalog projektu do sys.path, jeśli jeszcze go tam nie ma.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ustawiamy źródłowy plik treningowy dla bootstrapu.
TRAIN_FILE = "data/ratings.csv"
# Ustawiamy plik referencyjny do porównań RMSE.
TEST_FILE = "data/ratings.csv"
# Ustawiamy bazową ścieżkę zapisu modeli dla kolejnych runów.
MODEL_PATH = "models_trained/model_BEST_rmse.pkl"
# Ustawiamy bazową ścieżkę zapisu predykcji dla kolejnych runów.
PRED_FILE = "results/preds_best_rmse.csv"
# Ustawiamy ścieżkę pliku tymczasowego z parami user-film.
PAIRS_FILE = "results/_pairs_for_rmse.csv"
# Ustawiamy docelową ścieżkę pliku z podsumowaniem runów.
RESULTS_PKL = "results/best_rmse_runs.pkl"
# Ustawiamy liczbę powtórzeń bootstrap.
N_RUNS = 10
# Ustawiamy bazowe ziarno losowe dla bootstrapu.
BOOTSTRAP_BASE_SEED = 123


def main() -> None:
    """Run bootstrap RMSE experiment for BEST configuration.

    EN:
    Creates `RMSEEvalConfig` for BEST and runs shared bootstrap
    evaluation routine.

    Returns:
        None: Evaluation artifacts are written to disk.

    PL:
    Tworzy konfigurację `RMSEEvalConfig` dla BEST i uruchamia wspólną
    procedurę ewaluacji bootstrap.

    Zwraca:
        None: Artefakty ewaluacji są zapisywane na dysku.
    """
    # Tworzymy konfigurację uruchomienia dla algorytmu SGD.
    config = RMSEEvalConfig(
        alg="BEST",
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        model_path=MODEL_PATH,
        pred_file=PRED_FILE,
        pairs_file=PAIRS_FILE,
        results_pkl=RESULTS_PKL,
        n_runs=N_RUNS,
        bootstrap_base_seed=BOOTSTRAP_BASE_SEED,
    )
    # Delegujemy wykonanie do współdzielonej funkcji ewaluacji.
    evaluate_bootstrap_rmse(PROJECT_ROOT, config)


if __name__ == "__main__":
    # Uruchamiamy punkt wejścia skryptu przy bezpośrednim wykonaniu.
    main()
