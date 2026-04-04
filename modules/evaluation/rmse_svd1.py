import sys
from pathlib import Path

from modules.evaluation.rmse_common import RMSEEvalConfig
from modules.evaluation.rmse_common import evaluate_bootstrap_rmse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_FILE = "data/ratings.csv"
TEST_FILE = "data/ratings.csv"
MODEL_PATH = "models_trained/model_SVD1_rmse.pkl"
PRED_FILE = "results/preds_svd1_rmse.csv"
PAIRS_FILE = "results/_pairs_for_rmse.csv"
RESULTS_PKL = "results/svd1_rmse_runs.pkl"
N_RUNS = 10
BOOTSTRAP_BASE_SEED = 123


def main() -> None:
    """Run bootstrap RMSE experiment for SVD1 configuration.

    EN:
    Builds evaluation config for SVD1 and delegates execution to the shared
    bootstrap RMSE routine.

    Returns:
        None: Evaluation artifacts are written to disk.

    PL:
    Uruchamia eksperyment bootstrap RMSE dla konfiguracji SVD1.

    Buduje konfigurację ewaluacji dla SVD1 i przekazuje wykonanie do
    współdzielonej procedury obliczania RMSE.

    Zwraca:
        None: Artefakty ewaluacji są zapisywane na dysk.
    """
    config = RMSEEvalConfig(
        alg="SVD1",
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        model_path=MODEL_PATH,
        pred_file=PRED_FILE,
        pairs_file=PAIRS_FILE,
        results_pkl=RESULTS_PKL,
        n_runs=N_RUNS,
        bootstrap_base_seed=BOOTSTRAP_BASE_SEED,
    )
    evaluate_bootstrap_rmse(PROJECT_ROOT, config)


if __name__ == "__main__":
    main()
